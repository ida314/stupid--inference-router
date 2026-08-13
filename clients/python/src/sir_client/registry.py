"""Which host serves which model.

This is the whole of the SDK's multi-backend story, and it is deliberately the whole of
it. The registry maps a model tag to a base URL; it does not know whether that URL is a
`sir`, a vLLM, or an SGLang, and it never rewrites a request body to suit one.

That restraint is inherited from the router. `sir` refuses to translate between provider
dialects because the extras those projects accept — `guided_json` against `json_schema`,
`ebnf` against `guided_grammar` — drift with every release, and anything that parses them
has to be updated in lockstep. Moving that job into an SDK would not make the drift go
away; it would turn one process that needs updating into every service that depends on
this package. So bodies are forwarded exactly as the caller wrote them, and the knowledge
of what a given backend accepts stays in the service that already has it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

from sir_client.errors import ModelNotRouted, TransportError

ENV_BASE_URL = "SIR_BASE_URL"
ENV_ENDPOINTS = "SIR_ENDPOINTS"


class Registry:
    """Model tag to base URL, with an optional catch-all."""

    def __init__(
        self,
        endpoints: Mapping[str, str] | None = None,
        default: str | None = None,
    ) -> None:
        self._endpoints = {
            model: url.rstrip("/") for model, url in (endpoints or {}).items()
        }
        self._default = default.rstrip("/") if default else None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Registry:
        """Build from `SIR_BASE_URL` and `SIR_ENDPOINTS`.

        `SIR_ENDPOINTS` is `model=url` pairs separated by commas, for the case where not
        everything lives behind one router:

            SIR_ENDPOINTS='Qwen/Qwen3-8B=http://gpu:8000,bge-m3=http://cpu:8001'
        """
        env = os.environ if environ is None else environ
        endpoints: dict[str, str] = {}
        raw = env.get(ENV_ENDPOINTS, "")
        for pair in raw.split(","):
            pair = pair.strip()
            if not pair:
                continue
            model, _, url = pair.partition("=")
            if not model.strip() or not url.strip():
                raise ValueError(
                    f"malformed {ENV_ENDPOINTS} entry {pair!r}; expected 'model=url'"
                )
            endpoints[model.strip()] = url.strip()
        return cls(endpoints, default=env.get(ENV_BASE_URL))

    def register(self, model: str, base_url: str) -> None:
        self._endpoints[model] = base_url.rstrip("/")

    def resolve(self, model: str) -> str:
        """The base URL serving `model`. Raises before anything is sent if there isn't one."""
        url = self._endpoints.get(model) or self._default
        if url is None:
            known = ", ".join(sorted(self._endpoints)) or "none"
            raise ModelNotRouted(
                f"no endpoint configured for model {model!r}; "
                f"known models: {known}. Set {ENV_BASE_URL} for a catch-all, or "
                f"{ENV_ENDPOINTS} for per-model routing."
            )
        return url

    async def discover(self, http: object, base_urls: Iterable[str]) -> Registry:
        """Populate the map by asking each host what it serves, via `/v1/models`.

        Two hosts claiming the same tag is rejected rather than resolved by ordering, for
        the same reason the router rejects a duplicated `served_model_name`: routing that
        depends on the order things were listed in is routing that breaks silently.
        """
        for base_url in base_urls:
            base = base_url.rstrip("/")
            response = await http.get(f"{base}/v1/models")  # type: ignore[attr-defined]
            if response.status_code != 200:
                raise TransportError(response.status_code, response.text)
            for card in response.json().get("data", []):
                tag = card.get("id")
                if not tag:
                    continue
                claimed = self._endpoints.get(tag)
                if claimed is not None and claimed != base:
                    raise ValueError(
                        f"model {tag!r} is served by both {claimed} and {base}; "
                        "routing would depend on discovery order"
                    )
                self._endpoints[tag] = base
        return self

    @property
    def models(self) -> list[str]:
        return sorted(self._endpoints)
