/**
 * Which host serves which model.
 *
 * This is the whole of the SDK's multi-backend story, and it is deliberately the whole of
 * it. The registry maps a model tag to a base URL; it does not know whether that URL is a
 * `sir`, a vLLM, or an SGLang, and it never rewrites a request body to suit one.
 *
 * That restraint is inherited from the router. `sir` refuses to translate between provider
 * dialects because the extras those projects accept — `guided_json` against `json_schema`,
 * `ebnf` against `guided_grammar` — drift with every release, and anything that parses
 * them has to be updated in lockstep. Moving that job into an SDK would not make the drift
 * go away; it would turn one process that needs updating into every service that depends
 * on this package. So bodies are forwarded exactly as the caller wrote them, and the
 * knowledge of what a given backend accepts stays in the service that already has it.
 */

import { ModelNotRouted, TransportError } from "./errors.ts";

export const ENV_BASE_URL = "SIR_BASE_URL";
export const ENV_ENDPOINTS = "SIR_ENDPOINTS";

/** Just enough of `fetch` to be swappable in tests. */
export type FetchLike = (
  input: string,
  init?: { method?: string; headers?: Record<string, string>; body?: string; signal?: AbortSignal },
) => Promise<Response>;

export class Registry {
  readonly #endpoints = new Map<string, string>();
  readonly #default: string | undefined;

  constructor(endpoints: Record<string, string> = {}, defaultBaseUrl?: string) {
    for (const [model, url] of Object.entries(endpoints)) {
      this.#endpoints.set(model, trim(url));
    }
    this.#default = defaultBaseUrl ? trim(defaultBaseUrl) : undefined;
  }

  /**
   * Build from `SIR_BASE_URL` and `SIR_ENDPOINTS`.
   *
   * `SIR_ENDPOINTS` is `model=url` pairs separated by commas, for when not everything
   * lives behind one router:
   *
   *     SIR_ENDPOINTS='Qwen/Qwen3-8B=http://gpu:8000,bge-m3=http://cpu:8001'
   *
   * The environment is a parameter so this stays usable where `process` doesn't exist.
   */
  static fromEnv(env: Record<string, string | undefined> = ambientEnv()): Registry {
    const endpoints: Record<string, string> = {};
    for (const raw of (env[ENV_ENDPOINTS] ?? "").split(",")) {
      const pair = raw.trim();
      if (!pair) continue;
      const at = pair.indexOf("=");
      const model = at === -1 ? "" : pair.slice(0, at).trim();
      const url = at === -1 ? "" : pair.slice(at + 1).trim();
      if (!model || !url) {
        throw new Error(`malformed ${ENV_ENDPOINTS} entry '${pair}'; expected 'model=url'`);
      }
      endpoints[model] = url;
    }
    return new Registry(endpoints, env[ENV_BASE_URL]);
  }

  register(model: string, baseUrl: string): void {
    this.#endpoints.set(model, trim(baseUrl));
  }

  /** The base URL serving `model`. Throws before anything is sent if there isn't one. */
  resolve(model: string): string {
    const url = this.#endpoints.get(model) ?? this.#default;
    if (url === undefined) {
      const known = this.models.join(", ") || "none";
      throw new ModelNotRouted(
        `no endpoint configured for model '${model}'; known models: ${known}. ` +
          `Set ${ENV_BASE_URL} for a catch-all, or ${ENV_ENDPOINTS} for per-model routing.`,
      );
    }
    return url;
  }

  /**
   * Populate the map by asking each host what it serves, via `/v1/models`.
   *
   * Two hosts claiming the same tag is rejected rather than resolved by ordering, for the
   * same reason the router rejects a duplicated `served_model_name`: routing that depends
   * on the order things were listed in is routing that breaks silently.
   */
  async discover(baseUrls: Iterable<string>, fetchImpl: FetchLike = fetch): Promise<this> {
    for (const baseUrl of baseUrls) {
      const base = trim(baseUrl);
      const response = await fetchImpl(`${base}/v1/models`);
      if (!response.ok) {
        throw new TransportError(response.status, await bodyOf(response));
      }
      const payload = (await response.json()) as { data?: { id?: string }[] };
      for (const card of payload.data ?? []) {
        if (!card.id) continue;
        const claimed = this.#endpoints.get(card.id);
        if (claimed !== undefined && claimed !== base) {
          throw new Error(
            `model '${card.id}' is served by both ${claimed} and ${base}; ` +
              "routing would depend on discovery order",
          );
        }
        this.#endpoints.set(card.id, base);
      }
    }
    return this;
  }

  get models(): string[] {
    return [...this.#endpoints.keys()].sort();
  }
}

/**
 * `process.env` where there is one, an empty map where there isn't.
 *
 * Reached for through `globalThis` rather than imported, so this package needs no Node
 * type definitions and stays loadable in a browser or a worker.
 */
function ambientEnv(): Record<string, string | undefined> {
  const host = globalThis as { process?: { env?: Record<string, string | undefined> } };
  return host.process?.env ?? {};
}

function trim(url: string): string {
  return url.replace(/\/+$/, "");
}

export async function bodyOf(response: Response): Promise<unknown> {
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}
