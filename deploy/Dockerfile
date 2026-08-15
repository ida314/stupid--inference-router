# The router itself. Small on purpose — it holds no weights and does no CUDA; the GPU
# lives entirely behind Triton, on the other side of the backend interface.
FROM python:3.12-slim

# Pinned rather than :latest, so a rebuild six months from now resolves the same way.
COPY --from=ghcr.io/astral-sh/uv:0.11.19 /uv /usr/local/bin/uv

WORKDIR /app

# The workspace member has to be present for the lockfile to resolve even though
# `--no-dev` means it is never installed: `sir-client` is declared as a workspace source,
# and uv validates the whole workspace.
COPY pyproject.toml uv.lock README.md ./
COPY clients/python ./clients/python
COPY src ./src

# `--frozen` is the point of committing the lock: the build fails rather than quietly
# resolving a different FastAPI than the one the tests ran against.
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# One process, no `--workers`. The scheduler owns exclusive model residency and holds the
# job store in memory; a second worker would run a second scheduler and the two would
# fight over the same GPU while disagreeing about who is resident.
#
# The console script rather than `uv run`, so uvicorn is PID 1 and receives SIGTERM
# directly — that signal is what runs the lifespan shutdown that drains jobs and stops the
# engine. Through a wrapper the container would be SIGKILLed at the stop timeout instead.
CMD ["sir", "serve", "-c", "/etc/sir/config.yaml"]
