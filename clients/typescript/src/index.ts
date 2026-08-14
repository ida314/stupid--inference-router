/**
 * `sir-client` — call a model without caring whether it is loaded.
 *
 *     import { runLlm } from "sir-client";
 *
 *     const completion = await runLlm("Qwen/Qwen3-8B", {
 *       messages: [{ role: "user", content: "hello" }],
 *     });
 *
 * Endpoints come from `SIR_BASE_URL` (a catch-all) or `SIR_ENDPOINTS` (`model=url` pairs).
 * For a service that makes more than the occasional call, build a `Client` once and reuse
 * it — the module-level helpers exist for scripts and one-offs, and hold a shared client
 * built from the environment as it stood on first use.
 *
 * The surface mirrors the Python client deliberately. Two SDKs that behave the same way
 * mean a bug found in one is a bug found in both, and the shared contract tests in this
 * repo are what keep that true.
 */

export { Client, Job, type CallOptions, type ClientOptions } from "./client.ts";
export {
  ENV_BASE_URL,
  ENV_ENDPOINTS,
  Registry,
  type FetchLike,
} from "./registry.ts";
export {
  JobCancelled,
  JobFailed,
  JobLost,
  ModelNotRouted,
  RequestTimeout,
  SirClientError,
  TransportError,
} from "./errors.ts";
export type {
  ChatChoice,
  ChatCompletion,
  ChatRequestBody,
  ErrorBody,
  JobDocument,
  JobStatus,
  Usage,
  WaitInfo,
} from "./types.ts";

import { Client, type CallOptions } from "./client.ts";
import type { ChatCompletion, ChatRequestBody } from "./types.ts";
import type { Job } from "./client.ts";

let shared: Client | undefined;

/** The lazily-built shared client, configured from the environment. */
export function defaultClient(): Client {
  shared ??= new Client();
  return shared;
}

/** Drop the shared client, so the next call rebuilds it from the current environment. */
export function resetDefaultClient(): void {
  shared = undefined;
}

/** Submit a request and return the completion, queueing transparently. */
export function runLlm(
  model: string,
  body: ChatRequestBody,
  options: CallOptions = {},
): Promise<ChatCompletion> {
  return defaultClient().run(model, body, options);
}

/** Submit without waiting, for work to be collected later. */
export function submitLlm(
  model: string,
  body: ChatRequestBody,
  options: CallOptions = {},
): Promise<Job> {
  return defaultClient().submit(model, body, options);
}
