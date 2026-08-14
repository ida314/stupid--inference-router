/**
 * One call that returns a completion, however long the router had to queue it.
 *
 * The problem this exists to solve is that a swap-scheduled GPU can leave a request
 * waiting minutes before it starts generating, and holding an HTTP connection open that
 * long is a good way to discover every idle timeout between a service and the router. So
 * `sir` accepts work asynchronously and hands back a job to poll, and this hides that:
 * `run` looks like an ordinary request/response call and there is no queue in the
 * signature.
 *
 * Three things it is careful about, in rough order of how much they matter:
 *
 * 1. **Cancellation reaches the GPU.** Abort the signal, or let the deadline pass, and the
 *    job is cancelled too. Holding a socket open is what tells the router today that a
 *    client still wants its answer; once nobody is holding one, this is what carries that
 *    signal, and without it an abandoned request keeps a model resident for a response no
 *    one will read.
 * 2. **Poll cadence comes from the server.** The router knows the queue depth, whether a
 *    swap is pending, and how long its requests have been taking. Its `retry_after` is a
 *    better guess than any interval hardcoded here.
 * 3. **A vanished job is not silently retried.** Resubmitting a request that may already
 *    have run is a decision with a cost, so it is the caller's to make.
 *
 * It is also careful about what it does *not* do: it never reads or rewrites the request
 * body. See `registry.ts` for why.
 */

import {
  JobCancelled,
  JobFailed,
  JobLost,
  RequestTimeout,
  TransportError,
} from "./errors.ts";
import { Registry, bodyOf, type FetchLike } from "./registry.ts";
import type { ChatCompletion, ChatRequestBody, JobDocument, JobStatus, WaitInfo } from "./types.ts";

/**
 * Bounds on how long to wait between polls, whatever the server suggests. The floor keeps
 * a bad `retry_after` from turning into a busy loop; the ceiling keeps a finished response
 * from sitting unread for long enough to hit its retention window.
 */
const MIN_POLL_MS = 50;
const MAX_POLL_MS = 10_000;

/**
 * How many times `run` will resubmit after a lost job, when asked to. One retry covers a
 * router restart; more would just replay the request into a router that is still down.
 */
const MAX_ATTEMPTS = 2;

export interface ClientOptions {
  registry?: Registry | undefined;
  baseUrl?: string | undefined;
  endpoints?: Record<string, string> | undefined;
  /** Swappable for tests; defaults to the global `fetch`. */
  fetch?: FetchLike | undefined;
  /** Default deadline applied when a call doesn't set its own. */
  timeoutMs?: number | undefined;
  maxPollMs?: number | undefined;
}

export interface CallOptions {
  signal?: AbortSignal | undefined;
  timeoutMs?: number | undefined;
  idempotencyKey?: string | undefined;
  /** Only honoured together with `idempotencyKey`, which is what makes it safe. */
  resubmitOnLoss?: boolean | undefined;
}

/** A handle on work the router has accepted. */
export class Job {
  /**
   * Set when the endpoint answered synchronously — a plain vLLM, or `sir` with the async
   * path disabled. There is nothing to poll, but the caller gets the same handle either
   * way rather than two shapes to branch on.
   */
  completion: ChatCompletion | null = null;
  document: JobDocument | null = null;

  readonly id: string;
  readonly model: string;
  readonly baseUrl: string;
  readonly #client: Client;

  constructor(id: string, model: string, baseUrl: string, client: Client) {
    this.id = id;
    this.model = model;
    this.baseUrl = baseUrl;
    this.#client = client;
  }

  get status(): JobStatus {
    if (this.completion !== null) return "done";
    return this.document?.status ?? "queued";
  }

  /** The router's own account of the wait: position, residency, swap, estimate. */
  get wait(): WaitInfo | null {
    return this.document?.wait ?? null;
  }

  /** Re-read the job, renewing its lease. Throws `JobLost` if it is gone. */
  async refresh(options: CallOptions = {}): Promise<JobDocument | null> {
    if (this.completion !== null) return this.document;
    this.document = await this.#client._fetchJob(this.baseUrl, this.id, options.signal);
    return this.document;
  }

  async result(options: CallOptions = {}): Promise<ChatCompletion> {
    if (this.completion !== null) return this.completion;
    return this.#client._awaitJob(this, options);
  }

  async cancel(): Promise<void> {
    if (this.completion === null) await this.#client._cancelJob(this.baseUrl, this.id);
  }
}

/** Talks to one or more inference endpoints. Safe to share across concurrent callers. */
export class Client {
  readonly registry: Registry;
  readonly #fetch: FetchLike;
  readonly #timeoutMs: number | undefined;
  readonly #maxPollMs: number;

  constructor(options: ClientOptions = {}) {
    this.registry =
      options.registry ??
      (options.endpoints || options.baseUrl
        ? new Registry(options.endpoints ?? {}, options.baseUrl)
        : Registry.fromEnv());
    this.#fetch = options.fetch ?? ((input, init) => fetch(input, init));
    this.#timeoutMs = options.timeoutMs;
    this.#maxPollMs = options.maxPollMs ?? MAX_POLL_MS;
  }

  // ---------------------------------------------------------------- the common case

  /**
   * Submit `body` for `model` and return the completion.
   *
   * `body` is forwarded as written, except that `model` is set from the argument so
   * routing and payload cannot disagree. Whatever extras the target backend accepts, put
   * them in `body`; nothing here inspects them.
   */
  async run(
    model: string,
    body: ChatRequestBody,
    options: CallOptions = {},
  ): Promise<ChatCompletion> {
    const attempts = options.resubmitOnLoss && options.idempotencyKey ? MAX_ATTEMPTS : 1;
    const deadline = this.#deadline(options);

    for (let attempt = 0; attempt < attempts; attempt++) {
      const job = await this.submit(model, body, options);
      try {
        return await this._awaitJob(job, options, deadline);
      } catch (error) {
        if (!(error instanceof JobLost) || attempt === attempts - 1) throw error;
      }
    }
    throw new Error("unreachable");
  }

  /** Accept-and-return-a-handle, for work the caller doesn't want to wait on. */
  async submit(
    model: string,
    body: ChatRequestBody,
    options: CallOptions = {},
  ): Promise<Job> {
    const baseUrl = this.registry.resolve(model);
    const headers: Record<string, string> = {
      "content-type": "application/json",
      prefer: "respond-async",
    };
    if (options.idempotencyKey) headers["idempotency-key"] = options.idempotencyKey;

    const response = await this.#fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers,
      body: JSON.stringify({ ...body, model }),
      ...(options.signal ? { signal: options.signal } : {}),
    });

    // 202 means the preference was honoured. 200 means it wasn't — a plain vLLM ignoring
    // an unknown header, or `sir` with jobs turned off — and the body is already the
    // answer. Branching on the status code is the entire compatibility story: no probing,
    // no configuration, no per-backend client.
    if (response.status === 200) {
      const job = new Job("", model, baseUrl, this);
      job.completion = (await response.json()) as ChatCompletion;
      return job;
    }
    if (response.status !== 202) {
      throw new TransportError(response.status, await bodyOf(response));
    }

    const document = (await response.json()) as JobDocument;
    const job = new Job(document.id, model, baseUrl, this);
    job.document = document;
    return job;
  }

  // ---------------------------------------------------------------- polling

  /** @internal */
  async _awaitJob(
    job: Job,
    options: CallOptions = {},
    deadline?: number | undefined,
  ): Promise<ChatCompletion> {
    if (job.completion !== null) return job.completion;
    const { signal } = options;
    const timeoutAt = deadline ?? this.#deadline(options);
    const budget = options.timeoutMs ?? this.#timeoutMs ?? 0;

    try {
      for (;;) {
        switch (job.status) {
          case "done":
            return job.document!.response!;
          case "failed":
            throw new JobFailed(job.id, job.document?.error);
          case "cancelled":
            throw new JobCancelled(job.id);
        }

        const remaining = timeoutAt === undefined ? undefined : timeoutAt - Date.now();
        if (remaining !== undefined && remaining <= 0) {
          await this.#cancelQuietly(job);
          throw new RequestTimeout(job.id, budget);
        }

        await sleep(this.#delay(job.document, remaining), signal);
        job.document = await this._fetchJob(job.baseUrl, job.id, signal);
      }
    } catch (error) {
      // The caller gave up. Tell the router before unwinding, so the GPU stops working on
      // an answer that now has nowhere to go.
      if (signal?.aborted) await this.#cancelQuietly(job);
      throw error;
    }
  }

  #delay(document: JobDocument | null, remaining: number | undefined): number {
    const suggested = document?.retry_after;
    const advised = typeof suggested === "number" ? suggested * 1000 : 1000;
    let delay = Math.min(this.#maxPollMs, Math.max(MIN_POLL_MS, advised));
    if (remaining !== undefined) {
      // Never sleep past the deadline; wake up in time to cancel the job.
      delay = Math.min(delay, Math.max(MIN_POLL_MS, remaining));
    }
    return delay;
  }

  #deadline(options: CallOptions): number | undefined {
    const budget = options.timeoutMs ?? this.#timeoutMs;
    return budget === undefined ? undefined : Date.now() + budget;
  }

  /** @internal */
  async _fetchJob(
    baseUrl: string,
    jobId: string,
    signal?: AbortSignal | undefined,
  ): Promise<JobDocument> {
    const response = await this.#fetch(`${baseUrl}/v1/jobs/${jobId}`, {
      ...(signal ? { signal } : {}),
    });
    if (response.status === 404) throw new JobLost(jobId);
    if (!response.ok) throw new TransportError(response.status, await bodyOf(response));
    return (await response.json()) as JobDocument;
  }

  /** @internal */
  async _cancelJob(baseUrl: string, jobId: string): Promise<void> {
    const response = await this.#fetch(`${baseUrl}/v1/jobs/${jobId}`, { method: "DELETE" });
    if (response.status !== 200 && response.status !== 404) {
      throw new TransportError(response.status, await bodyOf(response));
    }
  }

  /**
   * Best-effort cancel from a path that is already unwinding.
   *
   * Deliberately sent without the caller's signal: that signal is the thing that just
   * aborted, and reusing it would abort the one request whose entire purpose is to run
   * after the abort.
   */
  async #cancelQuietly(job: Job): Promise<void> {
    try {
      await this._cancelJob(job.baseUrl, job.id);
    } catch {
      // Nothing useful to do here — the caller is already receiving a more interesting
      // error, and the lease will collect the job if this didn't land.
    }
  }
}

function sleep(ms: number, signal?: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason as Error);
      return;
    }
    const onAbort = () => {
      cleanup();
      reject(signal!.reason as Error);
    };
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}
