/**
 * Failure modes a caller might reasonably want to tell apart.
 *
 * Deliberately few, and deliberately the same set the Python client raises. The
 * distinctions that exist are the ones that change what a service should do next: retry,
 * give up, or resubmit.
 *
 * Fields are assigned in the constructor body rather than declared as TypeScript parameter
 * properties. Parameter properties are one of the few TS features that need a real
 * transform rather than type erasure, so avoiding them keeps this package loadable by
 * anything that merely strips types — Node's `--experimental-strip-types`, Bun, Deno,
 * esbuild — as well as by `tsc`.
 */

export class SirClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/**
 * No endpoint is configured for the requested model.
 *
 * A configuration mistake, not a runtime condition — thrown before anything is sent.
 */
export class ModelNotRouted extends SirClientError {}

/** A non-2xx response that isn't a recognised job condition. */
export class TransportError extends SirClientError {
  readonly status: number;
  readonly body: unknown;

  constructor(status: number, body: unknown) {
    super(`HTTP ${status}: ${typeof body === "string" ? body : JSON.stringify(body)}`);
    this.status = status;
    this.body = body;
  }
}

/**
 * A job vanished mid-poll.
 *
 * Either its result expired before it was read, or the router restarted and lost the
 * in-memory store. The two are indistinguishable from out here, which is why this is
 * thrown rather than silently resubmitted: replaying a request that may already have run
 * is the caller's decision, not the SDK's. Pass `resubmitOnLoss` with an `idempotencyKey`
 * to opt into it.
 */
export class JobLost extends SirClientError {
  readonly jobId: string;

  constructor(jobId: string) {
    super(`job '${jobId}' is gone; it expired or the router restarted`);
    this.jobId = jobId;
  }
}

/** The router accepted the request and generation failed. */
export class JobFailed extends SirClientError {
  readonly jobId: string;
  readonly error: unknown;

  constructor(jobId: string, error: unknown) {
    super(`job '${jobId}' failed: ${JSON.stringify(error)}`);
    this.jobId = jobId;
    this.error = error;
  }
}

/** The job was cancelled — by this client, by another, or by lease expiry. */
export class JobCancelled extends SirClientError {
  readonly jobId: string;

  constructor(jobId: string) {
    super(`job '${jobId}' was cancelled`);
    this.jobId = jobId;
  }
}

/**
 * The caller's deadline passed before the job finished.
 *
 * The job is cancelled on the way out, so a timeout stops costing GPU time rather than
 * leaving orphaned work behind.
 */
export class RequestTimeout extends SirClientError {
  readonly jobId: string;
  readonly timeoutMs: number;

  constructor(jobId: string, timeoutMs: number) {
    super(`job '${jobId}' did not finish within ${timeoutMs}ms`);
    this.jobId = jobId;
    this.timeoutMs = timeoutMs;
  }
}
