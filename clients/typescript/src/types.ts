/**
 * The wire shapes, mirroring `src/sir/schemas.py`.
 *
 * Request bodies are deliberately loose. `sir` reads `model` and `stream` and forwards
 * everything else untouched, so a type here that enumerated the sampling parameters would
 * be claiming knowledge this package has decided not to have — and would reject a field
 * the day a backend adds one. See `registry.ts` for the reasoning.
 */

/** Whatever your backend accepts. Only `messages` is required to be useful. */
export interface ChatRequestBody {
  messages: unknown[];
  [key: string]: unknown;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatChoice {
  index: number;
  message: { role: string; content: string | null; [key: string]: unknown };
  finish_reason: string | null;
}

export interface ChatCompletion {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatChoice[];
  usage: Usage;
}

/**
 * The router's account of a queued request's wait.
 *
 * Everything but `estimated_seconds` is read off scheduler state and is exact.
 */
export interface WaitInfo {
  /** Index in the model's queue among live requests. 0 means next to be dispatched. */
  position: number;
  resident: boolean;
  /** Whether the GPU has to change hands before this request can run at all. */
  needs_swap: boolean;
  /** What that swap costs, or 0 when the model is already resident. */
  load_seconds: number;
  /**
   * The starvation ceiling's remaining budget — a *head-of-queue* bound. It says when the
   * model is guaranteed the GPU, not when this request finishes. Behind fifty queued
   * requests, the model is served within this window and you are still fiftieth. `null`
   * when the model is already resident.
   */
  dispatch_within_seconds: number | null;
  /** Advisory projection over a running average of past durations. Never a deadline. */
  estimated_seconds: number;
}

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface ErrorBody {
  message: string;
  type: string;
  param: string | null;
  code: string | null;
}

export interface JobDocument {
  id: string;
  object: "sir.job";
  status: JobStatus;
  model: string;
  created: number;
  /** Present while queued, null once generating. */
  wait: WaitInfo | null;
  /** Seconds until the next poll. Always present, so a client never invents a cadence. */
  retry_after: number;
  response: ChatCompletion | null;
  error: ErrorBody | null;
}
