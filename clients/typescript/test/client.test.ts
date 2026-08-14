/**
 * Client behaviour that is about the protocol rather than the router, driven through the
 * injectable `fetch` so each case can be set up exactly.
 *
 * The cases that need a real scheduler underneath live in `contract.test.ts`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Client, JobLost, TransportError } from "../src/index.ts";
import type { FetchLike } from "../src/index.ts";

const COMPLETION = {
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 0,
  model: "chat",
  choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" }],
  usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
};

const BODY = { messages: [{ role: "user", content: "hello" }] };

function json(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function clientWith(handler: FetchLike): Client {
  return new Client({ baseUrl: "http://router", fetch: handler });
}

test("a synchronous endpoint needs no special handling", async () => {
  // The compatibility story: a plain vLLM ignores `Prefer` and answers 200 with the
  // completion. `202` means you got a job; `200` means you got the answer. Nothing is
  // probed and nothing is configured.
  const client = clientWith(async () => json(COMPLETION));

  assert.deepEqual(await client.run("chat", BODY), COMPLETION);
});

test("the request body is forwarded as written", async () => {
  let sent: Record<string, unknown> = {};
  const client = clientWith(async (_url, init) => {
    sent = JSON.parse(init!.body as string) as Record<string, unknown>;
    return json(COMPLETION);
  });

  await client.run("chat", {
    ...BODY,
    top_k: 40,
    guided_json: { type: "object" },
    some_field_invented_next_release: true,
  });

  // Provider extras are exactly what a translating client would mangle, so they are what
  // this checks. Only `model` is set, so routing and payload cannot disagree.
  assert.equal(sent["top_k"], 40);
  assert.deepEqual(sent["guided_json"], { type: "object" });
  assert.equal(sent["some_field_invented_next_release"], true);
  assert.equal(sent["model"], "chat");
});

test("the async preference is what asks for a job", async () => {
  let headers: Record<string, string> = {};
  const client = clientWith(async (_url, init) => {
    headers = init!.headers as Record<string, string>;
    return json(COMPLETION);
  });

  await client.run("chat", BODY);

  assert.equal(headers["prefer"], "respond-async");
});

test("polling follows the cadence the server asks for", async () => {
  const polls: number[] = [];
  let last = Date.now();
  let reads = 0;

  const client = clientWith(async (url, init) => {
    if (init?.method === "POST") {
      return json({ id: "req_1", status: "queued", retry_after: 0.06 }, 202);
    }
    polls.push(Date.now() - last);
    last = Date.now();
    reads += 1;
    return reads < 3
      ? json({ id: "req_1", status: "running", retry_after: 0.06 })
      : json({ id: "req_1", status: "done", response: COMPLETION, retry_after: 0 });
  });

  await client.run("chat", BODY);

  assert.equal(reads, 3);
  // 60ms advised, so each gap should be at least that and nowhere near a hardcoded second.
  for (const gap of polls) {
    assert.ok(gap >= 50, `polled after ${gap}ms, faster than advised`);
    assert.ok(gap < 500, `polled after ${gap}ms, ignoring the advice`);
  }
});

test("a vanished job is raised, not silently replayed", async () => {
  // Resubmitting work that may already have run is the caller's call, not the SDK's.
  let submissions = 0;
  const client = clientWith(async (_url, init) => {
    if (init?.method === "POST") {
      submissions += 1;
      return json({ id: "req_1", status: "queued", retry_after: 0.01 }, 202);
    }
    return json({ error: { message: "gone" } }, 404);
  });

  await assert.rejects(client.run("chat", BODY), JobLost);
  assert.equal(submissions, 1);
});

test("resubmission after a loss is opt-in and needs a key", async () => {
  let submissions = 0;
  const client = clientWith(async (_url, init) => {
    if (init?.method === "POST") {
      submissions += 1;
      return json({ id: `req_${submissions}`, status: "queued", retry_after: 0.01 }, 202);
    }
    return submissions < 2
      ? json({ error: { message: "gone" } }, 404)
      : json({ id: "req_2", status: "done", response: COMPLETION, retry_after: 0 });
  });

  const completion = await client.run("chat", BODY, {
    idempotencyKey: "k",
    resubmitOnLoss: true,
  });

  assert.deepEqual(completion, COMPLETION);
  assert.equal(submissions, 2);
});

test("resubmission does not happen without an idempotency key", async () => {
  let submissions = 0;
  const client = clientWith(async (_url, init) => {
    if (init?.method === "POST") {
      submissions += 1;
      return json({ id: "req_1", status: "queued", retry_after: 0.01 }, 202);
    }
    return json({ error: { message: "gone" } }, 404);
  });

  // Without a key the router cannot collapse the duplicate, so replaying would risk
  // paying for the same generation twice.
  await assert.rejects(client.run("chat", BODY, { resubmitOnLoss: true }), JobLost);
  assert.equal(submissions, 1);
});

test("an unexpected status is surfaced rather than guessed at", async () => {
  const client = clientWith(async () => json({ error: { message: "boom" } }, 500));

  await assert.rejects(client.run("chat", BODY), (error: unknown) => {
    assert.ok(error instanceof TransportError);
    assert.equal(error.status, 500);
    return true;
  });
});
