/**
 * The SDK against a real router.
 *
 * The Python client gets to drive the app in-process through an ASGI transport; there is
 * no cross-language equivalent, so this spawns the actual `sir serve` on a free port and
 * talks to it over loopback. Slower, and worth it: what these assert is the agreement
 * between two codebases, and a hand-written mock of the router would agree with itself
 * forever while drifting from the thing it stands in for.
 *
 * Skipped automatically when the Python side isn't installed, so `npm test` stays useful
 * to someone working on the client alone.
 */

import assert from "node:assert/strict";
import { spawn, type ChildProcess } from "node:child_process";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";
import { after, before, describe, test } from "node:test";

import { Client, Job, JobLost, RequestTimeout } from "../src/index.ts";
import type { JobDocument } from "../src/index.ts";

const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const CONFIG = fileURLToPath(new URL("./fixtures/router.yaml", import.meta.url));
const BODY = { messages: [{ role: "user", content: "hello" }] };

let router: ChildProcess | undefined;
let baseUrl = "";
let client: Client;

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      const port = typeof address === "object" && address ? address.port : 0;
      probe.close(() => resolve(port));
    });
  });
}

async function waitForHealth(url: string, deadlineMs = 30_000): Promise<boolean> {
  const until = Date.now() + deadlineMs;
  while (Date.now() < until) {
    if (router?.exitCode !== null && router?.exitCode !== undefined) return false;
    try {
      const response = await fetch(`${url}/healthz`);
      if (response.ok) return true;
    } catch {
      // Not listening yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  return false;
}

/** Read a job straight off the router, bypassing the client under test. */
async function readJob(id: string): Promise<JobDocument> {
  const response = await fetch(`${baseUrl}/v1/jobs/${id}`);
  return (await response.json()) as JobDocument;
}

describe("against a live router", { concurrency: false }, () => {
  let available = false;

  before(async () => {
    const port = await freePort();
    baseUrl = `http://127.0.0.1:${port}`;
    router = spawn("uv", ["run", "sir", "serve", "--config", CONFIG, "--port", String(port)], {
      cwd: REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
    });
    router.on("error", () => {
      available = false;
    });
    available = await waitForHealth(baseUrl);
    client = new Client({ baseUrl, fetch: (url, init) => fetch(url, init) });
  });

  after(() => {
    router?.kill("SIGTERM");
  });

  const live = (name: string, body: () => Promise<void>) =>
    test(name, async (t) => {
      if (!available) {
        t.skip("router did not start (is the Python package installed?)");
        return;
      }
      await body();
    });

  // ---------------------------------------------------------------- the common case

  live("run returns a completion and never mentions the queue", async () => {
    const completion = await client.run("Qwen/Qwen3-8B", BODY);

    assert.equal(completion.object, "chat.completion");
    assert.equal(completion.choices[0]?.message.role, "assistant");
    assert.ok(completion.choices[0]?.message.content);
  });

  live("a body field the router does not route on still reaches the backend", async () => {
    // `max_tokens` is neither `model` nor `stream`, so the router forwards it untouched
    // and the backend is what acts on it. A shorter answer is the proof it arrived.
    const short = await client.run("Qwen/Qwen3-8B", { ...BODY, max_tokens: 2 });
    const long = await client.run("Qwen/Qwen3-8B", { ...BODY, max_tokens: 16 });

    assert.ok(
      short.choices[0]!.message.content!.length < long.choices[0]!.message.content!.length,
      "max_tokens did not reach the backend",
    );
  });

  live("submit hands back the router's account of the wait", async () => {
    const job = await client.submit("translate", BODY);

    assert.ok(["queued", "running"].includes(job.status));
    const wait = job.wait;
    assert.ok(wait, "a queued job should report its wait");
    assert.equal(typeof wait.position, "number");
    assert.equal(typeof wait.estimated_seconds, "number");
    assert.ok(job.document!.retry_after > 0);

    await job.result();
  });

  // ---------------------------------------------------------------- giving up

  live("aborting the caller cancels the job", async () => {
    // The invariant this SDK exists to carry. A held-open socket is what tells the router
    // today that a client is still there; once nobody is holding one, this is the only
    // thing that stops the GPU generating an answer with nowhere to go.
    const controller = new AbortController();
    const job = await client.submit("translate", BODY);
    const waiting = job.result({ signal: controller.signal });

    await new Promise((resolve) => setTimeout(resolve, 150));
    controller.abort();
    await assert.rejects(waiting);

    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.equal((await readJob(job.id)).status, "cancelled");
  });

  live("a deadline cancels the job rather than orphaning it", async () => {
    const job = await client.submit("translate", BODY);

    await assert.rejects(job.result({ timeoutMs: 200 }), RequestTimeout);

    await new Promise((resolve) => setTimeout(resolve, 200));
    assert.equal((await readJob(job.id)).status, "cancelled");
  });

  live("cancelling a job the caller never waited on still reaches the router", async () => {
    const job = await client.submit("translate", BODY);
    await job.cancel();

    assert.equal((await readJob(job.id)).status, "cancelled");
  });

  // ---------------------------------------------------------------- a job that vanishes

  live("a job the router has never heard of is reported as lost", async () => {
    const ghost = new Job("req_nonexistent", "translate", baseUrl, client);

    await assert.rejects(ghost.result(), JobLost);
  });

  // ---------------------------------------------------------------- idempotency

  live("the same idempotency key collapses a duplicate submission", async () => {
    const first = await client.submit("translate", BODY, { idempotencyKey: "ts-key-1" });
    const second = await client.submit("translate", BODY, { idempotencyKey: "ts-key-1" });

    assert.equal(first.id, second.id);
    await first.result();
  });

  // ---------------------------------------------------------------- discovery

  live("discovery finds the tags the router advertises", async () => {
    const discovered = new Client({ baseUrl: "http://unused" });
    await discovered.registry.discover([baseUrl]);

    assert.deepEqual(discovered.registry.models, ["Qwen/Qwen3-8B", "translate"]);
    assert.equal(discovered.registry.resolve("translate"), baseUrl);
  });

  // ---------------------------------------------------------------- cross-SDK agreement

  live("the job document matches what the TypeScript types declare", async () => {
    // The two SDKs are only interchangeable if both agree with the server. This asserts
    // the shape the type definitions promise, against the real thing.
    const job = await client.submit("translate", BODY);
    const document = await readJob(job.id);

    assert.equal(document.object, "sir.job");
    assert.ok(["queued", "running", "done", "failed", "cancelled"].includes(document.status));
    assert.equal(typeof document.retry_after, "number");
    assert.ok("wait" in document && "response" in document && "error" in document);

    await job.cancel();
  });
});
