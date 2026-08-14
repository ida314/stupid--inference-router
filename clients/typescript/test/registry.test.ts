/** Routing decisions, which happen before anything is sent and need no server. */

import assert from "node:assert/strict";
import { test } from "node:test";

import { ModelNotRouted, Registry } from "../src/index.ts";

test("an unrouted model fails before anything is sent", () => {
  const registry = new Registry({ chat: "http://gpu:8000" });

  assert.throws(() => registry.resolve("something-else"), (error: unknown) => {
    assert.ok(error instanceof ModelNotRouted);
    assert.match(error.message, /chat/); // tells you what it does know
    return true;
  });
});

test("endpoints come from the environment", () => {
  const registry = Registry.fromEnv({
    SIR_ENDPOINTS: "Qwen/Qwen3-8B=http://gpu:8000, bge-m3=http://cpu:8001/",
    SIR_BASE_URL: "http://router:8000",
  });

  assert.equal(registry.resolve("Qwen/Qwen3-8B"), "http://gpu:8000");
  assert.equal(registry.resolve("bge-m3"), "http://cpu:8001");
  assert.equal(registry.resolve("anything-else"), "http://router:8000");
});

test("a malformed endpoint list is rejected at startup", () => {
  assert.throws(() => Registry.fromEnv({ SIR_ENDPOINTS: "no-url-here" }), /malformed/);
});

test("an empty environment yields a registry that routes nothing", () => {
  assert.throws(() => Registry.fromEnv({}).resolve("chat"), ModelNotRouted);
});

test("two hosts claiming one tag is an error, not a coin flip", async () => {
  // Same reasoning as the router's duplicate `served_model_name` check: routing that
  // depends on the order things were listed in is routing that breaks silently.
  const models = (ids: string[]) =>
    new Response(JSON.stringify({ data: ids.map((id) => ({ id })) }), { status: 200 });

  const registry = new Registry();
  await registry.discover(["http://a"], async () => models(["chat"]));

  await assert.rejects(
    registry.discover(["http://b"], async () => models(["chat"])),
    /routing would depend on discovery order/,
  );
});

test("discovery registers every tag a host advertises", async () => {
  const registry = await new Registry().discover(["http://router:8000/"], async () =>
    new Response(JSON.stringify({ data: [{ id: "chat" }, { id: "translate" }] }), {
      status: 200,
    }),
  );

  assert.deepEqual(registry.models, ["chat", "translate"]);
  assert.equal(registry.resolve("translate"), "http://router:8000");
});
