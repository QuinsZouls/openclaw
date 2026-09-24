import { reasoningTagTextPolicy } from "@openclaw/ai/internal/openai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createLlmStreamSimpleMock } from "../../../test/helpers/agents/llm-stream-simple-mock.js";
import type { Model } from "../../llm/types.js";
import { isMiMoStrictReasoningTagsModel } from "./extra-params.mimo-reasoning.js";

vi.mock("../../llm/stream.js", () => createLlmStreamSimpleMock());

let runExtraParamsCase: typeof import("./extra-params.test-support.js").runExtraParamsCase;

function mimoModel(modelId: string, api = "openai-completions"): Model<"openai-completions"> {
  return {
    api,
    provider: "xiaomi-coding",
    id: modelId,
  } as Model<"openai-completions">;
}

function runMiMoStrictCase(modelId: string) {
  return runExtraParamsCase({
    applyProvider: "xiaomi-coding",
    applyModelId: modelId,
    mockProviderRuntime: true,
    thinkingLevel: "high",
    model: mimoModel(modelId),
    payload: {
      model: modelId,
      messages: [],
    },
  });
}

describe("extra-params: MiMo strict reasoning-tag fallback", () => {
  beforeEach(async () => {
    ({ runExtraParamsCase } = await import("./extra-params.test-support.js"));
  });

  it("marks strict-on-flush reasoning tags for MiMo v2.6 openai-completions models", () => {
    const captured = runMiMoStrictCase("mimo-v2.6-pro");
    // Stream-safe level only: MiMo chat must keep streaming visible text, so the
    // wrapper must not select the full-strict level that buffers everything.
    expect(reasoningTagTextPolicy.isStrictOnFlush(captured.options)).toBe(true);
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(false);
  });

  it("does not mark legacy mimo-v2-pro visible-text models strict", () => {
    const captured = runMiMoStrictCase("mimo-v2-pro");
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(false);
    expect(reasoningTagTextPolicy.isStrictOnFlush(captured.options)).toBe(false);
  });

  it("does not mark non-MiMo openai-completions models strict", () => {
    const captured = runMiMoStrictCase("gpt-4o");
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(false);
    expect(reasoningTagTextPolicy.isStrictOnFlush(captured.options)).toBe(false);
  });

  it("classifies MiMo strict reasoning-tag models from the extracted sibling module", () => {
    expect(isMiMoStrictReasoningTagsModel(mimoModel("mimo-v2.6-pro"))).toBe(true);
    expect(isMiMoStrictReasoningTagsModel(mimoModel("mimo-v2.5"))).toBe(true);
    // Proxy routes and `:suffix` variants normalize to the same leaf id.
    expect(isMiMoStrictReasoningTagsModel(mimoModel("xiaomi-orbit/mimo-v2.6-flash:high"))).toBe(
      true,
    );
    // Legacy visible-text models and non-completions transports stay unmatched.
    expect(isMiMoStrictReasoningTagsModel(mimoModel("mimo-v2-pro"))).toBe(false);
    expect(isMiMoStrictReasoningTagsModel(mimoModel("mimo-v2.6-pro", "openai-responses"))).toBe(
      false,
    );
  });
});
