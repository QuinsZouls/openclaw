import { reasoningTagTextPolicy } from "@openclaw/ai/internal/openai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createLlmStreamSimpleMock } from "../../../test/helpers/agents/llm-stream-simple-mock.js";
import type { Model } from "../../llm/types.js";

vi.mock("../../llm/stream.js", () => createLlmStreamSimpleMock());

let runExtraParamsCase: typeof import("./extra-params.test-support.js").runExtraParamsCase;

function runMiMoStrictCase(modelId: string) {
  return runExtraParamsCase({
    applyProvider: "xiaomi-coding",
    applyModelId: modelId,
    mockProviderRuntime: true,
    thinkingLevel: "high",
    model: {
      api: "openai-completions",
      provider: "xiaomi-coding",
      id: modelId,
    } as Model<"openai-completions">,
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

  it("marks strict reasoning tags for MiMo v2.6 openai-completions models", () => {
    const captured = runMiMoStrictCase("mimo-v2.6-pro");
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(true);
  });

  it("does not mark legacy mimo-v2-pro visible-text models strict", () => {
    const captured = runMiMoStrictCase("mimo-v2-pro");
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(false);
  });

  it("does not mark non-MiMo openai-completions models strict", () => {
    const captured = runMiMoStrictCase("gpt-4o");
    expect(reasoningTagTextPolicy.isStrict(captured.options)).toBe(false);
  });
});
