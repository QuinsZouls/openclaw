// MiMo strict reasoning-tag wrapper tests cover policy marking and option safety.
import { reasoningTagTextPolicy } from "@openclaw/ai/internal/openai";
import type { StreamFn } from "openclaw/plugin-sdk/agent-core";
import type { Context, Model } from "openclaw/plugin-sdk/llm";
import { describe, expect, it } from "vitest";
import { createMiMoStrictReasoningTagsWrapper } from "./mimo-reasoning-tags.js";

const context = { messages: [] } as Context;

function makeModel(id: string, api = "openai-completions"): Model<"openai-completions"> {
  return { api, provider: "xiaomi-coding", id } as Model<"openai-completions">;
}

function isMiMoStrictModel(model: Parameters<StreamFn>[0]): boolean {
  return model.api === "openai-completions" && model.id === "mimo-v2.6-pro";
}

describe("createMiMoStrictReasoningTagsWrapper", () => {
  it("marks strict reasoning-tag policy for matching MiMo models", () => {
    let capturedOptions: Parameters<StreamFn>[2];
    const baseStreamFn: StreamFn = (_model, _context, options) => {
      capturedOptions = options;
      return {} as ReturnType<StreamFn>;
    };
    const wrapped = createMiMoStrictReasoningTagsWrapper({
      baseStreamFn,
      shouldMarkStrict: isMiMoStrictModel,
    });

    void wrapped?.(makeModel("mimo-v2.6-pro"), context, {});

    expect(reasoningTagTextPolicy.isStrict(capturedOptions)).toBe(true);
  });

  it("passes options through unmarked for non-matching models", () => {
    let capturedOptions: Parameters<StreamFn>[2];
    const baseStreamFn: StreamFn = (_model, _context, options) => {
      capturedOptions = options;
      return {} as ReturnType<StreamFn>;
    };
    const wrapped = createMiMoStrictReasoningTagsWrapper({
      baseStreamFn,
      shouldMarkStrict: isMiMoStrictModel,
    });

    void wrapped?.(makeModel("mimo-v2-pro"), context, {});

    expect(reasoningTagTextPolicy.isStrict(capturedOptions)).toBe(false);
  });

  it("does not mutate the caller's options object", () => {
    const baseStreamFn: StreamFn = () => ({}) as ReturnType<StreamFn>;
    const wrapped = createMiMoStrictReasoningTagsWrapper({
      baseStreamFn,
      shouldMarkStrict: isMiMoStrictModel,
    });
    const originalOptions = {};

    void wrapped?.(makeModel("mimo-v2.6-pro"), context, originalOptions);

    expect(reasoningTagTextPolicy.isStrict(originalOptions)).toBe(false);
  });

  it("returns undefined when the base stream function is missing", () => {
    expect(
      createMiMoStrictReasoningTagsWrapper({
        baseStreamFn: undefined,
        shouldMarkStrict: isMiMoStrictModel,
      }),
    ).toBeUndefined();
  });
});
