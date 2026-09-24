// MiMo reasoning-tag wrapper enforces strict reasoning-tag partitioning so
// inline `<think>...` thinking leaks from vLLM mimo-parser endpoints stay hidden.
import { reasoningTagTextPolicy } from "@openclaw/ai/internal/openai";
import type { StreamFn } from "../../../agents/runtime/index.js";

export function createMiMoStrictReasoningTagsWrapper(params: {
  baseStreamFn: StreamFn | undefined;
  shouldMarkStrict: (model: Parameters<StreamFn>[0]) => boolean;
}): StreamFn | undefined {
  if (!params.baseStreamFn) {
    return undefined;
  }
  const underlying = params.baseStreamFn;
  return (model, context, options) => {
    if (!params.shouldMarkStrict(model)) {
      return underlying(model, context, options);
    }
    // Mark a fresh spread copy so the caller's options object stays untouched.
    const strictOptions = { ...options };
    reasoningTagTextPolicy.markStrict(strictOptions);
    return underlying(model, context, strictOptions);
  };
}
