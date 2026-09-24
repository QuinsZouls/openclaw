import { describe, expect, it } from "vitest";
import { processCompletionsStream } from "./openai-completions-stream.js";
import {
  createAssistantOutput,
  expectRecordFields,
  makeCompletionsChunk,
  makeCompletionsModel,
  streamChunks,
} from "./openai-completions.test-support.js";

describe("openai completions stream", () => {
  it.each([
    {
      name: "keeps streamed tool call arguments intact when reasoning_details repeats",
      model: {
        id: "openrouter/qwen/qwen3-235b-a22b",
        name: "Qwen3 235B A22B",
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
      },
      chunks: [
        makeCompletionsChunk({
          reasoning_details: [{ type: "reasoning.text", text: "Need a tool." }],
          tool_calls: [
            {
              id: "call_1",
              type: "function" as const,
              function: { name: "lookup", arguments: '{"query":' },
            },
          ],
        }),
        makeCompletionsChunk({
          reasoning_details: [{ type: "reasoning.text", text: " Still thinking." }],
          tool_calls: [
            {
              id: "call_1",
              type: "function" as const,
              function: { arguments: '"qwen3"}' },
            },
          ],
        }),
        makeCompletionsChunk({}, "tool_calls"),
      ],
      expectedFirst: { type: "thinking", thinking: "Need a tool." },
      expectedSecond: {
        type: "toolCall",
        id: "call_1",
        name: "lookup",
        arguments: { query: "qwen3" },
      },
      expectedThird: {
        type: "thinking",
        thinking: " Still thinking.",
        thinkingSignature: "reasoning_details",
      },
    },
    {
      name: "surfaces visible OpenRouter response text from reasoning_details without dropping tools",
      model: {
        id: "openrouter/minimax/minimax-m2.7",
        name: "MiniMax M2.7",
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
      },
      chunks: [
        makeCompletionsChunk({
          reasoning_details: [
            { type: "reasoning.text", text: "Need to look something up." },
            { type: "response.output_text", text: "Working on it." },
          ],
          tool_calls: [
            {
              id: "call_1",
              type: "function" as const,
              function: { name: "lookup", arguments: '{"query":"weather"}' },
            },
          ],
        }),
        makeCompletionsChunk({}, "tool_calls" as const),
      ],
      expectedFirst: {
        type: "thinking",
        thinking: "Need to look something up.",
        thinkingSignature: "reasoning_details",
      },
      expectedSecond: { type: "text", text: "Working on it." },
      expectedThird: {
        type: "toolCall",
        id: "call_1",
        name: "lookup",
        arguments: { query: "weather" },
      },
    },
  ])(
    "$name",
    async ({ model: modelOverrides, chunks, expectedFirst, expectedSecond, expectedThird }) => {
      const model = makeCompletionsModel(modelOverrides);
      const output = createAssistantOutput(model);

      await processCompletionsStream(streamChunks(chunks), output, model, {
        push() {},
      });

      expect(output.stopReason).toBe("toolUse");
      expect(output.content).toHaveLength(3);
      expectRecordFields(output.content[0], expectedFirst);
      expectRecordFields(output.content[1], expectedSecond);
      expectRecordFields(output.content[2], expectedThird);
    },
  );

  it.each([
    {
      name: "does not surface ambiguous reasoning_details text without explicit compat opt-in",
      model: {
        id: "openrouter/x-ai/grok-4",
        name: "Grok 4",
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
      },
      chunks: [
        makeCompletionsChunk({
          reasoning_details: [
            { type: "reasoning.text", text: "Internal thought." },
            { type: "text", text: "Do not leak this by default." },
          ],
        }),
        makeCompletionsChunk({}, "stop" as const),
      ],
      expected: {
        type: "thinking",
        thinking: "Internal thought.",
        thinkingSignature: "reasoning_details",
      },
    },
    {
      name: "does not duplicate fallback reasoning fields when reasoning_details already provided thinking",
      model: {
        id: "openrouter/minimax/minimax-m2.7",
        name: "MiniMax M2.7",
        provider: "openrouter",
        baseUrl: "https://openrouter.ai/api/v1",
      },
      chunks: [
        makeCompletionsChunk(
          {
            reasoning_details: [{ type: "reasoning.text", text: "Primary reasoning." }],
            reasoning: "Duplicate fallback reasoning.",
          },
          "stop" as const,
        ),
      ],
      expected: {
        type: "thinking",
        thinking: "Primary reasoning.",
        thinkingSignature: "reasoning_details",
      },
    },
  ])("$name", async ({ model: modelOverrides, chunks, expected }) => {
    const model = makeCompletionsModel(modelOverrides);
    const output = createAssistantOutput(model);

    await processCompletionsStream(streamChunks(chunks), output, model, {
      push() {},
    });

    expect(output.content).toHaveLength(1);
    expectRecordFields(output.content[0], expected);
  });

  it("preserves explicitly visible reasoning_details without phase reclassification", async () => {
    const model = makeCompletionsModel({
      id: "openrouter/minimax/minimax-m2.7",
      name: "MiniMax M2.7",
      provider: "openrouter",
      baseUrl: "https://openrouter.ai/api/v1",
    });

    const output = createAssistantOutput(model);

    const stream: { push(event: unknown): void } = { push() {} };

    const mockChunks = [
      makeCompletionsChunk(
        {
          reasoning_details: [
            { type: "response.output_text", text: "Visible first." },
            { type: "reasoning.text", text: " Hidden second." },
            { type: "response.text", text: " Visible third." },
          ],
        },
        "stop",
      ),
    ] as const;

    await processCompletionsStream(streamChunks(mockChunks), output, model, stream);

    expect(output.content).toHaveLength(3);
    expectRecordFields(output.content[0], { type: "text", text: "Visible first." });
    expectRecordFields(output.content[1], {
      type: "thinking",
      thinking: " Hidden second.",
      thinkingSignature: "reasoning_details",
    });
    expectRecordFields(output.content[2], { type: "text", text: " Visible third." });
  });

  it("phases text interrupted by resumed reasoning_details", async () => {
    const model = makeCompletionsModel({
      id: "openrouter/qwen/qwen3-235b-a22b",
      name: "Qwen3 235B A22B",
      provider: "openrouter",
      baseUrl: "https://openrouter.ai/api/v1",
    });
    const output = createAssistantOutput(model);

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({
          reasoning_details: [{ type: "reasoning.text", text: "First thought." }],
        }),
        makeCompletionsChunk({ content: "Interim." }),
        makeCompletionsChunk({
          reasoning_details: [{ type: "reasoning.text", text: "Second thought." }],
        }),
        makeCompletionsChunk({ content: "Final." }),
        makeCompletionsChunk({}, "stop"),
      ]),
      output,
      model,
      { push() {} },
    );

    expect(output.content).toEqual([
      {
        type: "thinking",
        thinking: "First thought.",
        thinkingSignature: "reasoning_details",
      },
      {
        type: "text",
        text: "Interim.",
        textSignature: expect.stringMatching(
          /^\{"v":1,"id":"commentary-0-[0-9a-f]{24}","phase":"commentary"\}$/u,
        ),
      },
      {
        type: "thinking",
        thinking: "Second thought.",
        thinkingSignature: "reasoning_details",
      },
      {
        type: "text",
        text: "Final.",
        textSignature: expect.stringMatching(
          /^\{"v":1,"id":"final-answer-0-[0-9a-f]{24}","phase":"final_answer"\}$/u,
        ),
      },
    ]);
  });

  it("keeps fallback thinking when reasoning_details only carries visible text", async () => {
    const model = makeCompletionsModel({
      id: "openrouter/minimax/minimax-m2.7",
      name: "MiniMax M2.7",
      provider: "openrouter",
      baseUrl: "https://openrouter.ai/api/v1",
    });

    const output = createAssistantOutput(model);

    const stream: { push(event: unknown): void } = { push() {} };

    const mockChunks = [
      makeCompletionsChunk(
        {
          reasoning_details: [{ type: "response.output_text", text: "Visible answer." }],
          reasoning: "Hidden fallback reasoning.",
        },
        "stop",
      ),
    ] as const;

    await processCompletionsStream(streamChunks(mockChunks), output, model, stream);

    expect(output.content).toHaveLength(2);
    expectRecordFields(output.content[0], { type: "text", text: "Visible answer." });
    expectRecordFields(output.content[1], {
      type: "thinking",
      thinking: "Hidden fallback reasoning.",
      thinkingSignature: "reasoning",
    });
  });

  it.each([
    {
      name: "fails fast when post-tool-call buffering grows beyond the safety cap",
      makeChunks: () => [
        makeCompletionsChunk({
          tool_calls: [
            {
              id: "call_1",
              type: "function" as const,
              function: { name: "lookup", arguments: '{"query":' },
            },
          ],
        }),
        makeCompletionsChunk({ content: "x".repeat(300_000) }),
      ],
      expectedError: "Exceeded post-tool-call delta buffer limit",
    },
    {
      name: "fails fast when streaming tool-call arguments grow beyond the safety cap",
      makeChunks: () => {
        const oversizedArgs = `"${"x".repeat(300_000)}"}`;
        return [
          makeCompletionsChunk({
            tool_calls: [
              {
                id: "call_1",
                type: "function" as const,
                function: { name: "lookup", arguments: `{${oversizedArgs}` },
              },
            ],
          }),
        ];
      },
      expectedError: "Exceeded tool-call argument buffer limit",
    },
  ])("$name", async ({ makeChunks, expectedError }) => {
    const model = makeCompletionsModel({
      id: "openrouter/minimax/minimax-m2.7",
      name: "MiniMax M2.7",
      provider: "openrouter",
      baseUrl: "https://openrouter.ai/api/v1",
    });
    const output = createAssistantOutput(model);

    await expect(
      processCompletionsStream(streamChunks(makeChunks()), output, model, {
        push() {},
      }),
    ).rejects.toThrow(expectedError);
  });
});

describe("openai completions stream: MiMo inline reasoning leak on tool-call turns (#156803)", () => {
  function visibleTextOf(output: ReturnType<typeof createAssistantOutput>) {
    return output.content
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("");
  }

  function toolCallsOf(
    output: ReturnType<typeof createAssistantOutput>,
  ): Array<{ type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }> {
    return output.content.filter(
      (
        block,
      ): block is {
        type: "toolCall";
        id: string;
        name: string;
        arguments: Record<string, unknown>;
      } => block.type === "toolCall",
    );
  }

  async function runLeakyStream(chunks: unknown[], strict: boolean) {
    const model = makeCompletionsModel({
      id: "mimo-v2.6-pro",
      name: "MiMo V2.6 Pro",
      provider: "vllm",
      baseUrl: "http://localhost:8000/v1",
    });
    const output = createAssistantOutput(model);
    await processCompletionsStream(
      streamChunks(chunks as Parameters<typeof streamChunks>[0]),
      output,
      model,
      { push() {} },
      { strictReasoningTags: strict },
    );
    return output;
  }

  function makeLeakyToolCallChunks() {
    // vLLM mimo parser streams a literal opener and absorbs the closer server-side.
    return [
      makeCompletionsChunk({ content: "<think>secret reasoning step" }),
      makeCompletionsChunk({
        tool_calls: [
          {
            index: 0,
            id: "call_1",
            type: "function" as const,
            function: { name: "lookup", arguments: '{"query":"weather"}' },
          },
        ],
      }),
      makeCompletionsChunk({}, "tool_calls" as const),
    ];
  }

  it("hides unclosed inline reasoning from visible text when strictReasoningTags is enabled", async () => {
    const model = makeCompletionsModel({
      id: "mimo-v2.6-pro",
      name: "MiMo V2.6 Pro",
      provider: "vllm",
      baseUrl: "http://localhost:8000/v1",
    });
    const output = createAssistantOutput(model);

    await processCompletionsStream(
      streamChunks(makeLeakyToolCallChunks()),
      output,
      model,
      { push() {} },
      { strictReasoningTags: true },
    );

    const visibleText = output.content
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("");
    expect(visibleText).toBe("");
    expect(visibleText).not.toContain("secret reasoning step");

    const toolCalls = toolCallsOf(output);
    expect(output.stopReason).toBe("toolUse");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]?.name).toBe("lookup");
    expect(toolCalls[0]?.arguments).toEqual({ query: "weather" });
  });

  it("reproduces the leak as visible text when strictReasoningTags is disabled", async () => {
    const model = makeCompletionsModel({
      id: "mimo-v2.6-pro",
      name: "MiMo V2.6 Pro",
      provider: "vllm",
      baseUrl: "http://localhost:8000/v1",
    });
    const output = createAssistantOutput(model);

    await processCompletionsStream(
      streamChunks(makeLeakyToolCallChunks()),
      output,
      model,
      { push() {} },
      { strictReasoningTags: false },
    );

    // Non-strict mode recovers the unclosed pending buffer as visible TEXT at
    // the tool-call boundary — this is the leak reported in issue #156803.
    const visibleText = output.content
      .filter((block): block is { type: "text"; text: string } => block.type === "text")
      .map((block) => block.text)
      .join("");
    expect(visibleText).toContain("secret reasoning step");

    const toolCalls = toolCallsOf(output);
    expect(toolCalls).toHaveLength(1);
  });

  it("hides a reasoning opener split across streamed chunks when strict is enabled", async () => {
    // Packet boundaries may cut the opener itself ("<thi" | "nk>"); the tag
    // probe must still route the whole block away from visible text.
    const output = await runLeakyStream(
      [
        makeCompletionsChunk({ content: "<thi" }),
        makeCompletionsChunk({ content: "nk>secret reasoning step" }),
        makeCompletionsChunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function" as const,
              function: { name: "lookup", arguments: "{}" },
            },
          ],
        }),
        makeCompletionsChunk({}, "tool_calls" as const),
      ],
      true,
    );

    expect(visibleTextOf(output)).toBe("");
    expect(toolCallsOf(output)).toHaveLength(1);
  });

  it("hides inline reasoning when content and tool calls share one chunk", async () => {
    const output = await runLeakyStream(
      [
        makeCompletionsChunk({
          content: "<think>secret reasoning step",
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function" as const,
              function: { name: "lookup", arguments: "{}" },
            },
          ],
        }),
        makeCompletionsChunk({}, "tool_calls" as const),
      ],
      true,
    );

    expect(visibleTextOf(output)).toBe("");
    expect(toolCallsOf(output)).toHaveLength(1);
  });

  it("drops leaked reasoning instead of promoting it into the thinking lane", async () => {
    const output = await runLeakyStream(makeLeakyToolCallChunks(), true);

    // Strict mode classifies the unclosed block as reasoning and drops it;
    // it must never resurface as a thinking block on the visible output.
    expect(output.content.some((block) => block.type === "thinking")).toBe(false);
    expect(visibleTextOf(output)).toBe("");
  });

  it("keeps streaming ordinary visible text when strict is enabled", async () => {
    // Strict mode must not over-hide: normal answers without reasoning tags
    // still stream through (paragraph boundaries release buffered text).
    const output = await runLeakyStream(
      [
        makeCompletionsChunk({ content: "All good.\n\n" }),
        makeCompletionsChunk({ content: "Nothing here is hidden." }),
        makeCompletionsChunk({}, "stop" as const),
      ],
      true,
    );

    expect(visibleTextOf(output)).toContain("All good.");
    expect(visibleTextOf(output)).toContain("Nothing here is hidden.");
    expect(output.stopReason).toBe("stop");
  });
});
