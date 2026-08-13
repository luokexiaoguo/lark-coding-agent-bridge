import type { AgentEvent } from '../types';

interface ContentBlock {
  type: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  content?: unknown;
  is_error?: boolean;
}

interface ClaudeRawEvent {
  type?: string;
  subtype?: string;
  session_id?: string;
  cwd?: string;
  model?: string;
  message?: { content?: ContentBlock[] };
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
  };
  total_cost_usd?: number;
}

export interface ClaudeEventTranslator {
  translate(raw: unknown): AgentEvent[];
  /**
   * Flush any buffered text as ordinary progress (`text`). Used when the
   * process ends without a `result` (non-zero exit, kill): the last words the
   * agent produced are commentary, not a conclusion, and should still reach
   * the user. A normal run ends with `result`, which flushes as `final_text`.
   */
  flushAsText(): AgentEvent[];
}

/**
 * Stateful translator for claude's `stream-json` output.
 *
 * Claude emits each assistant turn as one event whose text block is the full
 * reply for that turn. Mid-run turns are progress commentary; the final turn
 * (the one just before `result`) is the answer. We buffer text so that when
 * `result` arrives we know the last text block is the conclusion and emit it
 * as `final_text` — mirroring codex/mimo — while everything before it streams
 * as regular `text` (progress the user can read while tools run).
 */
export function createTranslateEvent(): ClaudeEventTranslator {
  let pendingText: string | undefined;

  const flushPending = (events: AgentEvent[], asFinal: boolean): void => {
    if (pendingText === undefined) return;
    events.push(
      asFinal
        ? { type: 'final_text', content: pendingText }
        : { type: 'text', delta: pendingText },
    );
    pendingText = undefined;
  };

  return {
    translate(raw: unknown): AgentEvent[] {
      if (!raw || typeof raw !== 'object') return [];
      const evt = raw as ClaudeRawEvent;
      const events: AgentEvent[] = [];

      if (evt.type === 'system' && evt.subtype === 'init') {
        events.push({
          type: 'system',
          sessionId: evt.session_id,
          cwd: evt.cwd,
          model: evt.model,
        });
        return events;
      }

      if (evt.type === 'assistant' && evt.message?.content) {
        let sawToolUse = false;
        for (const block of evt.message.content) {
          if (block.type === 'text' && typeof block.text === 'string' && block.text) {
            // A text block that follows a tool_use in the same message is a new
            // turn's commentary; flush any buffered text as progress first.
            if (sawToolUse && pendingText !== undefined) {
              flushPending(events, false);
            }
            pendingText = block.text;
          } else if (
            block.type === 'thinking' &&
            typeof block.thinking === 'string' &&
            block.thinking
          ) {
            events.push({ type: 'thinking', delta: block.thinking });
          } else if (block.type === 'tool_use' && block.id && block.name) {
            sawToolUse = true;
            // A tool call ends the commentary turn: its text is progress, not
            // the answer.
            flushPending(events, false);
            events.push({
              type: 'tool_use',
              id: block.id,
              name: block.name,
              input: block.input,
            });
          }
        }
        return events;
      }

      if (evt.type === 'user' && evt.message?.content) {
        for (const block of evt.message.content) {
          if (block.type === 'tool_result' && block.tool_use_id) {
            const output =
              typeof block.content === 'string' ? block.content : JSON.stringify(block.content);
            events.push({
              type: 'tool_result',
              id: block.tool_use_id,
              output,
              isError: block.is_error === true,
            });
          }
        }
        return events;
      }

      if (evt.type === 'result') {
        // Whatever text is still buffered is the final answer.
        flushPending(events, true);
        if (evt.usage) {
          events.push({
            type: 'usage',
            inputTokens: evt.usage.input_tokens,
            outputTokens: evt.usage.output_tokens,
            cachedInputTokens: evt.usage.cache_read_input_tokens,
            costUsd: evt.total_cost_usd,
          });
        }
        events.push({ type: 'done', sessionId: evt.session_id, terminationReason: 'normal' });
      }
      return events;
    },

    flushAsText(): AgentEvent[] {
      const events: AgentEvent[] = [];
      flushPending(events, false);
      return events;
    },
  };
}

/** Backwards-compatible stateless wrapper for single-line callers/tests. */
export function translateEvent(raw: unknown): Generator<AgentEvent> {
  const translator = createTranslateEvent();
  return (function* () {
    for (const event of translator.translate(raw)) yield event;
  })();
}
