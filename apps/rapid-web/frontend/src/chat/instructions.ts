/**
 * The two user-authored instruction layers, ported from `rapid-mac`'s
 * `CustomInstructionsConfig` + `ChatViewModel.addingInstructionLayers`.
 *
 * A global default applies everywhere; a conversation may add its own, which
 * wins where the two conflict. Both are merged into ONE leading system row —
 * local chat templates often reject a second system message, so every caller
 * has to go through this.
 */

/** rapid-mac's `CustomInstructionsConfig.maximumLength`. */
export const MAX_INSTRUCTION_LENGTH = 4000;

/** The instruction as it would be sent, or null when it says nothing. */
export function normalizeInstruction(value: string): string | null {
  const trimmed = value.slice(0, MAX_INSTRUCTION_LENGTH).trim();
  return trimmed === '' ? null : trimmed;
}

/**
 * The single system message for a request.
 *
 * The labels are what make precedence legible to the model, so they are only
 * applied when there are actually two layers to rank — one instruction on its
 * own is sent as written.
 */
export function composeSystemPrompt(options: {
  global: string;
  conversation: string;
  /** The tool preamble, once a tool result is in the history. Leads, as in
   *  rapid-mac, because it describes the turn rather than the user. */
  guidance?: string | undefined;
}): string {
  const parts: string[] = [];
  if (options.guidance) parts.push(options.guidance);

  const global = normalizeInstruction(options.global);
  const conversation = normalizeInstruction(options.conversation);

  if (global !== null && conversation !== null) {
    parts.push(
      `[GLOBAL USER INSTRUCTIONS]\nThese user preferences apply unless this conversation has a conflicting instruction:\n${global}`,
    );
    parts.push(
      `[CONVERSATION INSTRUCTIONS - HIGHEST USER PRIORITY]\nThese instructions apply only to this conversation. If they conflict with the global user instructions above, follow THESE:\n${conversation}`,
    );
  } else if (global !== null) {
    parts.push(global);
  } else if (conversation !== null) {
    parts.push(conversation);
  }

  return parts.join('\n\n');
}
