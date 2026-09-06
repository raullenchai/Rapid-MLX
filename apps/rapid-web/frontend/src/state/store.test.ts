import { beforeEach, describe, expect, it } from 'vitest';
import { useStore, wireTurns } from './store';
import type { MessageNode } from './types';

/**
 * `createConversation` reuses an existing blank rather than stacking a new one.
 *
 * Pressing New Chat twice used to leave a column of identical "New chat" rows
 * that the user then had to delete one at a time.
 */
describe('selectAlias', () => {
  beforeEach(() => {
    useStore.setState({
      selectedByKind: { text: null, image: null, audio: null },
      status: null,
      models: [],
    });
  });

  it('keeps the chat and images selections apart', () => {
    // One shared field meant choosing an image model retargeted the chat,
    // which then reported the image model's start failure as its own.
    useStore.getState().selectAlias('text', 'qwen3-4b');
    useStore.getState().selectAlias('image', 'flux2-klein-4b');

    expect(useStore.getState().selectedByKind.text).toBe('qwen3-4b');
    expect(useStore.getState().selectedByKind.image).toBe('flux2-klein-4b');
  });

  it('adopts a served model as the TEXT selection only', () => {
    useStore.getState().setStatus(
      { state: 'ready', model: 'flux2-klein-4b', port: 1, detail: null, can_switch: true },
      false,
    );

    // A served image model must not become the chat's selection — that is
    // exactly the bug the split exists to prevent.
    expect(useStore.getState().selectedByKind.image).toBeNull();
  });

  it('does not overwrite a text selection the user already made', () => {
    useStore.getState().selectAlias('text', 'qwen3-4b');
    useStore.getState().setStatus(
      { state: 'ready', model: 'llama-8b', port: 1, detail: null, can_switch: true },
      false,
    );

    expect(useStore.getState().selectedByKind.text).toBe('qwen3-4b');
  });

  it('stops guessing text once the catalog names the served kind', () => {
    // A poll every few seconds re-ran the adoption, so an image model kept
    // reappearing in the chat picker right after `setModels` moved it out.
    useStore.getState().setModels([
      {
        alias: 'flux2-klein-4b',
        hf_path: 'Runpod/FLUX.2-klein-4B-mflux-4bit',
        size_bytes: null,
        cached: true,
        kind: 'image',
        loadable: true,
        cached_bytes: null,
        tool_call_parser: null,
        reasoning_parser: null,
        is_text_only: false,
        audio_kind: null,
        family: null,
        image_capability: 'both',
      },
    ]);
    useStore.getState().setStatus(
      { state: 'ready', model: 'flux2-klein-4b', port: 1, detail: null, can_switch: true },
      false,
    );

    expect(useStore.getState().selectedByKind.text).toBeNull();
  });
});

describe('createConversation', () => {
  beforeEach(() => {
    useStore.setState({ conversations: [], activeId: null });
  });

  const count = () => useStore.getState().conversations.length;

  it('creates one when there is nothing to reuse', () => {
    const id = useStore.getState().createConversation();
    expect(count()).toBe(1);
    expect(useStore.getState().activeId).toBe(id);
  });

  it('returns the same blank conversation when pressed repeatedly', () => {
    const first = useStore.getState().createConversation();
    const second = useStore.getState().createConversation();
    const third = useStore.getState().createConversation();

    expect(second).toBe(first);
    expect(third).toBe(first);
    expect(count()).toBe(1);
  });

  it('creates a new one once the blank has been used', () => {
    const first = useStore.getState().createConversation();
    useStore.getState().appendNode({ role: 'user', content: 'hi', status: 'complete', parentId: null });

    const second = useStore.getState().createConversation();
    expect(second).not.toBe(first);
    expect(count()).toBe(2);
  });

  it('does not reuse a blank the user deliberately named', () => {
    const first = useStore.getState().createConversation();
    // A renamed blank is one the user made on purpose; silently adopting it
    // would put the next chat under a title meant for something else.
    useStore.getState().updateConversation(first, { title: 'Scratch', hasCustomTitle: true });

    const second = useStore.getState().createConversation();
    expect(second).not.toBe(first);
    expect(count()).toBe(2);
  });

  it('does not reuse an archived blank', () => {
    const first = useStore.getState().createConversation();
    useStore.getState().updateConversation(first, { isArchived: true });

    const second = useStore.getState().createConversation();
    expect(second).not.toBe(first);
    // Archiving is how something is put out of the way; pulling it back is
    // the opposite of what the user asked for.
    expect(count()).toBe(2);
  });
});

describe('wireTurns', () => {
  const node = (partial: Partial<MessageNode>): MessageNode => ({
    id: 'n',
    parentId: null,
    role: 'assistant',
    content: '',
    status: 'complete',
    createdAt: 0,
    ...partial,
  });

  it('drops a failed turn', () => {
    const turns = wireTurns([node({ role: 'user', content: 'hi', status: 'failed' })], '');
    expect(turns).toEqual([]);
  });

  it('drops an empty turn that produced nothing at all', () => {
    expect(wireTurns([node({ content: '' })], '')).toEqual([]);
  });

  // A turn that only asked for tools has no prose but is not empty. Dropping
  // it orphans the tool results under it and the next request is malformed.
  it('keeps a tool-call turn that has no prose', () => {
    const calls = [
      { id: 'call_1', type: 'function' as const, function: { name: 'weather', arguments: '{}' } },
    ];
    const turns = wireTurns([node({ content: '', toolCalls: calls })], '');

    expect(turns).toEqual([{ role: 'assistant', content: '', tool_calls: calls }]);
  });

  it('carries the call id on a tool result', () => {
    const turns = wireTurns([node({ role: 'tool', content: '18°C', toolCallId: 'call_1' })], '');
    expect(turns).toEqual([{ role: 'tool', content: '18°C', tool_call_id: 'call_1' }]);
  });

  it('puts the system prompt first when there is one', () => {
    const turns = wireTurns([node({ role: 'user', content: 'hi' })], 'Be brief.');
    expect(turns[0]).toEqual({ role: 'system', content: 'Be brief.' });
  });
});

describe('wireTurns and failed tool results', () => {
  // A failed tool RAN, and its error is the result. Dropping it leaves the
  // call above it unanswered, which the model reads as a malformed history.
  it('keeps a failed tool result', () => {
    const turns = wireTurns(
      [
        {
          id: 'n',
          parentId: null,
          role: 'tool',
          content: 'browse error: declined',
          status: 'failed',
          createdAt: 0,
          toolCallId: 'call_1',
        },
      ],
      '',
    );

    expect(turns).toEqual([
      { role: 'tool', content: 'browse error: declined', tool_call_id: 'call_1' },
    ]);
  });
});
