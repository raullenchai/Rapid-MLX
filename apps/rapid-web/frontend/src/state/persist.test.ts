import { describe, expect, it, vi } from 'vitest';
import { activePath } from '@/chat/MessageTree';
import { HISTORY_KEY } from './migrate';
import {
  MAX_ACTIVE_PATH,
  MAX_CONVERSATIONS,
  MAX_NODES,
  capConversation,
  capConversations,
  capStore,
  persist,
  type StorageLike,
} from './persist';
import { SCHEMA_VERSION, type Conversation, type MessageNode, type PersistedStore } from './types';

function conversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    id: 'c',
    title: 'title',
    hasCustomTitle: false,
    createdAt: 1,
    updatedAt: 1,
    nodes: [],
    activeLeafId: null,
    branchChoices: {},
    isPinned: false,
    isArchived: false,
    folderId: null,
    ...overrides,
  };
}

/** A straight chain of `count` nodes, oldest first. */
function chain(count: number, prefix = 'n'): MessageNode[] {
  const nodes: MessageNode[] = [];
  for (let index = 0; index < count; index += 1) {
    nodes.push({
      id: `${prefix}${index}`,
      parentId: index === 0 ? null : `${prefix}${index - 1}`,
      role: index % 2 === 0 ? 'user' : 'assistant',
      content: `message ${index}`,
      status: 'complete',
      createdAt: 1000 + index,
    });
  }
  return nodes;
}

function memoryStorage(): StorageLike & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (key) => data.get(key) ?? null,
    setItem: (key, value) => void data.set(key, value),
    removeItem: (key) => void data.delete(key),
  };
}

describe('capConversations', () => {
  it('leaves a small library alone', () => {
    const rows = [conversation({ id: 'a' }), conversation({ id: 'b' })];
    expect(capConversations(rows).map((c) => c.id)).toEqual(['a', 'b']);
  });

  it('keeps the most recent when over the cap', () => {
    const rows = Array.from({ length: MAX_CONVERSATIONS + 5 }, (_, index) =>
      conversation({ id: `c${index}`, updatedAt: index }),
    );
    const kept = capConversations(rows);
    expect(kept).toHaveLength(MAX_CONVERSATIONS);
    expect(kept[0]?.id).toBe(`c${MAX_CONVERSATIONS + 4}`);
  });

  it('never drops a pinned conversation ahead of an unpinned one', () => {
    // A pin is an explicit instruction. Discarding one silently would be the
    // most infuriating thing this cap could do.
    const rows = [
      conversation({ id: 'ancient-but-pinned', updatedAt: 0, isPinned: true }),
      ...Array.from({ length: MAX_CONVERSATIONS + 5 }, (_, index) =>
        conversation({ id: `c${index}`, updatedAt: 100 + index }),
      ),
    ];
    expect(capConversations(rows).map((c) => c.id)).toContain('ancient-but-pinned');
  });

  it('yields the cap entirely when the user pins more than it allows', () => {
    const rows = Array.from({ length: MAX_CONVERSATIONS + 4 }, (_, index) =>
      conversation({ id: `p${index}`, updatedAt: index, isPinned: true }),
    );
    expect(capConversations(rows)).toHaveLength(MAX_CONVERSATIONS + 4);
  });
});

describe('capConversation', () => {
  it('leaves a small conversation untouched, by identity', () => {
    // Returning a new object every write would defeat any downstream memo.
    const small = conversation({ nodes: chain(10), activeLeafId: 'n9' });
    expect(capConversation(small)).toBe(small);
  });

  it('caps the active path and re-roots it', () => {
    const long = conversation({
      nodes: chain(MAX_ACTIVE_PATH + 50),
      activeLeafId: `n${MAX_ACTIVE_PATH + 49}`,
    });
    const capped = capConversation(long);

    const path = activePath(capped.nodes, capped.activeLeafId, capped.branchChoices);
    expect(path).toHaveLength(MAX_ACTIVE_PATH);
    // The new first node must be a root, or the whole conversation hangs off
    // a reference to something that was dropped.
    expect(path[0]?.parentId).toBeNull();
    // The tail — what the user is actually looking at — is what survives.
    expect(path[path.length - 1]?.id).toBe(`n${MAX_ACTIVE_PATH + 49}`);
  });

  it('keeps the alternatives hanging off retained nodes', () => {
    // A branch the user can still switch to must not lose its continuation.
    const nodes = [
      ...chain(4),
      {
        id: 'alt',
        parentId: 'n0',
        role: 'assistant' as const,
        content: 'alternative',
        status: 'complete' as const,
        createdAt: 2000,
      },
    ];
    const capped = capConversation(conversation({ nodes, activeLeafId: 'n3' }));
    expect(capped.nodes.map((n) => n.id)).toContain('alt');
  });

  it('drops whole off-path subtrees rather than splitting one', () => {
    // Half a branch is a conversation that never happened.
    const nodes = [...chain(4)];
    for (let index = 0; index < MAX_NODES; index += 1) {
      nodes.push(
        {
          id: `alt${index}`,
          parentId: 'n0',
          role: 'assistant',
          content: 'x',
          status: 'complete',
          createdAt: 5000 + index * 10,
        },
        {
          id: `alt${index}-child`,
          parentId: `alt${index}`,
          role: 'user',
          content: 'y',
          status: 'complete',
          createdAt: 5001 + index * 10,
        },
      );
    }

    const capped = capConversation(conversation({ nodes, activeLeafId: 'n3' }));
    const kept = new Set(capped.nodes.map((n) => n.id));
    for (let index = 0; index < MAX_NODES; index += 1) {
      // Either both halves of a branch survive or neither does.
      expect(kept.has(`alt${index}`)).toBe(kept.has(`alt${index}-child`));
    }
  });

  it('prunes branch choices that point at dropped nodes', () => {
    // A stale edge would steer the walk at the very fork the user is standing
    // on, sending them somewhere they never chose.
    const long = conversation({
      nodes: chain(MAX_ACTIVE_PATH + 50),
      activeLeafId: `n${MAX_ACTIVE_PATH + 49}`,
      branchChoices: {
        n0: 'n1',
        [`n${MAX_ACTIVE_PATH + 40}`]: `n${MAX_ACTIVE_PATH + 41}`,
      },
    });
    const capped = capConversation(long);
    const present = new Set(capped.nodes.map((n) => n.id));

    for (const [parent, child] of Object.entries(capped.branchChoices)) {
      expect(present.has(parent)).toBe(true);
      expect(present.has(child)).toBe(true);
    }
    expect(capped.branchChoices['n0']).toBeUndefined();
  });
});

describe('capStore', () => {
  it('repoints activeId when its conversation was dropped', () => {
    const store: PersistedStore = {
      v: SCHEMA_VERSION,
      conversations: Array.from({ length: MAX_CONVERSATIONS + 3 }, (_, index) =>
        conversation({
          id: `c${index}`,
          updatedAt: index,
          nodes: chain(2, `c${index}n`),
        }),
      ),
      folders: [],
      activeId: 'c0',
    };
    const capped = capStore(store);
    expect(capped.conversations.some((c) => c.id === 'c0')).toBe(false);
    expect(capped.conversations.some((c) => c.id === capped.activeId)).toBe(true);
  });
});

describe('persist', () => {
  it('writes the store under the history key', () => {
    const storage = memoryStorage();
    const store: PersistedStore = {
      v: SCHEMA_VERSION,
      conversations: [conversation({ nodes: chain(2) })],
      folders: [],
      activeId: 'c',
    };

    expect(persist(store, storage)).toEqual({ ok: true });
    expect(JSON.parse(storage.data.get(HISTORY_KEY) as string).v).toBe(SCHEMA_VERSION);
  });

  it('drops the oldest unpinned conversation and retries on a quota failure', () => {
    const storage = memoryStorage();
    let failures = 2;
    const setItem = vi.fn((key: string, value: string) => {
      if (failures > 0) {
        failures -= 1;
        const error = new Error('quota');
        error.name = 'QuotaExceededError';
        throw error;
      }
      storage.data.set(key, value);
    });

    const store: PersistedStore = {
      v: SCHEMA_VERSION,
      conversations: [
        conversation({ id: 'old', updatedAt: 1, nodes: chain(2, 'a') }),
        conversation({ id: 'mid', updatedAt: 2, nodes: chain(2, 'b') }),
        conversation({ id: 'new', updatedAt: 3, nodes: chain(2, 'c') }),
      ],
      folders: [],
      activeId: 'new',
    };

    const outcome = persist(store, { ...storage, setItem });
    expect(outcome).toEqual({ ok: true, evicted: 2 });
    expect(
      JSON.parse(storage.data.get(HISTORY_KEY) as string).conversations.map(
        (c: Conversation) => c.id,
      ),
    ).toEqual(['new']);
  });

  it('never evicts a pinned conversation to satisfy a quota', () => {
    const setItem = vi.fn(() => {
      const error = new Error('quota');
      error.name = 'QuotaExceededError';
      throw error;
    });

    const store: PersistedStore = {
      v: SCHEMA_VERSION,
      conversations: [conversation({ id: 'pinned', isPinned: true, nodes: chain(2) })],
      folders: [],
      activeId: 'pinned',
    };

    expect(persist(store, { ...memoryStorage(), setItem })).toEqual({
      ok: false,
      reason: 'quota',
    });
  });

  it('reports storage as unavailable rather than retrying, when writes always throw', () => {
    // Safari private browsing throws on EVERY write. Shrinking the store and
    // retrying would burn cycles and evict the user's history for nothing.
    const setItem = vi.fn(() => {
      throw new Error('SecurityError');
    });

    const store: PersistedStore = {
      v: SCHEMA_VERSION,
      conversations: [
        conversation({ id: 'a', nodes: chain(2, 'a') }),
        conversation({ id: 'b', nodes: chain(2, 'b') }),
      ],
      folders: [],
      activeId: 'a',
    };

    expect(persist(store, { ...memoryStorage(), setItem })).toEqual({
      ok: false,
      reason: 'unavailable',
    });
    expect(setItem).toHaveBeenCalledOnce();
  });

  it('does not throw when persistence fails', () => {
    // Losing persistence is survivable. Breaking the send path over it is not.
    const setItem = vi.fn(() => {
      throw new Error('nope');
    });
    expect(() =>
      persist(
        { v: SCHEMA_VERSION, conversations: [], folders: [], activeId: null },
        { ...memoryStorage(), setItem },
      ),
    ).not.toThrow();
  });
});
