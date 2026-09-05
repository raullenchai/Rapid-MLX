import { describe, expect, it, vi } from 'vitest';
import { activePath } from '@/chat/MessageTree';
import { SCHEMA_VERSION } from './types';
import { detect, deriveTitle, migrate, v1ToV2, v2ToV3 } from './migrate';

/** A v1 blob: a bare array of messages, no envelope. */
const v1 = [
  { role: 'user', content: 'what is metal' },
  {
    role: 'assistant',
    content: 'A graphics API.',
    stats: { ttft: 120, tokens: 4, tps: 9.1 },
  },
];

/** A v2 blob. */
const v2 = {
  conversations: [
    {
      id: 'conv-1',
      title: 'Shader question',
      updatedAt: 1_700_000_000_000,
      messages: [
        { role: 'user', content: 'explain shaders' },
        { role: 'assistant', content: 'They run on the GPU.' },
      ],
    },
  ],
  activeId: 'conv-1',
};

describe('detect', () => {
  it('recognises a bare array as v1', () => {
    expect(detect([])).toBe(1);
    expect(detect(v1)).toBe(1);
  });

  it('recognises the versionless envelope as v2', () => {
    expect(detect(v2)).toBe(2);
  });

  it('recognises the current version', () => {
    expect(detect({ v: SCHEMA_VERSION, conversations: [] })).toBe(3);
  });

  it('recognises a newer version as future', () => {
    expect(detect({ v: SCHEMA_VERSION + 1, conversations: [] })).toBe('future');
  });

  it.each([null, 42, 'string', true, {}, { conversations: 'not an array' }])(
    'rejects %o as unusable',
    (value) => {
      expect(detect(value)).toBe('unusable');
    },
  );
});

describe('deriveTitle', () => {
  it('uses the first user message', () => {
    expect(
      deriveTitle([
        { role: 'assistant', content: 'hello' },
        { role: 'user', content: 'a question' },
      ]),
    ).toBe('a question');
  });

  it('collapses whitespace', () => {
    expect(deriveTitle([{ role: 'user', content: '  spread   over\n lines ' }])).toBe(
      'spread over lines',
    );
  });

  it('truncates at 42 characters with an ellipsis', () => {
    const long = 'x'.repeat(80);
    const title = deriveTitle([{ role: 'user', content: long }]);
    expect(title).toHaveLength(43);
    expect(title.endsWith('…')).toBe(true);
  });

  it('skips a blank user message', () => {
    expect(
      deriveTitle([
        { role: 'user', content: '   ' },
        { role: 'user', content: 'real one' },
      ]),
    ).toBe('real one');
  });

  it('is empty when there is no user message', () => {
    expect(deriveTitle([{ role: 'assistant', content: 'unprompted' }])).toBe('');
  });
});

describe('v1 -> v2', () => {
  it('wraps the array in one conversation and makes it active', () => {
    const result = v1ToV2(v1);
    expect(result.conversations).toHaveLength(1);
    expect(result.activeId).toBe(result.conversations[0]?.id);
  });

  it('produces an empty store for an empty array', () => {
    expect(v1ToV2([])).toEqual({ conversations: [], activeId: null });
  });
});

describe('v2 -> v3', () => {
  it('turns a flat message list into a degenerate chain', () => {
    const store = v2ToV3(v2);
    const conversation = store.conversations[0];
    expect(conversation).toBeDefined();
    expect(conversation?.nodes).toHaveLength(2);
    // First node is a root, second hangs off it.
    expect(conversation?.nodes[0]?.parentId).toBeNull();
    expect(conversation?.nodes[1]?.parentId).toBe(conversation?.nodes[0]?.id);
  });

  it('points activeLeafId at the last turn, so the whole transcript renders', () => {
    const conversation = v2ToV3(v2).conversations[0];
    const path = activePath(
      conversation?.nodes ?? [],
      conversation?.activeLeafId ?? null,
      conversation?.branchChoices,
    );
    expect(path.map((n) => n.content)).toEqual(['explain shaders', 'They run on the GPU.']);
  });

  it('gives every node a distinct, increasing createdAt', () => {
    // Stamping them identically would make sibling order fall through to the
    // id tie-break, i.e. to a random string comparison, and the ‹2/3› index
    // would depend on generated ids rather than on when turns happened.
    const nodes = v2ToV3(v2).conversations[0]?.nodes ?? [];
    const stamps = nodes.map((n) => n.createdAt);
    expect(new Set(stamps).size).toBe(stamps.length);
    expect([...stamps].sort((a, b) => a - b)).toEqual(stamps);
  });

  it('preserves stats and reasoning', () => {
    const store = v2ToV3(v1ToV2(v1));
    const answer = store.conversations[0]?.nodes[1];
    expect(answer?.stats).toEqual({
      ttftMs: 120,
      tokens: 4,
      tps: 9.1,
      tokensEstimated: false,
    });
  });

  it('DROPS a synthetic error role rather than coercing it', () => {
    // The old page could persist an `error` row. It must never become a wire
    // turn — the model would be asked to continue from an error message.
    const store = v2ToV3({
      conversations: [
        {
          id: 'c',
          messages: [
            { role: 'user', content: 'hi' },
            { role: 'error', content: 'the engine failed' },
            { role: 'assistant', content: 'hello' },
          ],
        },
      ],
    });
    const contents = store.conversations[0]?.nodes.map((n) => n.content);
    expect(contents).toEqual(['hi', 'hello']);
  });

  it('derives a title for a conversation that has none', () => {
    // The regression this pins: a transcript brought forward from v1 never
    // passes through the "touch on write" path, so an underived title left it
    // labelled "New chat" in the drawer despite having messages.
    const store = v2ToV3(v1ToV2(v1));
    expect(store.conversations[0]?.title).toBe('what is metal');
    expect(store.conversations[0]?.title).not.toBe('');
  });

  it('keeps an existing v2 title', () => {
    expect(v2ToV3(v2).conversations[0]?.title).toBe('Shader question');
  });

  it('does not mark a migrated title as custom', () => {
    // A v2 title was itself auto-derived. Marking it custom would stop the
    // conversation ever being re-titled by its own first message.
    expect(v2ToV3(v2).conversations[0]?.hasCustomTitle).toBe(false);
  });

  it('drops a malformed conversation individually, not the whole store', () => {
    const store = v2ToV3({
      conversations: [
        { id: 'good', messages: [{ role: 'user', content: 'kept' }] },
        { id: 'bad', messages: 'not an array' },
        null,
        { id: 'also-good', messages: [{ role: 'user', content: 'also kept' }] },
      ],
      activeId: 'good',
    });
    expect(store.conversations.map((c) => c.id)).toEqual(['good', 'also-good']);
  });

  it('repairs an activeId that points at a dropped conversation', () => {
    const store = v2ToV3({
      conversations: [{ id: 'kept', messages: [{ role: 'user', content: 'x' }] }],
      activeId: 'gone',
    });
    expect(store.activeId).toBe('kept');
  });
});

describe('migrate', () => {
  it('returns an empty writable store for missing data', () => {
    const result = migrate(null);
    expect(result.store.conversations).toEqual([]);
    expect(result.writable).toBe(true);
  });

  it('migrates v1 end to end', () => {
    const result = migrate(JSON.stringify(v1));
    expect(result.detected).toBe(1);
    expect(result.store.v).toBe(SCHEMA_VERSION);
    expect(result.store.conversations).toHaveLength(1);
    expect(result.store.conversations[0]?.title).toBe('what is metal');
  });

  it('migrates v2 end to end', () => {
    const result = migrate(JSON.stringify(v2));
    expect(result.detected).toBe(2);
    expect(result.store.conversations[0]?.nodes).toHaveLength(2);
  });

  it('passes a v3 store through', () => {
    const migrated = migrate(JSON.stringify(v2)).store;
    const round = migrate(JSON.stringify(migrated));
    expect(round.detected).toBe(3);
    expect(round.store.conversations[0]?.nodes.map((n) => n.content)).toEqual([
      'explain shaders',
      'They run on the GPU.',
    ]);
  });

  it('is idempotent — migrating twice changes nothing', () => {
    const once = migrate(JSON.stringify(v2)).store;
    const twice = migrate(JSON.stringify(once)).store;
    expect(twice).toEqual(once);
  });

  it('refuses to write over a FUTURE schema', () => {
    // The user downgraded. Silently clobbering data this build cannot even
    // represent is unrecoverable, so the session runs in memory instead.
    const result = migrate(JSON.stringify({ v: 99, conversations: [{ anything: true }] }));
    expect(result.detected).toBe('future');
    expect(result.writable).toBe(false);
    expect(result.store.conversations).toEqual([]);
  });

  it('backs up an unparseable blob before starting clean', () => {
    const backup = vi.fn();
    const result = migrate('{ this is not json', backup);
    expect(backup).toHaveBeenCalledWith('{ this is not json');
    expect(result.backedUp).toBe(true);
    expect(result.store.conversations).toEqual([]);
  });

  it('backs up a parseable but unrecognised shape', () => {
    const backup = vi.fn();
    migrate(JSON.stringify({ something: 'else' }), backup);
    expect(backup).toHaveBeenCalledOnce();
  });

  it('survives a backup function that throws', () => {
    // Storage throwing on write is the NORMAL case here, not an edge one:
    // Safari private browsing throws on every write, and a full quota is
    // exactly the situation that produces an unusable blob. Losing the backup
    // is survivable; failing to boot over it is not.
    const throwing = () => {
      throw new Error('QuotaExceededError');
    };
    const result = migrate('nonsense', throwing);
    expect(result.store.conversations).toEqual([]);
    expect(result.writable).toBe(true);
    expect(result.backedUp).toBe(false);
  });

  it('repairs a v3 store whose parent links dangle', () => {
    // Reachable by hand-editing in a devtools console, or by a write
    // truncated mid-flight. It has to render, not throw.
    const result = migrate(
      JSON.stringify({
        v: SCHEMA_VERSION,
        conversations: [
          {
            id: 'c',
            nodes: [
              {
                id: 'a',
                parentId: 'ghost',
                role: 'user',
                content: 'orphan',
                status: 'complete',
                createdAt: 1,
              },
              {
                id: 'b',
                parentId: 'a',
                role: 'assistant',
                content: 'child',
                status: 'complete',
                createdAt: 2,
              },
            ],
            activeLeafId: 'b',
          },
        ],
        activeId: 'c',
      }),
    );
    const conversation = result.store.conversations[0];
    expect(conversation?.nodes[0]?.parentId).toBeNull();
    expect(activePath(conversation?.nodes ?? [], 'b').map((n) => n.content)).toEqual([
      'orphan',
      'child',
    ]);
  });

  it('drops a v3 conversation with no usable nodes', () => {
    const result = migrate(
      JSON.stringify({
        v: SCHEMA_VERSION,
        conversations: [{ id: 'empty', nodes: [] }, { id: 'nonsense' }],
      }),
    );
    expect(result.store.conversations).toEqual([]);
  });
});
