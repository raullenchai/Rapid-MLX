import { describe, expect, it } from 'vitest';
import {
  groupConversations,
  searchConversations,
  searchTerms,
} from './ConversationSearch';
import type { Conversation, MessageNode } from '@/state/types';

const DAY = 24 * 60 * 60 * 1000;

function node(id: string, content: string): MessageNode {
  return {
    id,
    parentId: null,
    role: 'user',
    content,
    createdAt: 0,
    status: 'complete',
  };
}

function conversation(overrides: Partial<Conversation> & { id: string }): Conversation {
  return {
    title: '',
    nodes: [],
    activeLeafId: null,
    branchChoices: {},
    createdAt: 0,
    updatedAt: 0,
    isPinned: false,
    isArchived: false,
    hasCustomTitle: false,
    ...overrides,
  } as Conversation;
}

describe('searchTerms', () => {
  it('splits on whitespace and lowercases', () => {
    expect(searchTerms('  Metal   Shader ')).toEqual(['metal', 'shader']);
  });

  it('is empty for a blank query', () => {
    expect(searchTerms('   ')).toEqual([]);
  });
});

describe('searchConversations', () => {
  const rows = [
    conversation({
      id: 'a',
      title: 'Metal shaders',
      updatedAt: 3,
      nodes: [node('n1', 'how do I write a compute kernel')],
    }),
    conversation({
      id: 'b',
      title: 'Untitled',
      updatedAt: 2,
      nodes: [node('n2', 'the cache invalidation problem in swift')],
    }),
    conversation({ id: 'c', title: 'Groceries', updatedAt: 1 }),
  ];

  it('returns everything for a blank query, most recent first', () => {
    expect(searchConversations(rows, '  ').map((row) => row.id)).toEqual(['a', 'b', 'c']);
  });

  it('matches the title', () => {
    expect(searchConversations(rows, 'metal').map((row) => row.id)).toEqual(['a']);
  });

  it('matches message bodies, not just the title', () => {
    // Someone looking for "that shader thing" is likelier to remember a
    // phrase from the answer than the derived title.
    expect(searchConversations(rows, 'kernel').map((row) => row.id)).toEqual(['a']);
  });

  it('requires every term, but they may come from different fields', () => {
    // "swift cache" spans the title and a message; both must match somewhere.
    expect(searchConversations(rows, 'swift cache').map((row) => row.id)).toEqual(['b']);
    expect(searchConversations(rows, 'swift metal')).toEqual([]);
  });

  it('is case insensitive', () => {
    expect(searchConversations(rows, 'METAL').map((row) => row.id)).toEqual(['a']);
  });

  it('searches every branch, not only the active path', () => {
    // An answer regenerated away is exactly what a user comes back looking
    // for. `nodes` holds the whole tree, so an off-path node must match.
    const branched = conversation({
      id: 'd',
      title: 'Untitled',
      activeLeafId: 'live',
      nodes: [node('live', 'the visible answer'), node('orphan', 'the abandoned answer')],
    });
    expect(searchConversations([branched], 'abandoned').map((row) => row.id)).toEqual(['d']);
  });

  it('includes archived conversations', () => {
    // Search is the archive's recovery path; filtering here would strand it.
    const archived = conversation({ id: 'e', title: 'Old notes', isArchived: true });
    expect(searchConversations([archived], 'notes').map((row) => row.id)).toEqual(['e']);
  });
});

describe('groupConversations', () => {
  const now = Date.now();

  it('puts pinned conversations in their own leading section', () => {
    const sections = groupConversations(
      [
        conversation({ id: 'a', title: 'recent', updatedAt: now }),
        conversation({ id: 'b', title: 'pinned but old', updatedAt: now - 10 * DAY, isPinned: true }),
      ],
      now,
    );

    expect(sections[0]?.label).toBe('Pinned');
    expect(sections[0]?.rows.map((row) => row.id)).toEqual(['b']);
    // And the pinned one is NOT also listed under its date group.
    expect(sections.slice(1).flatMap((s) => s.rows.map((row) => row.id))).toEqual(['a']);
  });

  it('buckets by date and sorts each bucket by recency', () => {
    const sections = groupConversations(
      [
        conversation({ id: 'older', updatedAt: now - 3 * DAY }),
        conversation({ id: 'newer', updatedAt: now - 2 * DAY }),
        conversation({ id: 'today', updatedAt: now }),
      ],
      now,
    );

    expect(sections[0]?.label).toBe('Today');
    expect(sections[0]?.rows.map((row) => row.id)).toEqual(['today']);
    const week = sections.find((section) => section.label === 'Previous 7 days');
    expect(week?.rows.map((row) => row.id)).toEqual(['newer', 'older']);
  });

  it('omits empty sections', () => {
    const sections = groupConversations([conversation({ id: 'a', updatedAt: now })], now);
    expect(sections.map((section) => section.label)).toEqual(['Today']);
  });

  it('returns nothing for an empty list', () => {
    expect(groupConversations([], now)).toEqual([]);
  });
});
