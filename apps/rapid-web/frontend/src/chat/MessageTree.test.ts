import { describe, expect, it } from 'vitest';
import type { MessageNode } from '@/state/types';
import {
  activePath,
  branchAnchor,
  children,
  choicesAlong,
  deduplicateById,
  deepestLeaf,
  defaultLeaf,
  deleteConfirmationTitle,
  deletionImpact,
  precedes,
  promoteOrphans,
  repairLegacyChain,
  siblings,
  subtree,
} from './MessageTree';

let clock = 1000;

function node(
  id: string,
  parentId: string | null,
  overrides: Partial<MessageNode> = {},
): MessageNode {
  return {
    id,
    parentId,
    role: 'assistant',
    content: id,
    status: 'complete',
    createdAt: (clock += 1000),
    ...overrides,
  };
}

/** A linear three-turn transcript: u1 -> a1 -> u2 -> a2. */
function linear(): MessageNode[] {
  clock = 1000;
  return [
    node('u1', null, { role: 'user' }),
    node('a1', 'u1'),
    node('u2', 'a1', { role: 'user' }),
    node('a2', 'u2'),
  ];
}

/** u1 with three alternative answers, the second of which continues. */
function branched(): MessageNode[] {
  clock = 1000;
  return [
    node('u1', null, { role: 'user' }),
    node('a1', 'u1'),
    node('a2', 'u1'),
    node('a3', 'u1'),
    node('u2', 'a2', { role: 'user' }),
    node('a4', 'u2'),
  ];
}

const ids = (nodes: MessageNode[]) => nodes.map((n) => n.id);

describe('precedes', () => {
  it('orders by createdAt', () => {
    const a = node('a', null, { createdAt: 1 });
    const b = node('b', null, { createdAt: 2 });
    expect(precedes(a, b)).toBe(true);
    expect(precedes(b, a)).toBe(false);
  });

  it('breaks a tie on id, so the order is stable across reloads', () => {
    // Two nodes stamped inside the same millisecond must still order
    // deterministically, or the ‹2/3› index jumps between sessions.
    const a = node('aaa', null, { createdAt: 5 });
    const b = node('bbb', null, { createdAt: 5 });
    expect(precedes(a, b)).toBe(true);
    expect(precedes(b, a)).toBe(false);
  });
});

describe('deduplicateById', () => {
  it('keeps the first occurrence', () => {
    const first = node('x', null, { content: 'first' });
    const second = node('x', null, { content: 'second' });
    const result = deduplicateById([first, second]);
    expect(result).toHaveLength(1);
    expect(result[0]?.content).toBe('first');
  });

  it('leaves a clean tree untouched', () => {
    const nodes = linear();
    expect(deduplicateById(nodes)).toEqual(nodes);
  });
});

describe('children and siblings', () => {
  it('returns the roots for a null parent', () => {
    expect(ids(children(null, branched()))).toEqual(['u1']);
  });

  it('returns children in sibling order', () => {
    expect(ids(children('u1', branched()))).toEqual(['a1', 'a2', 'a3']);
  });

  it('includes the node itself in its sibling group', () => {
    expect(ids(siblings('a2', branched()))).toEqual(['a1', 'a2', 'a3']);
  });

  it('returns a single-element group for a node with no alternatives', () => {
    // This is the signal the UI uses to hide the switcher rather than render
    // a permanent ‹1/1›.
    expect(ids(siblings('a1', linear()))).toEqual(['a1']);
  });

  it('returns nothing for an id that is not present', () => {
    expect(siblings('missing', linear())).toEqual([]);
  });
});

describe('deepestLeaf', () => {
  it('walks to the tip', () => {
    expect(deepestLeaf('u1', linear())).toBe('a2');
  });

  it('prefers the newest child when it has no instruction', () => {
    expect(deepestLeaf('u1', branched())).toBe('a3');
  });

  it('honours a remembered choice', () => {
    // Returning to a branch must land where the user left it, not at the tip
    // of whichever branch happens to be newest.
    expect(deepestLeaf('u1', branched(), { u1: 'a2' })).toBe('a4');
  });

  it('degrades to the newest child when the remembered choice is stale', () => {
    // A branch deleted since the choice was recorded, or a hand-edited store.
    expect(deepestLeaf('u1', branched(), { u1: 'deleted' })).toBe('a3');
  });

  it('does not hang on a cycle', () => {
    // Only reachable from a corrupt store, but walking one would hang the tab.
    clock = 1000;
    const cyclic = [node('a', 'b'), node('b', 'a')];
    expect(() => deepestLeaf('a', cyclic)).not.toThrow();
    expect(deepestLeaf('a', cyclic)).toBeTypeOf('string');
  });

  it('returns the id itself when it has no children', () => {
    expect(deepestLeaf('a2', linear())).toBe('a2');
  });
});

describe('defaultLeaf', () => {
  it('is null for an empty tree', () => {
    expect(defaultLeaf([])).toBeNull();
  });

  it('reduces to the last message for a linear transcript', () => {
    expect(defaultLeaf(linear())).toBe('a2');
  });

  it('lands on the tip of the branch most recently worked in', () => {
    expect(defaultLeaf(branched())).toBe('a4');
  });
});

describe('activePath', () => {
  it('returns the root-to-leaf path, oldest first', () => {
    expect(ids(activePath(linear(), 'a2'))).toEqual(['u1', 'a1', 'u2', 'a2']);
  });

  it('follows the requested branch', () => {
    expect(ids(activePath(branched(), 'a1'))).toEqual(['u1', 'a1']);
    expect(ids(activePath(branched(), 'a4'))).toEqual(['u1', 'a2', 'u2', 'a4']);
  });

  it('falls back to the default leaf when the pointer does not resolve', () => {
    // Rendering an empty conversation because a pointer went stale would look
    // exactly like data loss.
    expect(ids(activePath(branched(), 'deleted'))).toEqual(['u1', 'a2', 'u2', 'a4']);
  });

  it('falls back when the pointer is null', () => {
    expect(ids(activePath(linear(), null))).toEqual(['u1', 'a1', 'u2', 'a2']);
  });

  it('is empty for an empty tree', () => {
    expect(activePath([], 'anything')).toEqual([]);
  });

  it('does not hang on a corrupt parent chain', () => {
    clock = 1000;
    const cyclic = [node('a', 'b'), node('b', 'a')];
    expect(() => activePath(cyclic, 'a')).not.toThrow();
  });

  it('keeps the first copy when ids are duplicated', () => {
    const nodes = [...linear(), node('a2', 'u2', { content: 'duplicate' })];
    const path = activePath(nodes, 'a2');
    expect(path[path.length - 1]?.content).toBe('a2');
  });
});

describe('choicesAlong', () => {
  it('records one edge per fork on the path', () => {
    const path = activePath(branched(), 'a4');
    expect(choicesAlong(path)).toEqual({ u1: 'a2', a2: 'u2', u2: 'a4' });
  });

  it('produces no edge for a root', () => {
    expect(choicesAlong([node('root', null)])).toEqual({});
  });
});

describe('subtree', () => {
  it('collects the node and everything beneath it', () => {
    expect(subtree('a2', branched())).toEqual(new Set(['a2', 'u2', 'a4']));
  });

  it('crosses every branch, not just the visible one', () => {
    // After a few regenerations some of these turns are off screen, which is
    // exactly why deletion has to count them.
    expect(subtree('u1', branched())).toEqual(new Set(['u1', 'a1', 'a2', 'a3', 'u2', 'a4']));
  });

  it('is just the node itself for a leaf', () => {
    expect(subtree('a4', branched())).toEqual(new Set(['a4']));
  });

  it('does not hang on a cycle', () => {
    clock = 1000;
    const cyclic = [node('a', 'b'), node('b', 'a')];
    expect(() => subtree('a', cyclic)).not.toThrow();
  });
});

describe('repairLegacyChain', () => {
  it('links a flat transcript into a degenerate tree', () => {
    clock = 1000;
    const flat = [node('m1', null), node('m2', null), node('m3', null)];
    expect(repairLegacyChain(flat).map((n) => n.parentId)).toEqual([null, 'm1', 'm2']);
  });

  it('leaves a PARTIALLY linked tree alone', () => {
    // The rule is keyed on "no node has a parent", not "some node lacks one":
    // in the branching model a user who edits the opening prompt legitimately
    // owns several parentless roots, so this is already a real tree.
    clock = 1000;
    const partial = [node('r1', null), node('r2', null), node('c1', 'r1')];
    expect(repairLegacyChain(partial)).toEqual(partial);
  });

  it('leaves a single node alone', () => {
    const single = [node('only', null)];
    expect(repairLegacyChain(single)).toEqual(single);
  });

  it('leaves an empty array alone', () => {
    expect(repairLegacyChain([])).toEqual([]);
  });
});

describe('promoteOrphans', () => {
  it('promotes a node whose parent is absent', () => {
    // A dangling parent strands the whole subtree outside every path: present
    // in storage, invisible on screen, counted by nothing.
    clock = 1000;
    const orphaned = [node('a', 'ghost'), node('b', 'a')];
    const repaired = promoteOrphans(orphaned);
    expect(repaired[0]?.parentId).toBeNull();
    expect(repaired[1]?.parentId).toBe('a');
    expect(ids(activePath(repaired, 'b'))).toEqual(['a', 'b']);
  });

  it('leaves a well-formed tree untouched', () => {
    const nodes = linear();
    expect(promoteOrphans(nodes)).toEqual(nodes);
  });
});

describe('branchAnchor', () => {
  it('is the node itself when it directly follows a user turn', () => {
    expect(branchAnchor('a2', branched())).toBe('a2');
  });

  it('walks back to the first node after the owning user turn', () => {
    // Every row of a multi-part answer must pivot on the same fork, or the
    // switcher on row two offers row two's siblings instead of the answer's.
    clock = 1000;
    const multipart = [node('u1', null, { role: 'user' }), node('a1', 'u1'), node('a1b', 'a1')];
    expect(branchAnchor('a1b', multipart)).toBe('a1');
  });

  it('is the user turn itself for a user row', () => {
    expect(branchAnchor('u2', branched())).toBe('u2');
  });

  it('returns null for an id that is not present', () => {
    expect(branchAnchor('missing', linear())).toBeNull();
  });
});

describe('deletionImpact and its confirmation copy', () => {
  it('counts the node plus its whole subtree', () => {
    expect(deletionImpact('a4', branched())).toBe(1);
    expect(deletionImpact('a2', branched())).toBe(3);
    expect(deletionImpact('u1', branched())).toBe(6);
  });

  it('asks a plain question for a lone message', () => {
    expect(deleteConfirmationTitle(1)).toBe('Delete this message?');
  });

  it('states the cost, and gets the plural right', () => {
    expect(deleteConfirmationTitle(2)).toBe('Delete this message and the 1 turn below it?');
    expect(deleteConfirmationTitle(3)).toBe('Delete this message and the 2 turns below it?');
    expect(deleteConfirmationTitle(7)).toBe('Delete this message and the 6 turns below it?');
  });
});
