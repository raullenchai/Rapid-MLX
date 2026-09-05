import { activePath, subtree } from '@/chat/MessageTree';
import { HISTORY_KEY } from './migrate';
import type { Conversation, MessageNode, PersistedStore } from './types';

/**
 * Writing the store to localStorage, under caps.
 *
 * The caps are not housekeeping: localStorage is ~5 MB per origin and THROWS
 * on overflow, so an unbounded store does not degrade — it silently stops
 * persisting anything, including the conversation in progress.
 *
 * Caps apply AT WRITE TIME and never mutate the in-memory store: a user who
 * scrolled into an old branch keeps seeing it for the session.
 */

export const MAX_CONVERSATIONS = 30;
/** Nodes on the visible path. */
export const MAX_ACTIVE_PATH = 200;
/** Nodes in total, including every off-path alternative. */
export const MAX_NODES = 400;
/** Attempts to shrink the store in response to a quota failure. */
const QUOTA_RETRIES = 5;

/**
 * Trim one conversation's node bag. Order matters:
 *
 *   1. The active path is capped first and RE-ROOTED, so the visible
 *      transcript is always whole.
 *   2. Everything reachable from a retained node is retained with it, so a
 *      switchable branch does not lose its continuation.
 *   3. If still too large, whole off-path subtrees go, oldest first. Never
 *      split one — half a branch is a conversation that never happened.
 *   4. Stale `branchChoices` are pruned last, once survivors are known.
 */
export function capConversation(conversation: Conversation): Conversation {
  const { nodes, activeLeafId, branchChoices } = conversation;
  if (nodes.length <= MAX_NODES) {
    const path = activePath(nodes, activeLeafId, branchChoices);
    if (path.length <= MAX_ACTIVE_PATH) return conversation;
  }

  const path = activePath(nodes, activeLeafId, branchChoices);
  const keptPath = path.length > MAX_ACTIVE_PATH ? path.slice(-MAX_ACTIVE_PATH) : path;

  const retained = new Set<string>();
  for (const node of keptPath) {
    for (const id of subtree(node.id, nodes)) retained.add(id);
  }

  // 3. Still too big: drop whole off-path subtrees, oldest root first.
  if (retained.size > MAX_NODES) {
    const onPath = new Set(keptPath.map((node) => node.id));
    // An off-path subtree root is a retained node whose parent is on the path
    // but which is not itself on the path — i.e. the head of an alternative.
    const alternatives = nodes
      .filter(
        (node) =>
          retained.has(node.id) &&
          !onPath.has(node.id) &&
          node.parentId !== null &&
          onPath.has(node.parentId),
      )
      .sort((a, b) => a.createdAt - b.createdAt);

    for (const alternative of alternatives) {
      if (retained.size <= MAX_NODES) break;
      for (const id of subtree(alternative.id, nodes)) retained.delete(id);
    }
  }

  const keptNodes: MessageNode[] = [];
  const newRootId = keptPath[0]?.id ?? null;
  for (const node of nodes) {
    if (!retained.has(node.id)) continue;
    // Re-root: the new first node's parent was dropped, so keeping the link
    // would strand the whole conversation behind a dangling reference.
    keptNodes.push(node.id === newRootId ? { ...node, parentId: null } : node);
  }

  const present = new Set(keptNodes.map((node) => node.id));
  const choices: Record<string, string> = {};
  for (const [parent, child] of Object.entries(branchChoices)) {
    if (present.has(parent) && present.has(child)) choices[parent] = child;
  }

  const leaf = activeLeafId !== null && present.has(activeLeafId) ? activeLeafId : null;

  return {
    ...conversation,
    nodes: keptNodes,
    activeLeafId: leaf,
    branchChoices: choices,
  };
}

/**
 * Choose which conversations to keep.
 *
 * Pinned rows are never dropped ahead of an unpinned one: a pin is an explicit
 * instruction, and silently discarding one would be the single most
 * infuriating thing this cap could do. If the user has pinned more than the
 * cap allows, the cap yields — a pinned conversation is kept regardless.
 */
export function capConversations(conversations: Conversation[]): Conversation[] {
  const ordered = [...conversations].sort((a, b) => {
    if (a.isPinned !== b.isPinned) return a.isPinned ? -1 : 1;
    return b.updatedAt - a.updatedAt;
  });

  const pinned = ordered.filter((conversation) => conversation.isPinned).length;
  return ordered.slice(0, Math.max(MAX_CONVERSATIONS, pinned));
}

/** Apply every cap. Pure, so the whole policy is testable in isolation. */
export function capStore(store: PersistedStore): PersistedStore {
  const conversations = capConversations(store.conversations).map(capConversation);
  const present = new Set(conversations.map((conversation) => conversation.id));
  return {
    ...store,
    conversations,
    activeId:
      store.activeId !== null && present.has(store.activeId)
        ? store.activeId
        : (conversations[0]?.id ?? null),
  };
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export type PersistOutcome =
  | { ok: true }
  /** Written, but conversations had to be dropped to make it fit. */
  | { ok: true; evicted: number }
  | { ok: false; reason: 'quota' | 'unavailable' };

/**
 * Write the store, shrinking it if the quota refuses.
 *
 * On `QuotaExceededError` the OLDEST UNPINNED conversation is dropped and the
 * write retried. Failing outright is survivable — the session keeps working,
 * it just stops being remembered — but breaking the send path over a storage
 * failure is not, so nothing here throws (index.html:1010-1014 took the same
 * position).
 */
export function persist(store: PersistedStore, storage: StorageLike): PersistOutcome {
  let candidate = capStore(store);
  let evicted = 0;

  for (let attempt = 0; attempt <= QUOTA_RETRIES; attempt += 1) {
    try {
      storage.setItem(HISTORY_KEY, JSON.stringify(candidate));
      return evicted > 0 ? { ok: true, evicted } : { ok: true };
    } catch (cause) {
      if (!isQuotaError(cause)) return { ok: false, reason: 'unavailable' };

      const survivors = dropOldestUnpinned(candidate.conversations);
      if (survivors === null) return { ok: false, reason: 'quota' };
      evicted += 1;
      candidate = capStore({ ...candidate, conversations: survivors });
    }
  }

  return { ok: false, reason: 'quota' };
}

/** Null when there is nothing left that may be dropped. */
function dropOldestUnpinned(conversations: Conversation[]): Conversation[] | null {
  let oldest: Conversation | null = null;
  for (const conversation of conversations) {
    if (conversation.isPinned) continue;
    if (oldest === null || conversation.updatedAt < oldest.updatedAt) oldest = conversation;
  }
  if (oldest === null) return null;
  return conversations.filter((conversation) => conversation.id !== oldest.id);
}

/**
 * Is this the storage quota, or is storage simply unavailable?
 *
 * The distinction matters because the responses differ: a quota failure is
 * worth retrying with less data, while Safari private browsing throws on
 * EVERY write and retrying just burns cycles. Browsers disagree on the name
 * and the legacy code numbers, hence the breadth.
 */
function isQuotaError(cause: unknown): boolean {
  if (!(cause instanceof Error)) return false;
  if (cause.name === 'QuotaExceededError') return true;
  if (cause.name === 'NS_ERROR_DOM_QUOTA_REACHED') return true;
  const code = (cause as { code?: number }).code;
  return code === 22 || code === 1014;
}
