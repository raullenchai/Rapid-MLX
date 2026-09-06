import type { Conversation } from '@/state/types';
import { dateGroupOf, type DateGroup } from '@/lib/format';

/**
 * Conversation search and grouping.
 *
 * Ported from `apps/rapid-mac/Sources/Rapid/Chat/ConversationSearch.swift`,
 * including the two decisions that are easy to get wrong:
 *
 *   - **All terms, anywhere.** Each whitespace-delimited term matches the
 *     title OR any message, and they may land in different messages — which is
 *     what makes "swift cache" useful across two turns.
 *   - **Every branch, not just the visible path.** An answer you regenerated
 *     away is exactly what you come back looking for.
 *
 * Search stays local: it reads the already-loaded history and sends nothing.
 */

const GROUP_ORDER: DateGroup[] = [
  'Today',
  'Yesterday',
  'Previous 7 days',
  'Previous 30 days',
  'Older',
];

export interface SearchSection {
  label: string;
  rows: Conversation[];
}

/** Split a query into terms, dropping the empties an extra space produces. */
export function searchTerms(query: string): string[] {
  return query
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((term) => term !== '');
}

/** Conversations matching every term, most recently updated first. */
export function searchConversations(
  conversations: Conversation[],
  query: string,
): Conversation[] {
  const terms = searchTerms(query);
  const matched =
    terms.length === 0
      ? [...conversations]
      : conversations.filter((conversation) => {
          const haystack = [
            conversation.title,
            ...conversation.nodes.map((node) => node.content),
          ]
            .join('\n')
            .toLowerCase();
          return terms.every((term) => haystack.includes(term));
        });
  return byRecency(matched);
}

/**
 * Group into date sections, pinned first.
 *
 * Archived conversations are NOT filtered out here — the Mac app takes the
 * same position, because search is an archived conversation's direct recovery
 * path. Callers that show a plain list filter before calling.
 */
export function groupConversations(
  conversations: Conversation[],
  now: number,
): SearchSection[] {
  const pinned = conversations.filter((conversation) => conversation.isPinned);
  const rest = conversations.filter((conversation) => !conversation.isPinned);

  const buckets = new Map<DateGroup, Conversation[]>();
  for (const conversation of rest) {
    const group = dateGroupOf(conversation.updatedAt, now);
    const list = buckets.get(group) ?? [];
    list.push(conversation);
    buckets.set(group, list);
  }

  const sections: SearchSection[] = [];
  if (pinned.length > 0) sections.push({ label: 'Pinned', rows: byRecency(pinned) });
  for (const group of GROUP_ORDER) {
    const rows = buckets.get(group);
    if (rows?.length) sections.push({ label: group, rows: byRecency(rows) });
  }
  return sections;
}

export function byRecency(rows: Conversation[]): Conversation[] {
  return [...rows].sort((a, b) => b.updatedAt - a.updatedAt);
}
