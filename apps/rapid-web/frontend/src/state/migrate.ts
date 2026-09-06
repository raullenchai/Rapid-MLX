import { deduplicateById, promoteOrphans, repairLegacyChain } from '@/chat/MessageTree';
import { newId } from '@/lib/ids';
import {
  SCHEMA_VERSION,
  type Conversation,
  type MessageNode,
  type PersistedStore,
  type Role,
} from './types';

/**
 * The localStorage migration chain.
 *
 * Four shapes have existed under `rapid-mlx-web.history`:
 *
 *   v1  Message[]                                          (M1, no envelope)
 *   v2  { conversations: [{id,title,updatedAt,messages}],  (M4, no version
 *        activeId }                                         field)
 *   v3  { v: 3, conversations: Conversation[],
 *        folders: Folder[], activeId }
 *   v4  v3 plus the `tool` role and a node's tool envelope  (this build)
 *
 * v3 is the first shape to carry its own version, so the older two have to be
 * sniffed structurally. v3 -> v4 is a pure widening: the new fields are
 * optional and v3 nodes already validate, so no node is rewritten.
 *
 * THE MIGRATION IS ONE-WAY. Once v4 is written an older build sees a `v` it
 * does not know and returns an empty store, so history APPEARS deleted on a
 * downgrade. The data is still there; the README says so and the JSON export
 * is the backup path.
 *
 * Pure, and deliberately not a store middleware: it must be callable from a
 * test without standing up a store.
 */

/**
 * NOT renamed when the package became `rmlx-web`, and must never be. A
 * localStorage key is where an existing user's transcripts already live —
 * renaming it does not move them, it makes the app read an empty slot with no
 * path back, which is the exact class of loss the chain below exists to avoid.
 */
export const HISTORY_KEY = 'rapid-mlx-web.history';
/** Where an unparseable blob is copied before anything overwrites it. */
export const HISTORY_BACKUP_KEY = 'rapid-mlx-web.history.bak';

export type DetectedVersion = 1 | 2 | 3 | 'future' | 'unusable';

/** The v1 shape: a bare array of messages, no envelope at all. */
interface LegacyMessage {
  role?: unknown;
  content?: unknown;
  reasoning?: unknown;
  stats?: { ttft?: unknown; tokens?: unknown; tps?: unknown };
}

/** The v2 shape. */
interface LegacyConversation {
  id?: unknown;
  title?: unknown;
  updatedAt?: unknown;
  messages?: unknown;
}

export function detect(parsed: unknown): DetectedVersion {
  if (Array.isArray(parsed)) return 1;
  if (parsed === null || typeof parsed !== 'object') return 'unusable';

  const envelope = parsed as { v?: unknown; conversations?: unknown };

  if (typeof envelope.v === 'number') {
    if (envelope.v === SCHEMA_VERSION) return 3;
    // A NEWER build wrote this and the user went back. Do not migrate, do not
    // overwrite: a silent clobber here is unrecoverable.
    if (envelope.v > SCHEMA_VERSION) return 'future';
    // A version below 3 that nonetheless carries `v` is a shape this build has
    // never written and cannot reason about.
    return 'unusable';
  }

  // v2 had no version field, so its own structure is the only signal.
  if (Array.isArray(envelope.conversations)) return 2;
  return 'unusable';
}

/**
 * Derive a title from the first user message.
 *
 * Must run AT CONSTRUCTION for a migrated conversation: a transcript brought
 * forward from v1 never passes through the "touch on write" path, so an empty
 * title labelled a restored conversation "New chat" despite it having
 * messages. Found in a browser, not by a unit test — hence the test below.
 */
export function deriveTitle(nodes: Array<{ role: Role; content: string }>): string {
  for (const node of nodes) {
    if (node.role !== 'user') continue;
    const collapsed = node.content.replace(/\s+/g, ' ').trim();
    if (collapsed === '') continue;
    return collapsed.length > 42 ? `${collapsed.slice(0, 42)}…` : collapsed;
  }
  return '';
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asRole(value: unknown): Role | null {
  // The old page could persist a synthetic `error` role. It must never become
  // a wire turn, so it is dropped rather than coerced into `assistant`.
  return value === 'user' || value === 'assistant' || value === 'system' ? value : null;
}

/**
 * Turn a legacy flat message list into a degenerate node chain.
 *
 * `createdAt` is synthesised BACKWARDS from the conversation's `updatedAt`, one
 * second apart. Stamping them all with the same value would make sibling order
 * fall through to the id tie-break, i.e. to a random string comparison, and the
 * ‹2/3› index would then depend on generated ids rather than on when the turns
 * actually happened.
 */
function nodesFromLegacyMessages(messages: unknown, updatedAt: number): MessageNode[] {
  if (!Array.isArray(messages)) return [];

  const usable = (messages as LegacyMessage[])
    .map((message) => ({ role: asRole(message?.role), message }))
    .filter((entry): entry is { role: Role; message: LegacyMessage } => entry.role !== null);

  let parentId: string | null = null;
  const nodes: MessageNode[] = [];

  usable.forEach((entry, index) => {
    const id = newId();
    const node: MessageNode = {
      id,
      parentId,
      role: entry.role,
      content: asString(entry.message.content),
      status: 'complete',
      createdAt: updatedAt - (usable.length - index) * 1000,
    };

    const reasoning = entry.message.reasoning;
    if (typeof reasoning === 'string' && reasoning !== '') node.reasoning = reasoning;

    const stats = entry.message.stats;
    if (stats && typeof stats.tokens === 'number') {
      node.stats = {
        ttftMs: typeof stats.ttft === 'number' ? stats.ttft : null,
        tokens: stats.tokens,
        tps: typeof stats.tps === 'number' ? stats.tps : null,
        // The old page had no such flag; treat a persisted count as measured.
        tokensEstimated: false,
      };
    }

    nodes.push(node);
    parentId = id;
  });

  return nodes;
}

function conversationFromLegacy(legacy: LegacyConversation): Conversation | null {
  const id = typeof legacy.id === 'string' ? legacy.id : newId();
  const updatedAt = typeof legacy.updatedAt === 'number' ? legacy.updatedAt : Date.now();

  const nodes = promoteOrphans(
    deduplicateById(nodesFromLegacyMessages(legacy.messages, updatedAt)),
  );
  if (nodes.length === 0) return null;

  const storedTitle = asString(legacy.title);
  const title = storedTitle !== '' ? storedTitle : deriveTitle(nodes);

  return {
    id,
    title,
    // A title carried over from v2 was itself auto-derived, so it must not be
    // marked custom — otherwise a migrated conversation could never be
    // re-titled by its own first message.
    hasCustomTitle: false,
    createdAt: nodes[0]?.createdAt ?? updatedAt,
    updatedAt,
    nodes,
    activeLeafId: nodes[nodes.length - 1]?.id ?? null,
    branchChoices: {},
    isPinned: false,
    isArchived: false,
    folderId: null,
  };
}

/** v1 (a bare message array) -> the v2 envelope. */
export function v1ToV2(messages: unknown[]): {
  conversations: LegacyConversation[];
  activeId: string | null;
} {
  if (messages.length === 0) return { conversations: [], activeId: null };
  const id = newId();
  return {
    conversations: [{ id, title: '', updatedAt: Date.now(), messages }],
    activeId: id,
  };
}

/** The v2 envelope -> v3. */
export function v2ToV3(legacy: { conversations?: unknown; activeId?: unknown }): PersistedStore {
  const rows = Array.isArray(legacy.conversations)
    ? (legacy.conversations as LegacyConversation[])
    : [];

  const conversations: Conversation[] = [];
  for (const row of rows) {
    // A row that fails to convert is dropped INDIVIDUALLY. Losing the whole
    // store because one conversation is malformed is the failure mode this
    // guards against.
    const converted = conversationFromLegacy(row ?? {});
    if (converted) conversations.push(converted);
  }

  const requested = typeof legacy.activeId === 'string' ? legacy.activeId : null;
  const activeId =
    requested !== null && conversations.some((c) => c.id === requested)
      ? requested
      : (conversations[0]?.id ?? null);

  return { v: SCHEMA_VERSION, conversations, folders: [], activeId };
}

/** Narrow a value that claims to already be v3. */
function validateV3(parsed: unknown): PersistedStore {
  const envelope = parsed as Partial<PersistedStore>;
  const rows = Array.isArray(envelope.conversations) ? envelope.conversations : [];

  const conversations: Conversation[] = [];
  for (const row of rows) {
    if (!row || typeof row !== 'object') continue;
    if (typeof row.id !== 'string' || !Array.isArray(row.nodes)) continue;

    // Repair before use, not on write: a store hand-edited in a devtools
    // console, or truncated by a quota failure mid-write, has to render.
    const nodes = promoteOrphans(repairLegacyChain(deduplicateById(row.nodes)));
    if (nodes.length === 0) continue;

    conversations.push({
      id: row.id,
      title: asString(row.title),
      hasCustomTitle: row.hasCustomTitle === true,
      createdAt:
        typeof row.createdAt === 'number' ? row.createdAt : (nodes[0]?.createdAt ?? Date.now()),
      updatedAt: typeof row.updatedAt === 'number' ? row.updatedAt : Date.now(),
      nodes,
      activeLeafId: typeof row.activeLeafId === 'string' ? row.activeLeafId : null,
      branchChoices:
        row.branchChoices && typeof row.branchChoices === 'object' ? row.branchChoices : {},
      isPinned: row.isPinned === true,
      isArchived: row.isArchived === true,
      folderId: typeof row.folderId === 'string' ? row.folderId : null,
      ...(typeof row.customInstructions === 'string'
        ? { customInstructions: row.customInstructions }
        : {}),
    });
  }

  const folders = Array.isArray(envelope.folders)
    ? envelope.folders.filter(
        (folder) =>
          folder &&
          typeof folder === 'object' &&
          typeof folder.id === 'string' &&
          typeof folder.name === 'string',
      )
    : [];

  const requested = typeof envelope.activeId === 'string' ? envelope.activeId : null;
  const activeId =
    requested !== null && conversations.some((c) => c.id === requested)
      ? requested
      : (conversations[0]?.id ?? null);

  return { v: SCHEMA_VERSION, conversations, folders, activeId };
}

export const EMPTY_STORE: PersistedStore = {
  v: SCHEMA_VERSION,
  conversations: [],
  folders: [],
  activeId: null,
};

export interface MigrationResult {
  store: PersistedStore;
  /**
   * False when the stored data came from a newer build. The caller must then
   * suppress every write to this key — overwriting a future schema loses data
   * that this build cannot even represent.
   */
  writable: boolean;
  /** True when an unusable blob was set aside under the backup key. */
  backedUp: boolean;
  detected: DetectedVersion;
}

/**
 * Set the raw blob aside, best-effort.
 *
 * Wrapped because storage itself is what usually fails here: Safari private
 * browsing throws on every write, and a quota that is already full is exactly
 * the situation that produced an unusable blob in the first place. Losing the
 * backup is survivable; failing to boot over it is not.
 */
function tryBackup(raw: string, backup?: (raw: string) => void): boolean {
  if (!backup) return false;
  try {
    backup(raw);
    return true;
  } catch {
    return false;
  }
}

/**
 * Migrate whatever is in storage to v3.
 *
 * `backup` is injected rather than calling localStorage directly, so the
 * unusable path is testable without a real storage and so this stays a pure
 * function of its input.
 */
export function migrate(raw: string | null, backup?: (raw: string) => void): MigrationResult {
  if (raw === null || raw === '') {
    return {
      store: EMPTY_STORE,
      writable: true,
      backedUp: false,
      detected: 'unusable',
    };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return {
      store: EMPTY_STORE,
      writable: true,
      backedUp: tryBackup(raw, backup),
      detected: 'unusable',
    };
  }

  const detected = detect(parsed);

  switch (detected) {
    case 1:
      return {
        store: v2ToV3(v1ToV2(parsed as unknown[])),
        writable: true,
        backedUp: false,
        detected,
      };
    case 2:
      return {
        store: v2ToV3(parsed as { conversations?: unknown; activeId?: unknown }),
        writable: true,
        backedUp: false,
        detected,
      };
    case 3:
      return {
        store: validateV3(parsed),
        writable: true,
        backedUp: false,
        detected,
      };
    case 'future':
      // Start an empty in-memory session and touch nothing.
      return { store: EMPTY_STORE, writable: false, backedUp: false, detected };
    case 'unusable':
      return {
        store: EMPTY_STORE,
        writable: true,
        backedUp: tryBackup(raw, backup),
        detected,
      };
  }
}
