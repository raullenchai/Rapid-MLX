import { create } from 'zustand';
import { useShallow } from 'zustand/react/shallow';
import type { WireTurn } from '@/api/chat';
import type { DownloadJob, ModelEntry, ModelKind, StatusResponse } from '@/api/types';
import { activePath, choicesAlong, deepestLeaf } from '@/chat/MessageTree';
import { MAX_INSTRUCTION_LENGTH } from '@/chat/instructions';
import { newId } from '@/lib/ids';
import { HISTORY_BACKUP_KEY, HISTORY_KEY, deriveTitle, migrate } from './migrate';
import { persist } from './persist';
import {
  DEFAULT_SETTINGS,
  SCHEMA_VERSION,
  type Conversation,
  type MessageNode,
  type Settings,
} from './types';

/**
 * The application store.
 *
 * zustand rather than context + useReducer: one context holding this would
 * re-render every consumer on every change, and fixing that means sharding
 * into half a dozen contexts with hand-rolled selector equality — zustand,
 * reimplemented worse. Measured at 9 KB.
 *
 * STREAMING TEXT NEVER ENTERS THIS STORE. It lives in chat/StreamingStore and
 * is committed here once, at stream end, so the persist debounce does not
 * stringify the whole store ten times a second.
 */

// Pre-rename spelling, kept deliberately — see `HISTORY_KEY` in
// state/migrate.ts.
const SETTINGS_KEY = 'rapid-mlx-web.settings';

/** Trailing, so a burst of slider drags produces one write rather than sixty. */
const PERSIST_DEBOUNCE_MS = 400;

function safeRead(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    // Safari private browsing throws on access, not just on write.
    return null;
  }
}

function safeWrite(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Losing a preference is survivable; taking the app down for it is not.
  }
}

function loadSettings(): Settings {
  const raw = safeRead(SETTINGS_KEY);
  if (raw === null) return DEFAULT_SETTINGS;
  try {
    const parsed = JSON.parse(raw) as Partial<Settings>;
    return {
      system: typeof parsed.system === 'string' ? parsed.system : DEFAULT_SETTINGS.system,
      temperature: numberOr(parsed.temperature, DEFAULT_SETTINGS.temperature),
      topP: numberOr(parsed.topP, DEFAULT_SETTINGS.topP),
      maxTokens: numberOr(parsed.maxTokens, DEFAULT_SETTINGS.maxTokens),
      theme:
        parsed.theme === 'light' || parsed.theme === 'dark' || parsed.theme === 'auto'
          ? parsed.theme
          : DEFAULT_SETTINGS.theme,
      mathRendering: parsed.mathRendering === 'source' ? 'source' : DEFAULT_SETTINGS.mathRendering,
      enabledTools: Array.isArray(parsed.enabledTools)
        ? parsed.enabledTools.filter((name): name is string => typeof name === 'string')
        : DEFAULT_SETTINGS.enabledTools,
      // Only an explicit `true` turns it on: anything unparseable has to land
      // on the asking side, not the silent one.
      autoApproveBrowsing: parsed.autoApproveBrowsing === true,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

function numberOr(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export interface Notice {
  id: string;
  tone: 'info' | 'warning' | 'error';
  title: string;
  body?: string | undefined;
  action?: { label: string; run: () => void } | undefined;
}

type SelectionByKind = Record<ModelKind, string | null>;

/**
 * How an approval ended.
 *
 * `declined` and `unavailable` are kept apart deliberately: a decline is the
 * user's own answer and is reported as an ordinary outcome, while nobody was
 * asked in the `unavailable` case and it must not be attributed to them.
 *
 * `always` is `allowed` plus a durable grant, and only a connector tool can
 * produce it — a browse approval is scoped to one answer by design.
 */
export type ApprovalDecision = 'allowed' | 'always' | 'declined' | 'unavailable';

/**
 * What the user is being asked to approve.
 *
 * One slot for both kinds rather than two: only one prompt can be up at a
 * time (a second would replace the first on screen and leave its promise
 * unsettled), so a second slot could only ever disagree with that.
 */
export type ApprovalRequest =
  | {
      /** A page fetch. The MODEL picks the URL, so seeing the exact
       *  destination is the whole defence. */
      kind: 'browse';
      url: string;
      host: string;
    }
  | {
      /** A connector tool. The user has to know WHOSE `read_file` this is. */
      kind: 'tool';
      tool: string;
      server: string;
      short: string;
      /** The model's arguments, formatted and display-safe. */
      args: string;
    };

export type PendingApproval = ApprovalRequest & {
  answer(decision: ApprovalDecision): void;
};
/**
 * Adopt the engine's served model as the text selection.
 *
 * Provisional only while the catalog is missing: guessing `text` is right far
 * more often than not (the served model usually IS a chat model) and
 * `reconcileKinds` corrects it once the kinds arrive. Once they HAVE arrived
 * the guess must stop — status polls every few seconds, and re-adopting a
 * served image model would undo the reconciliation on the very next tick.
 */
function adoptServedModel(
  current: SelectionByKind,
  served: string | null,
  models: ModelEntry[],
): SelectionByKind {
  if (served === null || current.text !== null) return current;
  const entry = models.find((model) => model.alias === served);
  if (entry !== undefined && entry.kind !== 'text') return current;
  return { ...current, text: served };
}

/** Move a provisionally-adopted alias to the slot its real kind says it owns. */
function reconcileKinds(current: SelectionByKind, models: ModelEntry[]): SelectionByKind {
  const adopted = current.text;
  if (adopted === null) return current;

  const entry = models.find((model) => model.alias === adopted);
  if (entry === undefined || entry.kind === 'text') return current;

  return {
    ...current,
    text: null,
    // Only if that slot is free — a selection the user made themselves wins
    // over one inferred from whatever the engine happened to be serving.
    [entry.kind]: current[entry.kind] ?? adopted,
  };
}

interface StoreState {
  // ---- conversations
  conversations: Conversation[];
  activeId: string | null;
  /** False when storage holds a newer schema; every write is suppressed. */
  writable: boolean;

  // ---- session
  status: StatusResponse | null;
  statusFailures: number;
  models: ModelEntry[];
  catalogLoaded: boolean;
  /**
   * What the user last picked, PER KIND.
   *
   * Not one shared value: the chat and images surfaces each have a model, and
   * a single field means choosing an image model silently retargets the chat
   * — which then reports the image model's start failure as its own.
   */
  selectedByKind: Record<ModelKind, string | null>;
  canSwitch: boolean;
  allowDownloads: boolean;
  download: DownloadJob | null;
  notices: Notice[];
  /** Bumped when a gated send is attempted, to flash the readiness surface. */
  attentionToken: number;
  /**
   * What a tool is waiting on the user to approve, if any.
   *
   * A `browse` names a host the MODEL chose, and a connector call names a
   * program on this Mac. In both cases the user seeing what will actually run
   * is the only thing between the model and an action it picked for itself.
   */
  pendingApproval: PendingApproval | null;

  settings: Settings;

  // ---- actions
  createConversation(): string;
  setActiveConversation(id: string): void;
  deleteConversation(id: string): void;
  /** Pin, archive, rename. Deliberately does NOT touch `updatedAt` — see
   *  the action for why. */
  updateConversation(id: string, patch: Partial<Conversation>): void;
  /** Set the open conversation's own system prompt, creating one if the user
   *  has not started a conversation yet. */
  setConversationInstructions(text: string): void;
  appendNode(
    node: Omit<MessageNode, 'id' | 'createdAt'> & Partial<Pick<MessageNode, 'id' | 'createdAt'>>,
  ): string;
  patchNode(id: string, patch: Partial<MessageNode>): void;
  removeSubtree(ids: Set<string>): void;
  setActiveLeaf(leafId: string): void;

  setStatus(status: StatusResponse | null, failed: boolean): void;
  setModels(models: ModelEntry[]): void;
  setCapabilities(canSwitch: boolean, allowDownloads: boolean): void;
  selectAlias(kind: ModelKind, alias: string | null): void;
  setDownload(job: DownloadJob | null): void;

  pushNotice(notice: Omit<Notice, 'id'>): void;
  dismissNotice(id: string): void;
  flagBlockedSend(): void;
  askApproval(request: ApprovalRequest): Promise<ApprovalDecision>;
  answerApproval(decision: ApprovalDecision): void;

  updateSettings(patch: Partial<Settings>): void;
}

const boot = migrate(safeRead(HISTORY_KEY), (raw) => safeWrite(HISTORY_BACKUP_KEY, raw));

export const useStore = create<StoreState>()((set, get) => {
  let persistTimer: ReturnType<typeof setTimeout> | undefined;

  /** Debounced write. `flushPersist` is exported below for the unload path. */
  const schedulePersist = () => {
    if (!get().writable) return;
    if (persistTimer !== undefined) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
      persistTimer = undefined;
      flushPersist();
    }, PERSIST_DEBOUNCE_MS);
  };

  const flushPersist = () => {
    const state = get();
    if (!state.writable) return;
    if (persistTimer !== undefined) {
      clearTimeout(persistTimer);
      persistTimer = undefined;
    }
    persist(
      {
        v: SCHEMA_VERSION,
        conversations: state.conversations,
        folders: [],
        activeId: state.activeId,
      },
      localStorage,
    );
  };

  /** Apply `mutate` to the active conversation and schedule a write. */
  const updateActive = (mutate: (conversation: Conversation) => Conversation) => {
    set((state) => {
      if (state.activeId === null) return state;
      return {
        conversations: state.conversations.map((conversation) =>
          conversation.id === state.activeId ? mutate(conversation) : conversation,
        ),
      };
    });
    schedulePersist();
  };

  /**
   * Bump `updatedAt` and back-fill the title.
   *
   * Deliberately NOT called by branch switching: navigating between existing
   * answers is not work, and reshuffling the sidebar for it would move a
   * conversation to the top for merely being looked at.
   */
  const touch = (conversation: Conversation): Conversation => ({
    ...conversation,
    updatedAt: Date.now(),
    title: conversation.hasCustomTitle
      ? conversation.title
      : conversation.title ||
        deriveTitle(
          activePath(conversation.nodes, conversation.activeLeafId, conversation.branchChoices),
        ),
  });

  return {
    conversations: boot.store.conversations,
    activeId: boot.store.activeId,
    writable: boot.writable,

    status: null,
    statusFailures: 0,
    models: [],
    catalogLoaded: false,
    selectedByKind: { text: null, image: null, audio: null },
    canSwitch: false,
    allowDownloads: false,
    download: null,
    notices: [],
    attentionToken: 0,
    pendingApproval: null,

    settings: loadSettings(),

    createConversation() {
      // Reuse an existing empty conversation rather than stacking another one
      // beside it. Pressing New Chat twice, or pressing it when the app has
      // just opened on a blank one, otherwise leaves a column of identical
      // "New chat" rows that the user then has to delete one at a time.
      // Reuse is only safe when it is UNTITLED as well as empty: a renamed
      // blank conversation is one the user deliberately made, and so is one
      // carrying a system prompt they typed before their first message.
      const existing = get().conversations.find(
        (conversation) =>
          conversation.nodes.length === 0 &&
          !conversation.hasCustomTitle &&
          conversation.title.trim() === '' &&
          (conversation.customInstructions ?? '').trim() === '' &&
          !conversation.isArchived,
      );
      if (existing) {
        set({ activeId: existing.id });
        schedulePersist();
        return existing.id;
      }

      const id = newId();
      const now = Date.now();
      set((state) => ({
        conversations: [
          {
            id,
            title: '',
            hasCustomTitle: false,
            createdAt: now,
            updatedAt: now,
            nodes: [],
            activeLeafId: null,
            branchChoices: {},
            isPinned: false,
            isArchived: false,
            folderId: null,
          },
          ...state.conversations,
        ],
        activeId: id,
      }));
      schedulePersist();
      return id;
    },

    setActiveConversation(id) {
      set({ activeId: id });
      schedulePersist();
    },

    deleteConversation(id) {
      set((state) => {
        const conversations = state.conversations.filter((conversation) => conversation.id !== id);
        return {
          conversations,
          activeId: state.activeId === id ? (conversations[0]?.id ?? null) : state.activeId,
        };
      });
      schedulePersist();
    },

    updateConversation(id, patch) {
      set((state) => ({
        conversations: state.conversations.map((conversation) =>
          conversation.id === id ? { ...conversation, ...patch } : conversation,
        ),
      }));
      // `updatedAt` is deliberately NOT bumped. Pinning, archiving or
      // renaming is organising the list, not working in the conversation, and
      // moving a row to the top of "Today" for having been renamed is exactly
      // the opposite of what the user was trying to do.
      schedulePersist();
    },

    setConversationInstructions(text) {
      // A prompt typed before the first message has to have somewhere to
      // live, or it is silently lost the moment the user presses send.
      if (get().activeId === null) get().createConversation();
      updateActive((conversation) => ({
        ...conversation,
        customInstructions: text.slice(0, MAX_INSTRUCTION_LENGTH),
      }));
    },

    appendNode(partial) {
      const id = partial.id ?? newId();
      const node: MessageNode = {
        ...partial,
        id,
        createdAt: partial.createdAt ?? Date.now(),
      };

      updateActive((conversation) => {
        const nodes = [...conversation.nodes, node];
        const path = activePath(nodes, id, conversation.branchChoices);
        return touch({
          ...conversation,
          nodes,
          activeLeafId: id,
          // Merged, not replaced: a fork the user has not visited this session
          // must keep the position it had.
          branchChoices: {
            ...conversation.branchChoices,
            ...choicesAlong(path),
          },
        });
      });

      return id;
    },

    patchNode(id, patch) {
      updateActive((conversation) => ({
        ...conversation,
        nodes: conversation.nodes.map((node) => (node.id === id ? { ...node, ...patch } : node)),
        updatedAt: Date.now(),
      }));
    },

    removeSubtree(ids) {
      updateActive((conversation) => {
        const nodes = conversation.nodes.filter((node) => !ids.has(node.id));

        // Prune the doomed edges BEFORE resolving a new leaf, or a stale edge
        // steers the walk at the very fork the user is standing on.
        const choices: Record<string, string> = {};
        for (const [parent, child] of Object.entries(conversation.branchChoices)) {
          if (!ids.has(parent) && !ids.has(child)) choices[parent] = child;
        }

        const leafGone = conversation.activeLeafId === null || ids.has(conversation.activeLeafId);
        return {
          ...conversation,
          nodes,
          branchChoices: choices,
          activeLeafId: leafGone ? null : conversation.activeLeafId,
        };
      });
    },

    setActiveLeaf(leafId) {
      updateActive((conversation) => {
        // Resolve DOWNWARDS so returning to a branch lands where the user left
        // it rather than at its deepest tip.
        const resolved = deepestLeaf(leafId, conversation.nodes, conversation.branchChoices);
        const path = activePath(conversation.nodes, resolved, conversation.branchChoices);
        return {
          ...conversation,
          activeLeafId: resolved,
          branchChoices: {
            ...conversation.branchChoices,
            ...choicesAlong(path),
          },
          // `updatedAt` deliberately untouched — see `touch`.
        };
      });
    },

    setStatus(status, failed) {
      set((state) => ({
        status: status ?? state.status,
        statusFailures: failed ? state.statusFailures + 1 : 0,
        canSwitch: status?.can_switch ?? state.canSwitch,
        // Adopt the serving model as the TEXT selection until the user picks
        // one, so a page opened against a running engine is immediately
        // usable — and so `--attach` mode, where there is no catalog at all,
        // still names the model it is attached to.
        //
        // The kind cannot be checked here: the catalog may not have loaded
        // yet. `setModels` corrects a wrong guess when it does.
        selectedByKind: adoptServedModel(
          state.selectedByKind,
          status?.model ?? null,
          state.models,
        ),
      }));
    },

    setModels(models) {
      set((state) => ({
        models,
        catalogLoaded: true,
        // Now that the kinds are known, undo an adoption `setStatus` could
        // not have got right: a served IMAGE model must not be sitting in
        // the chat's slot, or the chat reports its failures as its own.
        selectedByKind: reconcileKinds(state.selectedByKind, models),
      }));
    },

    setCapabilities(canSwitch, allowDownloads) {
      set({ canSwitch, allowDownloads });
    },

    selectAlias(kind, alias) {
      set((state) => ({
        selectedByKind: { ...state.selectedByKind, [kind]: alias },
        // Mark the status as no longer describing the selection. Without this
        // the cached snapshot still names the PREVIOUS model, so
        // `resolveReadiness` finds no serving state and tells the user to
        // press Start on something that is already starting.
        //
        // `null`, not a synthesised "starting": we do not know yet, and null
        // resolves to `needsStart` at worst, which is honest.
        status: alias !== null && alias !== state.status?.model ? null : state.status,
      }));
    },

    setDownload(job) {
      set({ download: job });
    },

    pushNotice(notice) {
      set((state) => ({
        notices: [...state.notices, { ...notice, id: newId() }],
      }));
    },

    dismissNotice(id) {
      set((state) => ({
        notices: state.notices.filter((notice) => notice.id !== id),
      }));
    },

    flagBlockedSend() {
      set((state) => ({ attentionToken: state.attentionToken + 1 }));
    },

    askApproval(request) {
      // One prompt at a time. A second would replace the first on screen and
      // leave its promise unsettled, hanging the turn that is waiting on it.
      if (get().pendingApproval !== null) return Promise.resolve('unavailable');
      return new Promise<ApprovalDecision>((resolve) => {
        set({
          pendingApproval: {
            ...request,
            answer(decision) {
              resolve(decision);
            },
          },
        });
      });
    },

    answerApproval(decision) {
      const pending = get().pendingApproval;
      if (pending === null) return;
      set({ pendingApproval: null });
      pending.answer(decision);
    },

    updateSettings(patch) {
      set((state) => {
        const settings = { ...state.settings, ...patch };
        safeWrite(SETTINGS_KEY, JSON.stringify(settings));
        return { settings };
      });
    },
  };
});

/**
 * Force a write immediately, bypassing the debounce.
 *
 * Reads the store rather than reaching into the factory's closure. An earlier
 * version assigned a mutable module-level binding from the factory, which is a
 * TDZ error — `create()` runs during module evaluation, before the `let` below
 * it exists — so the bundle threw on load and the page rendered nothing. Only
 * e2e catches this: unit tests import the store lazily, never in load order.
 *
 * iOS Safari kills a backgrounded tab WITHOUT firing `beforeunload`, so
 * `visibilitychange` is the one that fires on a phone and `pagehide` covers
 * bfcache. All three are registered because which fires depends on the
 * teardown path.
 */
export function flushPersistNow(): void {
  const state = useStore.getState();
  if (!state.writable) return;
  persist(
    {
      v: SCHEMA_VERSION,
      conversations: state.conversations,
      folders: [],
      activeId: state.activeId,
    },
    localStorage,
  );
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', flushPersistNow);
  window.addEventListener('pagehide', flushPersistNow);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flushPersistNow();
  });
}

// ------------------------------------------------------------- selectors

export function useActiveConversation(): Conversation | null {
  return useStore(
    (state) =>
      state.conversations.find((conversation) => conversation.id === state.activeId) ?? null,
  );
}

/**
 * The visible transcript.
 *
 * `useShallow` so a store change that does not alter the path — a status poll,
 * a notice, a settings tweak — does not re-render the whole transcript.
 */
export function useActivePath(): MessageNode[] {
  return useStore(
    useShallow((state) => {
      const conversation = state.conversations.find((c) => c.id === state.activeId);
      if (!conversation) return [];
      return activePath(conversation.nodes, conversation.activeLeafId, conversation.branchChoices);
    }),
  );
}

/** The turns to send, stripped to what the wire accepts. */
export function wireTurns(path: MessageNode[], system: string): WireTurn[] {
  const turns = path
    .filter((node) => {
      // A tool row is kept whatever its status. It answers a call the
      // assistant turn above it made, and the model needs one result per id
      // — a failed tool ran and its error IS the result.
      if (node.role === 'tool') return true;
      // A failed turn is not history the model should continue from. An empty
      // one is dropped too, unless it carried tool calls: that is a real turn
      // with no prose, and dropping it orphans the results under it.
      return node.status !== 'failed' && (node.content !== '' || (node.toolCalls?.length ?? 0) > 0);
    })
    .map((node) => ({
      role: node.role,
      content: node.content,
      ...(node.toolCalls?.length ? { tool_calls: node.toolCalls } : {}),
      ...(node.toolCallId ? { tool_call_id: node.toolCallId } : {}),
    }));

  return system.trim() === '' ? turns : [{ role: 'system', content: system }, ...turns];
}
