import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { requestJson, requestPublic, setToken } from '@/api/client';
import { asApiError } from '@/api/errors';
import { fetchModels, fetchStatus, pullModel } from '@/api/models';
import { startModel } from '@/state/startModel';
import type { AuthResponse, ConfigResponse, ModelKind } from '@/api/types';
import { Gate } from '@/components/common/Gate';
import { consumeFragmentToken, rememberToken, storedToken } from '@/auth/token';
import { branchPosition, editAndResend, retry, send, stopTurn, switchBranch } from '@/chat/turn';
import { deleteConfirmationTitle, deletionImpact, subtree } from '@/chat/MessageTree';
import { formatBytes } from '@/lib/format';
import { probeMathMLSupport } from '@/markdown/math';
import { LifecycleBand } from '@/components/models/LifecycleBand';
import { ComposerModelPicker } from '@/components/models/ComposerModelPicker';
import {
  composerPlaceholder,
  emptyStateHint,
  emptyStateSubtitle,
  headline,
  resolveReadiness,
  sendAllowed,
  sendTooltip,
  type CacheState,
  type ReadinessAction,
} from '@/readiness/ModelReadiness';
import { useActiveConversation, useActivePath, useStore } from '@/state/store';
import type { MessageNode } from '@/state/types';
import { Composer } from '@/components/chat/Composer';
import { ConfirmDialog } from '@/components/common/ConfirmDialog';
import { ApprovalDialog } from '@/components/common/ApprovalDialog';
import { ChatBar } from '@/components/chat/ChatBar';
import {
  Sidebar,
  SidebarDrawer,
  useWideLayout,
  type Surface,
} from '@/components/conversations/Sidebar';
import { SearchPalette, useSearchShortcut } from '@/components/conversations/SearchPalette';
import { MessageRow } from '@/components/chat/MessageRow';
import { LiveRegion, Transcript } from '@/components/chat/Transcript';
import { ImagesView } from '@/components/images/ImagesView';
import { AudioView } from '@/components/audio/AudioView';
import { NoticeStack } from '@/components/common/Notice';
import { noticeFor } from '@/state/notices';
import { SettingsSheet, type SettingsCategory } from '@/components/common/SettingsSheet';

type Phase = { kind: 'booting' } | { kind: 'gate'; initial: string } | { kind: 'ready' };

/** Bar titles for the non-chat surfaces; chat uses the conversation's own. */
const SURFACE_TITLES: Partial<Record<Surface, string>> = {
  images: 'Images',
  audio: 'Audio',
};

export function App() {
  const [phase, setPhase] = useState<Phase>({ kind: 'booting' });

  useEffect(() => {
    // The fragment is consumed FIRST, before anything can navigate, so the
    // token is out of the address bar as early as possible.
    const fromFragment = consumeFragmentToken();

    void (async () => {
      try {
        const config = await requestPublic<ConfigResponse>('/api/config');
        if (!config.auth_required) {
          setToken(null);
          const capabilities = await probeCapabilities();
          applyCapabilities(capabilities);
          setPhase({ kind: 'ready' });
          return;
        }
      } catch {
        // The probe failed. Fall through to the gate rather than assuming
        // auth is off — failing toward the login screen is the safe
        // direction, and a wrong guess the other way silently sends
        // unauthenticated requests.
      }
      setPhase({ kind: 'gate', initial: fromFragment ?? storedToken() ?? '' });
    })();
  }, []);

  // A token from the fragment or from a previous visit: try it rather than
  // making the user press Enter on a field that is already filled in.
  useEffect(() => {
    if (phase.kind !== 'gate' || phase.initial === '') return;
    setToken(phase.initial);
    void probeCapabilities()
      .then((capabilities) => {
        // Persisted only AFTER the server accepts it, and it MUST happen on
        // this path too: a fragment token is stripped from the URL
        // immediately, so validating without storing means the next reload
        // has nothing to present. Only e2e catches this.
        rememberToken(phase.initial);
        applyCapabilities(capabilities);
        setPhase({ kind: 'ready' });
      })
      .catch(() => {
        // Rejected or unreachable. The gate is already showing, prefilled.
        setToken(null);
      });
    // Intentionally runs once per gate entry, not per keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase.kind]);

  if (phase.kind === 'booting') return <div className="bg-background h-dvh" />;

  if (phase.kind === 'gate') {
    return (
      <Gate
        initialToken={phase.initial}
        onAuthenticated={(response) => {
          applyCapabilities(response);
          setPhase({ kind: 'ready' });
        }}
      />
    );
  }

  return <Chat />;
}

function probeCapabilities(): Promise<AuthResponse> {
  return requestJson<AuthResponse>('/api/auth', { method: 'POST', body: {} });
}

function applyCapabilities(response: AuthResponse): void {
  useStore.getState().setCapabilities(response.can_switch, response.allow_downloads);
}

function Chat() {
  const path = useActivePath();
  const conversation = useActiveConversation();
  const settings = useStore((state) => state.settings);
  const status = useStore((state) => state.status);
  const statusFailures = useStore((state) => state.statusFailures);
  const models = useStore((state) => state.models);
  const catalogLoaded = useStore((state) => state.catalogLoaded);
  const download = useStore((state) => state.download);
  const canSwitch = useStore((state) => state.canSwitch);
  const attentionToken = useStore((state) => state.attentionToken);
  const pushNotice = useStore((state) => state.pushNotice);

  // One window, opened on the category the caller means: the model button
  // and the footer button are both "settings", but they are asking for
  // different pages of it.
  const [settingsPage, setSettingsPage] = useState<SettingsCategory | null>(null);
  const [surface, setSurface] = useState<Surface>('chat');

  // The model is per surface, so choosing one for images must not retarget
  // the chat — and a failure on one must not be reported by the other.
  const kind: ModelKind = surface === 'images' ? 'image' : 'text';
  const selectedAlias = useStore((state) => state.selectedByKind[kind]);

  const wide = useWideLayout();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [railCollapsed, setRailCollapsed] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<{
    id: string;
    impact: number;
  } | null>(null);
  // Derived, not tracked: a turn that fails or is aborted cannot leave the
  // composer stuck in "stop", and there is no cascading render.
  const streaming = path.some((node) => node.status === 'streaming');
  const [revision, setRevision] = useState(0);

  // Tool results are indexed by the call they answer rather than rendered as
  // rows of their own: a result belongs under the call that asked for it.
  const toolResults = useMemo(() => {
    const byCallId = new Map<string, MessageNode>();
    for (const node of path) {
      if (node.role === 'tool' && node.toolCallId) byCallId.set(node.toolCallId, node);
    }
    return byCallId;
  }, [path]);

  useThemeAttribute(settings.theme);
  useMathProbe();
  useStatusPolling();
  useCatalogOnBoot();

  // A commit inside the streaming store does not change the app store, so the
  // transcript needs its own signal to re-run the scroll-follow effect.
  useEffect(() => {
    if (!streaming) return;
    const timer = setInterval(() => setRevision((value) => value + 1), 100);
    return () => clearInterval(timer);
  }, [streaming]);

  // Bumping a counter to re-run the scroll effect, not deriving state.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => setRevision((value) => value + 1), [path.length]);

  const cacheState: CacheState = useMemo(() => {
    if (!catalogLoaded) return 'catalogPending';
    if (selectedAlias === null) return 'catalogPending';
    const entry = models.find((model) => model.alias === selectedAlias);
    if (!entry) return 'notInCatalog';
    return entry.cached ? 'onDisk' : 'notOnDisk';
  }, [catalogLoaded, models, selectedAlias]);

  // Is the engine currently busy with a model of THIS surface's kind? The
  // engine is shared, so its state — a failure most of all — must not be
  // reported by the surface it does not concern.
  const statusIsForThisKind = useMemo(() => {
    const serving = status?.model ?? null;
    if (serving === null) return true;
    const entry = models.find((model) => model.alias === serving);
    // Unknown alias: no reason to suppress. Better a stale statement than a
    // silently blank surface.
    return entry === undefined || entry.kind === kind;
  }, [status, models, kind]);

  const readiness = useMemo(
    () =>
      resolveReadiness({
        status,
        statusFailures,
        selectedAlias,
        cacheState,
        sizeText: sizeTextFor(selectedAlias, models),
        download:
          download && download.state === 'running'
            ? {
                alias: download.alias ?? null,
                fraction:
                  download.total_bytes && download.total_bytes > 0
                    ? (download.done_bytes ?? 0) / download.total_bytes
                    : null,
                detail: download.detail ?? null,
              }
            : null,
        turnError: lastFailure(path),
        canSwitch,
        statusIsForThisKind,
      }),
    [
      status,
      statusFailures,
      selectedAlias,
      cacheState,
      models,
      download,
      path,
      canSwitch,
      statusIsForThisKind,
    ],
  );

  const canSend = sendAllowed(readiness) && !streaming;

  const onAction = useCallback(
    (action: ReadinessAction) => {
      void (async () => {
        try {
          switch (action.kind) {
            case 'download':
              useStore.getState().setDownload(await pullModel(action.alias));
              break;
            case 'start':
            case 'retry':
              // `startModel`, not `loadModel`: it adopts the server's answer,
              // without which the band keeps offering Start for the whole
              // load. See state/startModel.ts.
              await startModel(action.alias);
              break;
            case 'reconnect':
              await fetchStatus();
              break;
            case 'chooseModel':
              setSettingsPage('models');
              break;
          }
        } catch (cause) {
          pushNotice(noticeFor(asApiError(cause)));
        }
      })();
    },
    [pushNotice],
  );

  const runSend = useCallback((text: string) => {
    send(text);
  }, []);

  // Also returns to the chat surface: "New Chat" is the way back from Images,
  // and starting one while the images view stays on screen would look like it
  // had done nothing.
  const newChat = useCallback(() => {
    setSurface('chat');
    useStore.getState().createConversation();
  }, []);
  const openSearch = useCallback(() => setSearchOpen(true), []);
  useSearchShortcut(openSearch);

  // The model picker lives in the composer, not the sidebar: it is a property
  // of the message about to be sent. Selecting one goes through the same
  // `onAction` path as the readiness banner, so a switch is refused for the
  // same reasons in both places.
  const chooseModel = useCallback(
    (alias: string) => {
      const entry = useStore.getState().models.find((model) => model.alias === alias);
      // Select FIRST, and against the model's OWN kind rather than the
      // current surface: `readiness` derives from the selection, so without
      // this the band keeps reporting the previous model for the whole load.
      useStore.getState().selectAlias(entry?.kind ?? 'text', alias);
      onAction(
        entry && !entry.cached ? { kind: 'download', alias } : { kind: 'start', alias },
      );
    },
    [onAction],
  );

  const shellProps = {
    onNewChat: newChat,
    onOpenSettings: () => setSettingsPage('chat'),
    onSearch: openSearch,
    surface,
    onSelectSurface: setSurface,
  };

  return (
    <div className="relative flex h-dvh">
      {wide ? (
        <Sidebar
          {...shellProps}
          collapsed={railCollapsed}
          onToggleCollapsed={() => setRailCollapsed((value) => !value)}
        />
      ) : (
        <SidebarDrawer
          {...shellProps}
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        />
      )}

      <div className="relative flex h-dvh min-w-0 flex-1 flex-col">
        <ChatBar
          title={SURFACE_TITLES[surface] ?? conversationTitle(conversation)}
          onOpenSidebar={
            wide
              ? railCollapsed
                ? () => setRailCollapsed(false)
                : null
              : () => setDrawerOpen(true)
          }
          onNewChat={surface === 'chat' ? newChat : null}
        />

        <NoticeStack />

        {surface === 'audio' ? (
          <AudioView />
        ) : surface === 'images' ? (
          <ImagesView
            onChooseModel={() => setSettingsPage('models')}
            onSelectModel={chooseModel}
            blockedPlaceholder={
              readiness.kind === 'noModel' && readiness.canSwitch
                ? 'Choose an image model first'
                : composerPlaceholder(readiness)
            }
            // The same lifecycle surface the chat uses, so a model that is
            // downloading, starting or failed reports itself identically on
            // both. Built here rather than inside ImagesView so there is one
            // `readiness` value for the whole app.
            band={
              <LifecycleBand
                readiness={readiness}
                attentionToken={attentionToken}
                onAction={onAction}
              />
            }
          />
        ) : (
          <>
            <Transcript revision={revision} streaming={streaming}>
              {path.length === 0 ? (
                <EmptyState readiness={readiness} />
              ) : (
                path.map((node) => (
                  <MessageRow
                    key={node.id}
                    node={node}
                    mathRendering={settings.mathRendering}
                    branch={conversation ? branchPosition(node.id, conversation.nodes) : null}
                    toolResults={toolResults}
                    onBranch={(direction) => switchBranch(node.id, direction)}
                    onRetry={() => retry(node.id)}
                    onEdit={(text) => editAndResend(node.id, text)}
                    onDelete={() =>
                      setPendingDelete({
                        id: node.id,
                        impact: conversation ? deletionImpact(node.id, conversation.nodes) : 1,
                      })
                    }
                    busy={streaming}
                  />
                ))
              )}
            </Transcript>

            <LifecycleBand
              readiness={readiness}
              attentionToken={attentionToken}
              onAction={onAction}
            />

            <Composer
              placeholder={composerPlaceholder(readiness)}
              sendTooltip={sendTooltip(readiness)}
              canSend={canSend}
              streaming={streaming}
              onSend={runSend}
              onStop={stopTurn}
              onBlocked={() => useStore.getState().flagBlockedSend()}
              picker={
                <ComposerModelPicker
                  kind="text"
                  onManage={() => setSettingsPage('models')}
                  onSelect={(model) => chooseModel(model.alias)}
                />
              }
            />

            <LiveRegion message={headline(readiness)} />
          </>
        )}
      </div>

      <SearchPalette open={searchOpen} onOpenChange={setSearchOpen} onNewChat={newChat} />

      <SettingsSheet
        open={settingsPage !== null}
        onClose={() => setSettingsPage(null)}
        initialCategory={settingsPage ?? 'models'}
      />

      <ConfirmDialog
        open={pendingDelete !== null}
        title={deleteConfirmationTitle(pendingDelete?.impact ?? 1)}
        body={
          (pendingDelete?.impact ?? 1) > 1
            ? 'Alternatives below this point go with it, including any that are not on screen.'
            : undefined
        }
        confirmLabel="Delete"
        destructive
        onCancel={() => setPendingDelete(null)}
        onConfirm={() => {
          if (pendingDelete && conversation) {
            useStore.getState().removeSubtree(subtree(pendingDelete.id, conversation.nodes));
          }
          setPendingDelete(null);
        }}
      />

      <ApprovalDialog />
    </div>
  );
}

function EmptyState({ readiness }: { readiness: Parameters<typeof emptyStateSubtitle>[0] }) {
  const hint = emptyStateHint(readiness);
  return (
    // pb-[6%] centres optically: a text block on the exact midpoint reads as
    // sitting slightly low.
    <div className="flex flex-1 flex-col items-center justify-center gap-2 pb-[6%] text-center">
      <h1 className="m-0 text-3xl font-semibold tracking-tight">Ask anything</h1>
      <p className="text-muted-foreground m-0 text-sm">{emptyStateSubtitle(readiness)}</p>
      {hint ? <p className="text-muted-foreground m-0 max-w-[34ch] text-xs">{hint}</p> : null}
    </div>
  );
}

// ------------------------------------------------------------------- hooks

/** Reflect the theme choice onto the document, for tokens.css to pick up. */
function useThemeAttribute(theme: 'auto' | 'light' | 'dark') {
  useEffect(() => {
    if (theme === 'auto') {
      // REMOVED, not set to "auto": tokens.css keys the media query on
      // `:root:not([data-theme])`, so an attribute of any value would pin
      // the palette to light.
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', theme);
    }
  }, [theme]);
}

/**
 * Fall back to source rendering when the browser cannot lay out MathML.
 *
 * An old Android WebView parses MathML and then flattens a fraction into
 * "12" — not a crash, but silently wrong output that reads as the model's
 * fault. Probed once, and only ever downgrades: a user who chose `source`
 * explicitly must not have it switched back.
 */
function useMathProbe() {
  const probed = useRef(false);
  useEffect(() => {
    if (probed.current) return;
    probed.current = true;
    if (useStore.getState().settings.mathRendering !== 'mathml') return;
    if (!probeMathMLSupport()) useStore.getState().updateSettings({ mathRendering: 'source' });
  }, []);
}

/**
 * Poll `/api/status`.
 *
 * Adaptive: fast while the engine is starting, because that is when the user
 * is waiting and watching; slow once it is settled, because a phone on a
 * cellular radio pays for every request in battery.
 */
function useStatusPolling() {
  // Keyed on the engine state so a TRANSITION restarts the loop. Pacing off
  // the fetched snapshot alone is not enough: a load adopts `starting`
  // between polls (see state/startModel.ts) with a settled 15 s timer
  // already pending, leaving that start unwatched for a quarter of a minute
  // — so a failure took that long to reach the band. Transitions are rare
  // and are exactly when the user is watching, so the extra request is cheap.
  const state = useStore((store) => store.status?.state ?? null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      try {
        const snapshot = await fetchStatus();
        if (cancelled) return;
        useStore.getState().setStatus(snapshot, false);
        timer = setTimeout(tick, snapshot.state === 'starting' ? 2000 : 15000);
      } catch {
        if (cancelled) return;
        useStore.getState().setStatus(null, true);
        timer = setTimeout(tick, 5000);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [state]);
}

/**
 * Read the catalog once at boot.
 *
 * The model list used to be fetched only when the picker opened, which is
 * enough for a picker but not for anything that needs to know what KIND the
 * loaded model is — the images surface cannot tell whether it is usable until
 * the catalog has arrived. Failures are swallowed: the picker re-reads on
 * open and reports its own errors there.
 */
function useCatalogOnBoot() {
  useEffect(() => {
    void fetchModels()
      .then((response) => {
        useStore.getState().setModels(response.models);
        useStore.getState().setCapabilities(response.can_switch, response.allow_downloads);
      })
      .catch(() => undefined);
  }, []);
}

// ----------------------------------------------------------------- helpers

function sizeTextFor(
  alias: string | null,
  models: ReturnType<typeof useStore.getState>['models'],
): string | null {
  if (alias === null) return null;
  const entry = models.find((model) => model.alias === alias);
  if (!entry) return null;
  // From the catalog's byte count, never parsed out of the alias name — the
  // Mac app's parser sizes `embeddinggemma-300m-6bit` to zero for exactly
  // that reason.
  return formatBytes(entry.size_bytes);
}

function lastFailure(path: ReturnType<typeof useActivePath>) {
  for (let index = path.length - 1; index >= 0; index -= 1) {
    const node = path[index];
    if (node?.status !== 'failed' || !node.error) continue;
    return { message: node.error.message, alias: node.model ?? null };
  }
  return null;
}

/** The chat bar's title. Falls back rather than showing an empty strip. */
function conversationTitle(conversation: { title: string } | null): string {
  if (!conversation) return 'New chat';
  return conversation.title.trim() === '' ? 'New chat' : conversation.title;
}
