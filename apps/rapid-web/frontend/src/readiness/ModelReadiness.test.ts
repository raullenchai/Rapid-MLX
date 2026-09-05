import { describe, expect, it } from 'vitest';
import {
  actionIsRenderable,
  accessibilityLabel,
  composerPlaceholder,
  detail,
  displayable,
  emptyStateHint,
  emptyStateSubtitle,
  failureApplies,
  headline,
  isWorking,
  readinessAction,
  readyDescribesSelection,
  resolveReadiness,
  sendAllowed,
  sendTooltip,
  serveStateSpeaksForSelection,
  statusRole,
  type ModelReadiness,
  type ResolveInput,
} from './ModelReadiness';

/** A resolve input with nothing happening. Each test overrides only the
 *  fields its case is about, so the case is legible at the call site. */
function input(overrides: Partial<ResolveInput> = {}): ResolveInput {
  return {
    status: null,
    statusFailures: 0,
    selectedAlias: null,
    cacheState: 'catalogPending',
    sizeText: null,
    download: null,
    turnError: null,
    canSwitch: true,
    ...overrides,
  };
}

const status = (state: string, model: string | null = null, detailText: string | null = null) => ({
  state,
  model,
  detail: detailText,
});

describe('displayable', () => {
  it('accepts a real alias', () => {
    expect(displayable('qwen3-0.6b-8bit')).toBe('qwen3-0.6b-8bit');
  });

  it.each(['', '   ', 'Loading', 'starting', 'WARMING UP', 'unknown', 'none'])(
    'rejects the placeholder %o',
    (placeholder) => {
      // Without this filter the page renders "Couldn't start ." and
      // "Starting Loading" — a real defect the Mac app hit first.
      expect(displayable(placeholder)).toBeNull();
    },
  );

  it('rejects null and undefined', () => {
    expect(displayable(null)).toBeNull();
    expect(displayable(undefined)).toBeNull();
  });
});

describe('serveStateSpeaksForSelection (permissive, non-send-enabling)', () => {
  it('matches two real names', () => {
    expect(serveStateSpeaksForSelection('a', 'a')).toBe(true);
    expect(serveStateSpeaksForSelection('a', 'b')).toBe(false);
  });

  it('lets a real serving model speak for an unsynced selection', () => {
    // The launch frame where the picker lags the auto-started model.
    expect(serveStateSpeaksForSelection('a', null)).toBe(true);
  });

  it('lets a placeholder start speak when nothing real is selected', () => {
    expect(serveStateSpeaksForSelection('Loading', '')).toBe(true);
  });

  it('does NOT let a placeholder start speak for a real selection', () => {
    // Claiming "Starting B" for a start we cannot prove is B's would
    // suppress B's own Start button.
    expect(serveStateSpeaksForSelection('Loading', 'b')).toBe(false);
  });
});

describe('readyDescribesSelection (strict, send-enabling)', () => {
  it('requires two real names that match', () => {
    expect(readyDescribesSelection('a', 'a')).toBe(true);
    expect(readyDescribesSelection('a', 'b')).toBe(false);
  });

  it('refuses an unsynced selection, unlike the permissive rule', () => {
    // This is the whole difference between the two rules: `ready` is the one
    // state that enables Send, and it must never light up against an alias
    // the send path is not holding.
    expect(readyDescribesSelection('a', null)).toBe(false);
    expect(readyDescribesSelection(null, 'a')).toBe(false);
  });

  it('accepts a co-resident model that is not the primary', () => {
    // A hot load leaves the engine holding several models at once. Comparing
    // against the primary alone would tell the images surface to start a
    // model the engine is already running.
    expect(readyDescribesSelection('chat', 'image', ['chat', 'image'])).toBe(true);
  });

  it('still refuses a selection the engine is not holding', () => {
    expect(readyDescribesSelection('chat', 'other', ['chat', 'image'])).toBe(false);
  });
});

describe('failureApplies', () => {
  it('shows a failure when nothing is selected', () => {
    expect(failureApplies('a', null)).toBe(true);
    expect(failureApplies(null, null)).toBe(true);
  });

  it('shows an attributed failure only for its own model', () => {
    expect(failureApplies('a', 'a')).toBe(true);
    expect(failureApplies('a', 'b')).toBe(false);
  });

  it('SUPPRESSES an unattributed failure when a model is selected', () => {
    // Blaming the user's fresh pick for an error we cannot attribute is the
    // worse of the two mistakes.
    expect(failureApplies(null, 'a')).toBe(false);
  });
});

describe('resolveReadiness — precedence', () => {
  it('needs two consecutive failures before declaring the server unreachable', () => {
    // One miss is a phone radio dropping a packet. The old page turned the
    // chip red for exactly that.
    expect(resolveReadiness(input({ statusFailures: 1 })).kind).not.toBe('serverUnreachable');
    expect(resolveReadiness(input({ statusFailures: 2 })).kind).toBe('serverUnreachable');
  });

  it('reports unreachable even when a stale ready status is cached', () => {
    // A stale "ready" would enable Send against a server that is gone.
    const r = resolveReadiness(
      input({
        statusFailures: 3,
        status: status('ready', 'a'),
        selectedAlias: 'a',
      }),
    );
    expect(r.kind).toBe('serverUnreachable');
    expect(sendAllowed(r)).toBe(false);
  });

  it('lets an in-flight download beat a stale failure', () => {
    // The user pressed Download and it IS working; showing the old error
    // would be a lie about the present.
    const r = resolveReadiness(
      input({
        selectedAlias: 'a',
        download: { alias: 'a', fraction: 0.4, detail: '2.1 GB of 5.3 GB' },
        turnError: { message: 'earlier failure', alias: 'a' },
      }),
    );
    expect(r).toEqual({
      kind: 'downloading',
      alias: 'a',
      detail: '2.1 GB of 5.3 GB',
      fraction: 0.4,
    });
  });

  it('lets an in-flight start beat a stale failure', () => {
    const r = resolveReadiness(
      input({
        status: status('starting', 'a', 'loading weights'),
        selectedAlias: 'a',
        turnError: { message: 'earlier failure', alias: 'a' },
      }),
    );
    expect(r).toEqual({
      kind: 'starting',
      alias: 'a',
      detail: 'loading weights',
    });
  });

  it('does not claim a placeholder start belongs to a real selection', () => {
    const r = resolveReadiness(
      input({
        status: status('starting', 'Loading'),
        selectedAlias: 'b',
        cacheState: 'onDisk',
      }),
    );
    // Resolves B's own state instead, so B's Start stays available.
    expect(r).toEqual({ kind: 'needsStart', alias: 'b' });
  });

  it('is ready only when the serving model is the selected model', () => {
    expect(resolveReadiness(input({ status: status('ready', 'a'), selectedAlias: 'a' }))).toEqual({
      kind: 'ready',
      alias: 'a',
    });

    const mismatched = resolveReadiness(
      input({
        status: status('ready', 'a'),
        selectedAlias: 'b',
        cacheState: 'onDisk',
      }),
    );
    expect(mismatched.kind).toBe('needsStart');
    expect(sendAllowed(mismatched)).toBe(false);
  });

  it('lets an engine failure outrank a turn error', () => {
    // If the child is down, that is WHY the turn failed, and it is the more
    // actionable statement.
    const r = resolveReadiness(
      input({
        status: status('failed', 'a', 'out of memory'),
        selectedAlias: 'a',
        turnError: { message: 'stream ended', alias: 'a' },
      }),
    );
    expect(r).toEqual({
      kind: 'failed',
      alias: 'a',
      message: 'out of memory',
      action: { kind: 'retry', alias: 'a' },
    });
  });

  it('does not blame this selection for another model’s engine failure', () => {
    // ONE engine serves chat and images alike, so a failed image model would
    // otherwise be reported by the chat as its own — with a Retry that
    // switches the engine to a model the chat cannot use.
    const r = resolveReadiness(
      input({
        status: status('failed', 'flux2-klein-4b', 'needs the image extra'),
        selectedAlias: 'qwen3-4b',
        cacheState: 'onDisk',
      }),
    );
    expect(r).toEqual({ kind: 'needsStart', alias: 'qwen3-4b' });
  });

  it('still reports an engine failure naming nothing when nothing is selected', () => {
    const r = resolveReadiness(
      input({ status: status('failed', null, 'the engine stopped'), selectedAlias: null }),
    );
    expect(r.kind).toBe('failed');
  });

  it('shows a turn error attributed to the selected model', () => {
    const r = resolveReadiness(
      input({
        status: status('stopped'),
        selectedAlias: 'a',
        cacheState: 'onDisk',
        turnError: { message: 'stream ended', alias: 'a' },
      }),
    );
    expect(r.kind).toBe('failed');
  });

  it('suppresses an unattributed turn error when a model is selected', () => {
    const r = resolveReadiness(
      input({
        status: status('stopped'),
        selectedAlias: 'a',
        cacheState: 'onDisk',
        turnError: { message: 'something went wrong', alias: null },
      }),
    );
    expect(r).toEqual({ kind: 'needsStart', alias: 'a' });
  });

  it('falls through to noModel when nothing is selected', () => {
    expect(resolveReadiness(input({ status: status('stopped') }))).toEqual({
      kind: 'noModel',
      canSwitch: true,
    });
  });
});

describe('resolveReadiness — two models resident at once', () => {
  /**
   * A hot `POST /v1/models/load` leaves a chat model and an image model in
   * one engine. `status.model` names only the PRIMARY (the assistant-group
   * model), so every rule that reasoned from it alone had to learn about the
   * `resident` set.
   */
  const shared = {
    state: 'ready',
    model: 'chat-model',
    detail: null,
    resident: ['chat-model', 'image-model'],
  };

  it('reports the non-primary image model as ready on the images surface', () => {
    const r = resolveReadiness(
      input({
        status: shared,
        selectedAlias: 'image-model',
        cacheState: 'onDisk',
        // The primary is a TEXT model, so the images surface is told the
        // engine's state is not about it.
        statusIsForThisKind: false,
      }),
    );
    expect(r).toEqual({ kind: 'ready', alias: 'image-model' });
  });

  it('reports the primary as ready on the chat surface', () => {
    expect(
      resolveReadiness(input({ status: shared, selectedAlias: 'chat-model', cacheState: 'onDisk' })),
    ).toEqual({ kind: 'ready', alias: 'chat-model' });
  });

  it('still asks to start a model the engine is NOT holding', () => {
    // The whole point of keying on `resident` rather than just relaxing the
    // rule: an unloaded model must still offer its Start button.
    expect(
      resolveReadiness(
        input({
          status: shared,
          selectedAlias: 'third-model',
          cacheState: 'onDisk',
          statusIsForThisKind: false,
        }),
      ),
    ).toEqual({ kind: 'needsStart', alias: 'third-model' });
  });

  it('does not blame a resident model for another model\u2019s failure', () => {
    // The engine failed on something else while holding my pick. Reporting
    // that as mine would offer a Retry that restarts the engine underneath a
    // model that is working.
    const r = resolveReadiness(
      input({
        status: {
          state: 'failed',
          model: 'other-model',
          detail: 'boom',
          resident: ['image-model'],
        },
        selectedAlias: 'image-model',
        cacheState: 'onDisk',
      }),
    );
    expect(r.kind).not.toBe('failed');
  });
});

describe('resolveReadiness — the four cache states', () => {
  const base = { status: status('stopped'), selectedAlias: 'a' } as const;

  it('notOnDisk needs a download and carries the size', () => {
    expect(
      resolveReadiness(input({ ...base, cacheState: 'notOnDisk', sizeText: '5.3 GB' })),
    ).toEqual({ kind: 'needsDownload', alias: 'a', sizeText: '5.3 GB' });
  });

  it('onDisk needs a start', () => {
    expect(resolveReadiness(input({ ...base, cacheState: 'onDisk' }))).toEqual({
      kind: 'needsStart',
      alias: 'a',
    });
  });

  it('catalogPending needs a start — it must not claim a download is needed', () => {
    // Before the catalog lands we know nothing. Claiming "isn't downloaded"
    // would be a guess, and Start is the action either way.
    expect(resolveReadiness(input({ ...base, cacheState: 'catalogPending' }))).toEqual({
      kind: 'needsStart',
      alias: 'a',
    });
  });

  it('notInCatalog is its own state, distinct from needsStart', () => {
    // This is why cacheState is four-valued. Collapsing "we do not know"
    // into a boolean told an unknown alias it was already downloaded.
    expect(resolveReadiness(input({ ...base, cacheState: 'notInCatalog' }))).toEqual({
      kind: 'unknownModel',
      alias: 'a',
    });
  });

  it('gives needsStart and unknownModel the same headline but different detail', () => {
    const needsStart: ModelReadiness = { kind: 'needsStart', alias: 'a' };
    const unknown: ModelReadiness = { kind: 'unknownModel', alias: 'a' };
    expect(headline(needsStart)).toBe(headline(unknown));
    expect(detail(needsStart)).not.toBe(detail(unknown));
    expect(detail(unknown)).not.toContain('already downloaded');
  });
});

describe('derived predicates', () => {
  it('allows sending in exactly one state', () => {
    const states: ModelReadiness[] = [
      { kind: 'serverUnreachable', consecutiveFailures: 2 },
      { kind: 'noModel', canSwitch: true },
      { kind: 'needsDownload', alias: 'a', sizeText: null },
      { kind: 'needsStart', alias: 'a' },
      { kind: 'unknownModel', alias: 'a' },
      { kind: 'downloading', alias: 'a', detail: null, fraction: null },
      { kind: 'starting', alias: 'a', detail: null },
      { kind: 'failed', alias: 'a', message: 'x', action: null },
      { kind: 'ready', alias: 'a' },
    ];
    expect(states.filter(sendAllowed)).toEqual([{ kind: 'ready', alias: 'a' }]);
  });

  it('marks only download and start as working', () => {
    expect(
      isWorking({
        kind: 'downloading',
        alias: 'a',
        detail: null,
        fraction: null,
      }),
    ).toBe(true);
    expect(isWorking({ kind: 'starting', alias: 'a', detail: null })).toBe(true);
    expect(isWorking({ kind: 'ready', alias: 'a' })).toBe(false);
  });

  it('maps states onto status roles', () => {
    expect(statusRole({ kind: 'ready', alias: 'a' })).toBe('ready');
    expect(statusRole({ kind: 'starting', alias: 'a', detail: null })).toBe('working');
    expect(statusRole({ kind: 'failed', alias: null, message: 'x', action: null })).toBe('error');
    expect(statusRole({ kind: 'serverUnreachable', consecutiveFailures: 2 })).toBe('error');
    expect(statusRole({ kind: 'noModel', canSwitch: true })).toBe('idle');
  });

  it('does not render chooseModel as a button', () => {
    // The model chip is a few pixels away and already says the same thing.
    expect(actionIsRenderable(readinessAction({ kind: 'noModel', canSwitch: true }))).toBe(false);
    expect(actionIsRenderable(readinessAction({ kind: 'needsStart', alias: 'a' }))).toBe(true);
  });

  it('offers no action while work is in flight', () => {
    expect(readinessAction({ kind: 'starting', alias: 'a', detail: null })).toBeNull();
    expect(
      readinessAction({
        kind: 'downloading',
        alias: 'a',
        detail: null,
        fraction: null,
      }),
    ).toBeNull();
    expect(readinessAction({ kind: 'ready', alias: 'a' })).toBeNull();
  });
});

describe('--attach mode', () => {
  it('does not tell the user to use a picker that cannot switch', () => {
    // The dead end this fixes: the empty state said "choose a model in the
    // header" while the header control was disabled, with the only
    // explanation in a `title` that never fires on a touch screen.
    const attached: ModelReadiness = { kind: 'noModel', canSwitch: false };
    expect(detail(attached)).not.toContain('Choose a model');
    expect(detail(attached)).toContain('attached engine');
  });

  it('still points at the picker when switching IS possible', () => {
    const owned: ModelReadiness = { kind: 'noModel', canSwitch: true };
    expect(detail(owned)).toContain('Choose a model');
  });

  it('offers no action when there is nothing the user can do', () => {
    // An action the user cannot take is worse than no action.
    expect(readinessAction({ kind: 'noModel', canSwitch: false })).toBeNull();
    expect(readinessAction({ kind: 'noModel', canSwitch: true })).not.toBeNull();
  });

  it('carries canSwitch through resolve', () => {
    expect(resolveReadiness(input({ canSwitch: false }))).toEqual({
      kind: 'noModel',
      canSwitch: false,
    });
  });
});

describe('copy', () => {
  const every: ModelReadiness[] = [
    { kind: 'serverUnreachable', consecutiveFailures: 2 },
    { kind: 'noModel', canSwitch: true },
    { kind: 'noModel', canSwitch: false },
    { kind: 'needsDownload', alias: 'qwen3-4b', sizeText: '2.4 GB' },
    { kind: 'needsDownload', alias: 'qwen3-4b', sizeText: null },
    { kind: 'needsStart', alias: 'qwen3-4b' },
    { kind: 'unknownModel', alias: 'org/custom' },
    {
      kind: 'downloading',
      alias: 'qwen3-4b',
      detail: '1.1 GB of 2.4 GB',
      fraction: 0.46,
    },
    { kind: 'downloading', alias: 'qwen3-4b', detail: null, fraction: null },
    { kind: 'starting', alias: 'qwen3-4b', detail: null },
    { kind: 'ready', alias: 'qwen3-4b' },
    {
      kind: 'failed',
      alias: 'qwen3-4b',
      message: 'out of memory',
      action: null,
    },
    { kind: 'failed', alias: null, message: 'unknown', action: null },
  ];

  it('never leaves a required string empty', () => {
    for (const r of every) {
      expect(headline(r), `headline for ${r.kind}`).not.toBe('');
      expect(composerPlaceholder(r), `placeholder for ${r.kind}`).not.toBe('');
      expect(sendTooltip(r), `tooltip for ${r.kind}`).not.toBe('');
      expect(emptyStateSubtitle(r), `subtitle for ${r.kind}`).not.toBe('');
    }
  });

  it('never interpolates an empty alias into copy', () => {
    // "Couldn't start ." with a dangling space is the defect this guards.
    for (const r of every) {
      expect(headline(r)).not.toMatch(/\s\.$|\s{2}|\s$/);
      expect(composerPlaceholder(r)).not.toMatch(/\s{2}|\s$/);
    }
  });

  it('says nothing extra when ready', () => {
    // An empty state must not stack three sentences that mean the same thing.
    expect(detail({ kind: 'ready', alias: 'a' })).toBeNull();
    expect(emptyStateHint({ kind: 'ready', alias: 'a' })).toBeNull();
  });

  it('composes the accessibility label from headline and detail', () => {
    expect(accessibilityLabel({ kind: 'needsStart', alias: 'a' })).toBe(
      "a isn't running, It's already downloaded — starting takes a few seconds.",
    );
    // Ready has no detail, so the label is just the headline.
    expect(accessibilityLabel({ kind: 'ready', alias: 'a' })).toBe('Ready — a');
  });

  it('pins the full copy table', () => {
    expect(
      every.map((r) => ({
        kind: r.kind,
        headline: headline(r),
        detail: detail(r),
        placeholder: composerPlaceholder(r),
        tooltip: sendTooltip(r),
        subtitle: emptyStateSubtitle(r),
        hint: emptyStateHint(r),
      })),
    ).toMatchSnapshot();
  });
});
