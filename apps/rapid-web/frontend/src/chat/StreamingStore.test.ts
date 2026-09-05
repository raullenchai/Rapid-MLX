import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { COMMIT_INTERVAL_MS, StreamingStore } from './StreamingStore';
import { tokensOf } from '@/markdown/lex';

/**
 * Fake timers throughout. A test that asserted a real wall-clock bound here
 * would be measuring machine load, not this code — and Vitest runs files
 * concurrently, so that load is not even constant.
 */
beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe('StreamingStore', () => {
  it('does no work per token — the snapshot only moves on a commit', () => {
    // The whole point: appending must be free, or a fast model turns every
    // token into a render.
    const store = new StreamingStore(() => 0);
    const before = store.getSnapshot();

    store.appendContent('a');
    store.appendContent('b');
    store.appendContent('c');

    expect(store.getSnapshot()).toBe(before);
    expect(store.current().content).toBe('abc');
  });

  it('coalesces a burst of tokens into ONE notification', () => {
    const store = new StreamingStore(() => 0);
    const listener = vi.fn();
    store.subscribe(listener);

    for (let index = 0; index < 50; index += 1) store.appendContent('x');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().content).toBe('x'.repeat(50));
  });

  it('keeps the snapshot referentially stable between commits', () => {
    // useSyncExternalStore compares by identity; a fresh object per call
    // would loop forever.
    const store = new StreamingStore(() => 0);
    store.appendContent('hello');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);

    expect(store.getSnapshot()).toBe(store.getSnapshot());
  });

  it('commits again after the next interval', () => {
    const store = new StreamingStore(() => 0);
    const listener = vi.fn();
    store.subscribe(listener);

    store.appendContent('one');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);
    store.appendContent(' two');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);

    expect(listener).toHaveBeenCalledTimes(2);
    expect(store.getSnapshot().content).toBe('one two');
  });

  it('flushes pending text immediately', () => {
    // At stream end the last tokens must not sit on a timer while the user
    // looks at an answer that appears to have stopped short.
    const store = new StreamingStore(() => 0);
    store.appendContent('final words');
    store.flush();

    expect(store.getSnapshot().content).toBe('final words');
    // And the pending timer must be cancelled, not left to fire again.
    const listener = vi.fn();
    store.subscribe(listener);
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS * 2);
    expect(listener).not.toHaveBeenCalled();
  });

  it('backs the interval off when a commit runs long', () => {
    // A fast model can burst hard enough that a commit costs more than the
    // interval, at which point commits queue behind each other and the tab
    // stops responding to touch.
    //
    // A commit reads the clock twice, before and after the work. A clock that
    // returns 0 then 100 makes every commit measure as 100 ms, which is over
    // the slow threshold.
    let reads = 0;
    const store = new StreamingStore(() => (reads++ % 2 === 0 ? 0 : 100));
    const listener = vi.fn();
    store.subscribe(listener);

    store.appendContent('a');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);
    expect(listener).toHaveBeenCalledTimes(1);

    // The next commit must now wait longer than the default interval.
    store.appendContent('b');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);
    expect(listener).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('lexes incrementally, freezing settled blocks', () => {
    const store = new StreamingStore(() => 0);

    store.appendContent('# Heading\n\nA settled paragraph.\n\n');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);
    const frozen = store.getSnapshot().lex.frozen[0];
    expect(frozen).toBeDefined();

    store.appendContent('Still being written');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);

    // Identity preserved, so React.memo skips everything above the tail.
    expect(store.getSnapshot().lex.frozen[0]).toBe(frozen);
  });

  it('renders an unterminated code fence as code, not literal backticks', () => {
    // Every streaming reply containing code hit this in the old page.
    const store = new StreamingStore(() => 0);
    store.appendContent('```python\nprint("partial"');
    store.flush();

    expect(tokensOf(store.getSnapshot().lex)[0]?.type).toBe('code');
  });

  it('accumulates reasoning separately from content', () => {
    const store = new StreamingStore(() => 0);
    store.appendReasoning('thinking...');
    store.appendContent('the answer');
    store.flush();

    const snapshot = store.getSnapshot();
    expect(snapshot.reasoning).toBe('thinking...');
    expect(snapshot.content).toBe('the answer');
  });

  it('clears everything on start', () => {
    const store = new StreamingStore(() => 0);
    store.appendContent('previous turn');
    store.appendReasoning('previous thoughts');
    store.flush();

    store.start();

    expect(store.getSnapshot().content).toBe('');
    expect(store.getSnapshot().reasoning).toBe('');
    expect(store.getSnapshot().lex.frozen).toEqual([]);
    expect(store.current().content).toBe('');
  });

  it('drops a listener that unsubscribes', () => {
    const store = new StreamingStore(() => 0);
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    unsubscribe();

    store.appendContent('x');
    vi.advanceTimersByTime(COMMIT_INTERVAL_MS);

    expect(listener).not.toHaveBeenCalled();
  });
});
