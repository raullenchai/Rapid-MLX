import { setToken } from '@/api/client';

/**
 * Token acquisition.
 *
 * The token arrives in the URL FRAGMENT, never the query string: a fragment is
 * not sent to the server, so it cannot reach an access log or a tunnel
 * provider's history. The page consumes it and strips it from the address bar
 * immediately, so it does not linger in history or a screenshot.
 */

/**
 * Keeps the pre-rename spelling on purpose — see `HISTORY_KEY` in
 * state/migrate.ts. Renaming this costs less (a phone is just asked for its
 * token again) but is the same mistake.
 */
export const TOKEN_KEY = 'rapid-mlx-web.token';

/** Read and strip a token from the URL fragment. */
export function consumeFragmentToken(): string | null {
  const match = /(?:^#|&)token=([^&]+)/.exec(window.location.hash);
  if (!match?.[1]) return null;

  let token: string;
  try {
    token = decodeURIComponent(match[1]);
  } catch {
    // A malformed escape sequence. Not a usable token.
    return null;
  }

  try {
    // Drops the whole fragment, not just the token, so nothing survives in
    // history. `replaceState` rather than assigning `hash`, which would push
    // a new entry and leave the token one Back away.
    window.history.replaceState(null, '', window.location.pathname + window.location.search);
  } catch {
    window.location.hash = '';
  }

  return token;
}

export function storedToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function rememberToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    // Private browsing. The session still works; it just will not survive a
    // reload, which is better than refusing to start.
  }
  setToken(token);
}

export function forgetToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Nothing to do; the in-memory token is cleared regardless.
  }
  setToken(null);
}
