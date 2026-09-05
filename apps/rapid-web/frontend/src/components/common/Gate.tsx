import { useState } from 'react';
import { requestJson, setToken } from '@/api/client';
import { asApiError } from '@/api/errors';
import type { AuthResponse } from '@/api/types';
import { rememberToken } from '@/auth/token';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Wordmark } from '@/components/common/Wordmark';

/**
 * The token entry screen.
 *
 * Shown only when `/api/config` says a token is required — which is never on a
 * loopback bind, where the OS already guarantees the caller is a process on
 * this Mac and a token would only mean copying 43 characters to reach your own
 * machine.
 */
export function Gate({
  initialToken,
  onAuthenticated,
}: {
  initialToken: string;
  onAuthenticated(response: AuthResponse): void;
}) {
  const [value, setValue] = useState(initialToken);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    const candidate = value.trim();
    if (candidate === '' || busy) return;

    setBusy(true);
    setError('');
    // Set it before probing, because the probe itself must carry it.
    setToken(candidate);

    try {
      const response = await requestJson<AuthResponse>('/api/auth', {
        method: 'POST',
        body: {},
      });
      // Only persisted AFTER the server accepts it: storing a rejected token
      // means the next reload retries a credential already known to be wrong.
      rememberToken(candidate);
      onAuthenticated(response);
    } catch (cause) {
      const failure = asApiError(cause);
      // Roll back, or the app is left holding a credential the server
      // refused and every later request fails with no explanation.
      setToken(null);
      setError(
        failure.status === 401
          ? 'That token was not accepted.'
          : failure.type === 'network'
            ? "Couldn't reach the server."
            : `Server error (${failure.status}).`,
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="bg-background fixed inset-0 z-30 flex items-center justify-center p-6">
      <form className="flex w-[min(320px,100%)] flex-col gap-3" onSubmit={submit}>
        {/* `justify-center`, not `text-center`: the lockup is a flex row now,
            and text alignment does not centre one. */}
        <Wordmark className="mb-1.5 justify-center text-3xl" />

        <label htmlFor="token" className="sr-only">
          Access token
        </label>
        <Input
          id="token"
          // Monospace: the user is reading a base64 secret off another screen
          // and needs to tell l from 1 and O from 0.
          className="h-11 text-center font-mono tracking-[0.02em]"
          type="password"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="Access token"
          // All off: a token is not a password a manager should learn, and
          // autocapitalising a base64 secret silently corrupts it.
          autoComplete="off"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          disabled={busy}
        />

        <Button type="submit" size="lg" disabled={busy || value.trim() === ''}>
          {busy ? 'Checking…' : 'Enter'}
        </Button>

        {/* Reserved height, so accepting a token does not make the layout
            jump as the message appears and disappears. */}
        <p className="text-destructive m-0 min-h-5 text-center text-[13px]" role="alert">
          {error}
        </p>
      </form>
    </div>
  );
}
