import Foundation
import Security

/// Issue #17 desktop-half: per-launch bearer secret.
///
/// Today the embedded ``rapid-mlx serve`` accepts any local request
/// against ``127.0.0.1:<port>/v1/chat/completions``. That binding is
/// loopback-only so a remote attacker can't reach it, but ANY local
/// process — a sandbox-escaped browser tab, an unrelated terminal
/// helper, a curious Python script the user just ran — can drive
/// inference and consume GPU.
///
/// We close that gap by generating a fresh 32-byte secret per
/// ``ServerManager.start()``, handing it to the child via the
/// ``RAPID_MLX_API_KEY`` env, and adding ``Authorization: Bearer
/// <secret>`` to every request the desktop app issues. The desktop
/// app and the spawned child are the only two parties that share the
/// secret.
///
/// SECURITY NOTE — env-vs-argv delivery has a narrower benefit than
/// the previous comment claimed. The split on macOS is:
///   * argv: ``ps -axww`` shows it to ANY user on the system (the
///     adv_cmds ``ps`` source reads ``kp_proc.p_args`` directly and
///     does no UID check before printing).
///   * env:  ``ps eww <pid>`` only prints env when the caller is
///     ``root`` or the same UID as the target process (adv_cmds gates
///     env behind ``getuid() == 0 || ruid == getuid()`` before
///     decoding it from ``kp_proc.p_args``).
/// So env DOES beat argv in three places:
///   1. Cross-UID snooping — another user on the same Mac can read
///      argv with ``ps -axww`` but cannot read env without root.
///   2. Shoulder-surfing default ``ps`` output — ``ps`` shows argv by
///      default; ``-e`` is required to print env.
///   3. Crash dumps / sysdiagnose bundles that capture argv but not
///      env (asymmetric in practice — both CAN appear depending on
///      the dump path).
///
/// What env DOES NOT defeat is the threat model this bearer was
/// originally meant to address — a same-UID attacker process
/// (sandbox-escaped browser tab, helper script, curious Python
/// script the user just ran). Those processes can call
/// ``ps eww <pid>`` and read the secret directly, just as easily as
/// they can read argv. So the bearer materially raises the bar
/// against careless single-user scripts and stops other-user
/// snooping cold, but a deliberate same-UID reader still wins.
///
/// Honest hardening against same-UID readers requires either a
/// per-launch pipe-based handoff (write the secret to the child's
/// stdin / a unix-socket pair instead of leaking it through the
/// process environment) or NSXPC. Tracked in issue #303 / #305.
///
/// By default the secret rotates on every launch / restart, so a leak via
/// memory dump or log scrub miss is bounded to the current session. The user
/// may opt into a Keychain-backed daily or explicit lifetime; unavailable or
/// malformed persisted credentials always fall back to a one-time secret.
/// ``LogScrubber`` already redacts ``Authorization: Bearer`` and
/// ``--api-key=`` shapes before they reach the log tail.
enum BearerSecret {
    /// Generate a fresh 64-character hex string (32 random bytes).
    ///
    /// Uses ``SecRandomCopyBytes`` for cryptographic-quality
    /// randomness — same source the platform uses for keychain
    /// keys. ``arc4random`` would also work for the unprivileged
    /// case but ``SecRandomCopyBytes`` is the documented "use this"
    /// API for secrets you don't want a debugger / instrumentation
    /// tool to predict.
    ///
    /// Returns ``nil`` only if ``SecRandomCopyBytes`` itself fails
    /// — which on Darwin requires the system to be in a broken
    /// state (RNG entropy starvation pre-userspace). Callers should
    /// treat ``nil`` as "abort start, surface to user" rather than
    /// "spawn without auth" — the latter would silently regress to
    /// the pre-fix behaviour.
    static func generate() -> String? {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = bytes.withUnsafeMutableBytes { buf -> Int32 in
            SecRandomCopyBytes(kSecRandomDefault, buf.count, buf.baseAddress!)
        }
        guard status == errSecSuccess else { return nil }
        return bytes.map { String(format: "%02x", $0) }.joined()
    }
}
