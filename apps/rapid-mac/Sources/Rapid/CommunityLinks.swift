import Foundation

/// The project's public community links, in one place.
///
/// The Discord invite is the **voice channel**: telemetry can only ever
/// show us *what* people do, and this is where they tell us *why*. The
/// same invite is what the README badge, `rapidmlx.com`, and
/// `rapid-mlx feedback` on the CLI side all open — one invite means one
/// community, so it is a named constant rather than a literal typed at
/// each call site.
enum CommunityLinks {
    /// `https://discord.gg/nZcXkUjY5R` — verified against the invite the
    /// repository already publishes; see `FeedbackLinkTests`.
    static let discordInviteURLString = "https://discord.gg/nZcXkUjY5R"

    /// Force-unwrapped deliberately: the string above is a compile-time
    /// literal, so a failure here would mean somebody edited it into
    /// something that is not a URL at all — which the test suite catches
    /// before it can ship.
    static let discordInvite = URL(string: discordInviteURLString)!
}
