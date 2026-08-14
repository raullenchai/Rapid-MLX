import SwiftUI

/// Centralised colour + dimension tokens for the v0.4 refresh.
///
/// The pre-v0.4 surface composed colours ad-hoc at the call site
/// (``Color.accentColor.opacity(0.22)`` on the user bubble,
/// ``Color.secondary.opacity(0.12)`` on the tools chip, system
/// material on the sidebar, etc.). That made it impossible to keep
/// the chrome cohesive — the surfaces drifted apart visually as
/// individual call sites tweaked their opacity values.
///
/// This file exists so every chat-surface paint goes through a
/// single token. When we want to dial the canvas darker or warm
/// the user bubble, we change one constant here instead of
/// chasing the literal across four views.
///
/// All tokens are dark/light aware via ``Color(nsColor:)`` and
/// ``NSColor(name:dynamicProvider:)``. Snapshot tests pin
/// ``NSAppearance.aqua`` so the light variants are what get
/// asserted; manual launch verifies the dark variants by eye.
enum RapidTheme {
    // MARK: - Brand accent
    //
    // The single source of truth for "Rapid blue." Before this token
    // the app leaned on ``Color.accentColor`` everywhere, which is
    // whatever accent the user picked in macOS System Settings — so
    // the product could render pink, graphite, or orange and never
    // reliably showed the Rapid-MLX identity. This is a *softer
    // azure* than the indigo wordmark on the GitHub banner: calmer in
    // a large light UI, friendlier, and still unmistakably "Rapid."
    //
    // Applied app-wide via ``.tint(RapidTheme.brand)`` at the scene
    // root (so every button / link / selection inherits it) AND used
    // directly wherever a literal ``Color.accentColor`` used to paint
    // a brand surface (the empty-state disc, capability glyphs, the
    // compose focus ring). The dark variant lifts toward a brighter
    // sky-blue so the accent keeps its punch against a dark canvas.
    //
    // v0.5: shifted from azure (#2F75EC) toward a softer blue-violet
    // (#5E7CFF) — the Rapid-MLX brand hue. It reads as friendlier and
    // less "default macOS system blue" in a large light UI, matching
    // the ChatGPT / Linear / Apple-desktop feel the refresh targets.
    // v0.6: realigned to the rapidmlx.com design system. ``brand`` is
    // now the website's muted *steel blue* (`--accent` #3A5C86), NOT
    // the old saturated blue-violet and NOT macOS system blue. Paired
    // with a warm amber-gold (the cheetah hue) defined below.
    static let brand = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x6E/255.0, green: 0x96/255.0, blue: 0xC8/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0x3A/255.0, green: 0x5C/255.0, blue: 0x86/255.0, alpha: 1.0)
    }))

    /// Bright blue highlight (`--accent-hi` #7EA8FF) — focus rings and
    /// "live"/hover accents that want a little more pop than ``brand``.
    static let brandHi = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x9C/255.0, green: 0xBE/255.0, blue: 0xFF/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0x7E/255.0, green: 0xA8/255.0, blue: 0xFF/255.0, alpha: 1.0)
    }))

    /// Soft blue wash (`--accent-tint` #EEF2F7) — the calm tinted fill
    /// behind brand-adjacent surfaces that should NOT be a saturated
    /// block: the active sidebar row, status pills, the setup-card icon
    /// halo. Dark mode is the site's deep blue tint.
    static let brandTint = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x1E/255.0, green: 0x2A/255.0, blue: 0x3A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xEE/255.0, green: 0xF2/255.0, blue: 0xF7/255.0, alpha: 1.0)
    }))

    // MARK: - Brand amber (the cheetah-fur hue)

    /// Brand amber-gold (#EFA23A) — energy accents, the primary CTAs
    /// (New chat / Start), the mascot glow, loading states, leopard
    /// spots. This is the warm "yellow" of the Rapid brand.
    static let amber = Color(nsColor: .init(name: nil, dynamicProvider: { _ in
        NSColor(deviceRed: 0xEF/255.0, green: 0xA2/255.0, blue: 0x3A/255.0, alpha: 1.0)
    }))

    /// Deeper amber — amber text / glyphs that need more contrast on a
    /// light surface (raw ``amber`` is a touch light for small text).
    /// A darker shade of the same #EFA23A hue; dark mode uses the
    /// lighter ``amber``.
    static let amberDeep = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0xEF/255.0, green: 0xA2/255.0, blue: 0x3A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xC9/255.0, green: 0x82/255.0, blue: 0x1F/255.0, alpha: 1.0)
    }))

    /// Soft amber wash (`--amber-tint` #FBF1E2) — warm cream fill
    /// behind mascot / setup / energy surfaces.
    static let amberTint = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x2A/255.0, green: 0x21/255.0, blue: 0x13/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xFB/255.0, green: 0xF1/255.0, blue: 0xE2/255.0, alpha: 1.0)
    }))

    /// Speed / success green (`--green` #2E7D55). Reserved for the
    /// "ready" server state — part of the existing status semantics.
    static let green = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x5F/255.0, green: 0xC7/255.0, blue: 0xA0/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0x2E/255.0, green: 0x7D/255.0, blue: 0x55/255.0, alpha: 1.0)
    }))

    // MARK: - Surfaces
    //
    // Card + hairline tokens introduced for the v0.5 light-first pass.
    // The refresh leans on rounded "cards" (settings sections, the
    // setup panel) floating on the window canvas; these two tokens
    // keep every card visually consistent instead of each call site
    // hand-rolling ``Color.secondary.opacity(0.x)``.

    /// App canvas — the off-white surface the chat sits on. A hair
    /// cooler and darker than pure white in light mode so white
    /// ``card`` surfaces and the compose pill read as gently raised
    /// above it (the ChatGPT / Linear "soft gray canvas, white
    /// content" separation). Dark mode is a near-black that's a touch
    /// lifted off pure black for depth.
    static let canvas = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        // v0.6: warm off-white (was cool #F7F7FA) so the canvas pairs
        // with the amber/cheetah accents; dark = the site's `--bg`.
        //
        // Phase UI-2 (Direction D refinement, decision 01 "bone →
        // mineral"): #F8F7F4 → #FAFAF9, and #15171B → #141619. The warm
        // bone was correct on a card and wrong on a whole window — across
        // 1440pt it read as beige, which put a second warm surface in
        // competition with the one amber moment each screen is allowed.
        // #FAFAF9 is a near-white with a slate cast, so amber stays the
        // only warm thing on screen.
        appearance.isDark ? NSColor(deviceRed: 0x14/255.0, green: 0x16/255.0, blue: 0x19/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xFA/255.0, green: 0xFA/255.0, blue: 0xF9/255.0, alpha: 1.0)
    }))

    /// Elevated card fill — a hair lighter than the window canvas in
    /// light mode (near-white), a hair lighter than black in dark
    /// mode. Reads as a raised surface without a heavy drop shadow.
    static let card = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x1A/255.0, green: 0x1D/255.0, blue: 0x21/255.0, alpha: 1.0)
                          : NSColor.white
    }))

    // NOTE: a ``sidebarSurface`` token (cool #F3F5F9) used to sit here.
    // It was superseded by ``surfaceSidebar`` in the v1.0 layer below —
    // the cool grey read as a blue slab beside the warm canvas — and had
    // no consumers left, so it was removed rather than kept as a second,
    // divergent name for the same plane.

    /// Hairline border around cards / inputs (`--line-soft`). A defined
    /// but quiet warm-gray edge in light mode (pairs with the warm
    /// canvas); the site's soft line in dark mode.
    ///
    /// Phase UI-2: re-cast onto the mineral ramp alongside ``canvas``
    /// (#E7E6E1 → #E7E7E4, #262B31 → #26292E). A warm hairline on a
    /// mineral canvas reads as a faint tan line rather than an edge.
    static let hairline = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x26/255.0, green: 0x29/255.0, blue: 0x2E/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xE7/255.0, green: 0xE7/255.0, blue: 0xE4/255.0, alpha: 1.0)
    }))

    /// Standard corner radius for the refresh's rounded cards.
    static let cardRadius: CGFloat = 12

    // MARK: - Message paint
    // The v0.4 scope deliberately limits theme tokens to surfaces the
    // refresh actually repaints — the user pill and the compose pill.
    // A canvas / sidebar / divider token was drafted alongside but
    // dropped before merge: leaving tokens defined-but-unused was
    // worse than no token at all (codex round 1 P1). When the v0.5
    // pass extends the refresh to the window canvas + sidebar
    // material, reintroduce them with the same dark/light dynamic
    // provider pattern below.

    /// Right-aligned user pill background. Warm gray in both
    /// modes — NOT the accent colour, which the v0.3 build leaned
    /// on. ChatGPT-Desktop's user bubble is purely a neutral
    /// container; the accent is reserved for actions.
    static let userBubble = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x33/255.0, green: 0x31/255.0, blue: 0x3D/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xEC/255.0, green: 0xEC/255.0, blue: 0xEE/255.0, alpha: 1.0)
    }))

    /// Foreground text on the user bubble. Auto-flips to white in
    /// dark mode; the system ``.primary`` already does this but
    /// being explicit lets snapshot tests assert a known value.
    static let userBubbleText = Color.primary

    // MARK: - Compose pill

    /// Compose pill background — a warm off-white surface that
    /// reads as actionable. Sits one step elevated above whatever
    /// the system window background paints behind it; the chosen
    /// hex pair is calibrated so the pill is visually distinct
    /// even when the window canvas inherits a neutral system
    /// colour.
    static let composePill = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x2A/255.0, green: 0x28/255.0, blue: 0x33/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xEF/255.0, green: 0xEE/255.0, blue: 0xEC/255.0, alpha: 1.0)
    }))

    /// Subtle outline around the compose pill. A 1-pt hairline is
    /// enough to lift it off the canvas without competing with
    /// the focus ring.
    static let composePillStroke = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(white: 1.0, alpha: 0.06)
                          : NSColor(white: 0.0, alpha: 0.08)
    }))

    /// Send/stop circle fill — high-contrast inverse of the
    /// canvas (black in light mode, near-white in dark mode).
    /// Matches ChatGPT-Desktop's compose-row CTA which uses a
    /// solid neutral so the send action reads as the primary
    /// affordance regardless of the user's system accent.
    static let sendButton = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(white: 0.94, alpha: 1.0)
                          : NSColor(white: 0.08, alpha: 1.0)
    }))

    /// Icon foreground on the send button. Inverse of
    /// ``sendButton`` so the arrow/stop glyph always reads at
    /// AAA contrast.
    static let sendButtonIcon = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(white: 0.08, alpha: 1.0)
                          : NSColor(white: 1.0, alpha: 1.0)
    }))

    /// Disabled-state fill for the send button — the same hue
    /// at ~30% opacity. Keeps the affordance visible (so the
    /// user can see WHERE the button is) without inviting a
    /// click that won't fire.
    static let sendButtonDisabled = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(white: 0.94, alpha: 0.28)
                          : NSColor(white: 0.08, alpha: 0.28)
    }))

    // MARK: - Brand (v0.5.6 — rapidmlx.com alignment)
    //
    // The site's brand spec (PR #1 on raullenchai/rapidmlx.com,
    // landed 2026-06-12) standardises on amber for speed / live /
    // selected accents and forbids blue → amber gradients (solid
    // colours only). Steel-blue remains the engineering/data accent
    // but lives off the chat surface. The desktop empty-state hero
    // previously used ``Color.accentColor`` which falls back to the
    // user's system accent — a saturated blue on most machines —
    // and read as "default SwiftUI sample." These tokens pull the
    // chrome into the brand the user sees in the browser.

    /// Amber-tint background for the hero disc. Mirrors the site's
    /// ``--amber-tint`` (#FBF1E2 / #2A2113). Solid fill, no gradient
    /// — per the site spec, no two-stop blends on the brand axis.
    static let brandAmberTint = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x2A/255.0, green: 0x21/255.0, blue: 0x13/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xFB/255.0, green: 0xF1/255.0, blue: 0xE2/255.0, alpha: 1.0)
    }))

    /// Deep amber for the cheetah silhouette on the hero disc.
    /// Site's ``--amber`` (#CC8730 light) lifts to ``--amber #E0A95A``
    /// in dark mode for legibility against the dark-tint surface.
    static let brandAmber = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0xE0/255.0, green: 0xA9/255.0, blue: 0x5A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xCC/255.0, green: 0x87/255.0, blue: 0x30/255.0, alpha: 1.0)
    }))

    // MARK: - Dimensions
    //
    // ``userBubbleRadius`` (18) was removed here: its only call site now
    // reads ``Radius.bubble`` (14), and leaving the old constant behind
    // would have been a second, divergent radius for the same shape —
    // exactly the drift the ``Radius`` group exists to end.

    /// Compose pill corner radius. v0.5: tightened 22 → 18 to match
    /// the user bubble exactly — the larger radius read as a bubbly,
    /// oversized field; 18 is calmer and more "modern AI input."
    /// v1.0 visual foundation: superseded by ``Radius.input`` (10) —
    /// kept so the legacy token name still resolves, now pointing at
    /// the single input radius rather than its own value.
    static let composePillRadius: CGFloat = Radius.input

    // MARK: - v1.0 semantic layer
    //
    // Everything below is the Phase-1 "visual foundation": a semantic
    // vocabulary business views spend instead of writing literals.
    //
    // The brand hierarchy this encodes (and which the pre-v1.0 tokens
    // above got backwards):
    //
    //   * AMBER (#EFA23A) is the FIRST brand colour. Primary CTAs,
    //     selection, focus, working/progress states, key icons, the
    //     cheetah moments.
    //   * STEEL BLUE (#3A5C86) is SECONDARY. Informational data,
    //     links, utility icons, engineering detail. It no longer fills
    //     primary buttons.
    //   * GREEN (#2E7D55) means Ready / success and nothing else.
    //   * RED means error / destructive and nothing else.
    //   * Warm neutrals carry every large surface, so the product reads
    //     as a calm desktop app with an amber accent — not an orange
    //     theme.
    //
    // The legacy tokens above are intentionally left in place and
    // re-pointed here rather than deleted: they have call sites across
    // the app, and a rename sweep is churn this phase doesn't need.

    // MARK: Brand

    /// The primary brand colour. Amber #EFA23A holds in both modes —
    /// it is the product's single strongest visual memory, and shifting
    /// it per-appearance would weaken the recall it exists to build.
    static let brandPrimary = amber

    /// Amber for small text and glyphs on a light surface. Raw
    /// ``brandPrimary`` fails contrast under ~15pt on warm white, so
    /// anything type-sized uses this deeper shade of the same hue.
    static let brandPrimaryDeep = amberDeep

    /// The calm amber wash behind selected rows, working states, and
    /// brand-adjacent surfaces that must not become a saturated block.
    static let brandPrimaryTint = amberTint

    /// Foreground for content sitting ON a ``brandPrimary`` fill.
    ///
    /// Deliberately a near-black graphite, never white. White on
    /// #EFA23A lands around 2.0:1 — below every WCAG threshold — and is
    /// exactly the low-contrast default this phase was asked to remove.
    /// Graphite on amber measures ~9:1.
    ///
    /// It is appearance-INDEPENDENT on purpose. The amber fill is the
    /// same in Light and Dark (see ``amber``), so the ink on it must be
    /// too: flipping to white in Dark would recreate the 2:1 pairing on
    /// exactly the surface this token exists to protect.
    static let onBrandPrimary = Color(nsColor: .init(name: nil, dynamicProvider: { _ in
        NSColor(deviceRed: 0x24/255.0, green: 0x1A/255.0, blue: 0x08/255.0, alpha: 1.0)
    }))

    /// The canonical name for "ink on an amber-filled control".
    ///
    /// Same value as ``onBrandPrimary``, which predates it and is still
    /// what the action tokens below are defined in terms of. This is the
    /// name new call sites should reach for: every amber-filled surface
    /// in the app — primary buttons, selected segmented items, selected
    /// filter chips — reads its foreground from here, so there is one
    /// place to look when asking "why is that text dark?" and no reason
    /// for anybody to write `.foregroundStyle(.white)` on amber again.
    static let brandOnAccent = onBrandPrimary

    /// The secondary brand colour — steel blue. Data, links, secondary
    /// icons, engineering detail. Aliases the legacy ``brand`` token.
    static let brandSecondary = brand

    /// Soft steel-blue wash. Aliases the legacy ``brandTint``.
    static let brandSecondaryTint = brandTint

    // MARK: Surfaces
    //
    // Four planes, warmest-and-lowest to raised. Light mode is a warm
    // greyscale, NOT four shades of near-white; dark mode is authored
    // on its own ramp rather than an inversion of the light one (the
    // dark values carry a touch more blue so the amber accent stays
    // warm against them).

    /// The window canvas — the plane chat, pages, and empty states sit on.
    static let surfaceCanvas = canvas

    /// The sidebar rail. v1.0: re-warmed from the old cool #F3F5F9,
    /// which read as a blue-grey slab beside the warm canvas and made
    /// the whole left column feel like a different product.
    ///
    /// Phase UI-2 re-casts it onto the mineral ramp (#F2F0EB → #F3F3F1,
    /// #1A1C20 → #191B1E) so it stays one step below ``surfaceCanvas``
    /// without being a warmer plane than it. Direction D decision 04:
    /// the rail is mineral in BOTH the resting and the working
    /// composition — it never goes graphite, because dark chrome that
    /// persists after the work ends is a theme, and this is not one.
    static let surfaceSidebar = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x19/255.0, green: 0x1B/255.0, blue: 0x1E/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xF3/255.0, green: 0xF3/255.0, blue: 0xF1/255.0, alpha: 1.0)
    }))

    /// A raised surface — cards, popovers, grouped rows.
    static let surfaceRaised = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x1D/255.0, green: 0x20/255.0, blue: 0x24/255.0, alpha: 1.0)
                          : NSColor.white
    }))

    /// The ground under code, endpoints, and keys. Recessed relative to
    /// ``surfaceRaised`` — a snippet should read as inset into a card,
    /// not as a second card floating on top of one.
    static let surfaceCode = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x14/255.0, green: 0x16/255.0, blue: 0x1A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xF4/255.0, green: 0xF2/255.0, blue: 0xED/255.0, alpha: 1.0)
    }))

    /// Sheets and popovers — a hair lighter than ``surfaceRaised`` in
    /// dark mode so a sheet over a card still separates.
    static let surfaceOverlay = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x23/255.0, green: 0x26/255.0, blue: 0x2C/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xFC/255.0, green: 0xFB/255.0, blue: 0xF9/255.0, alpha: 1.0)
    }))

    /// A more present divider for structural separation (card headers,
    /// grouped-row separators) where ``hairline`` disappears.
    static let hairlineStrong = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x32/255.0, green: 0x36/255.0, blue: 0x3C/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xD5/255.0, green: 0xD5/255.0, blue: 0xD1/255.0, alpha: 1.0)
    }))

    // MARK: Lifecycle band (the "active AI work" ground)
    //
    // Direction D's second composition. A surface only paints these while
    // the app is doing work the user is waiting on — a download, a model
    // load — and drops straight back to the mineral ramp when it ends.
    // Graphite is the ground, amber is the single progress hue on it, and
    // the ink on amber is the same near-black every other amber fill uses.
    //
    // It is a COMPOSITION, not a theme: the sidebar and the status strip
    // stay mineral throughout, and nothing here survives the work.

    /// Ground for the lifecycle band. Near-black in Light so the band
    /// reads as a distinct plane over the near-white canvas; one step
    /// LIGHTER than the canvas in Dark, where a darker band would simply
    /// disappear into the window.
    static let surfaceBand = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x23/255.0, green: 0x26/255.0, blue: 0x2B/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0x1B/255.0, green: 0x1D/255.0, blue: 0x21/255.0, alpha: 1.0)
    }))

    /// Primary ink on ``surfaceBand``. Appearance-independent, because the
    /// band's ground is graphite in both modes — flipping it would put
    /// dark ink on a dark plane in exactly one appearance.
    static let bandInk = Color(nsColor: .init(name: nil, dynamicProvider: { _ in
        NSColor(deviceRed: 0xF4/255.0, green: 0xF3/255.0, blue: 0xF1/255.0, alpha: 1.0)
    }))

    /// Supporting ink on ``surfaceBand`` — the eyebrow, byte counts, ETA.
    /// ~7:1 on the band ground, so it is quiet without being unreadable.
    static let bandInkSecondary = Color(nsColor: .init(name: nil, dynamicProvider: { _ in
        NSColor(deviceRed: 0xA5/255.0, green: 0xA8/255.0, blue: 0xAE/255.0, alpha: 1.0)
    }))

    /// Unfilled progress track inside the band.
    static let bandTrack = Color(nsColor: .init(name: nil, dynamicProvider: { _ in
        NSColor(deviceRed: 0x33/255.0, green: 0x37/255.0, blue: 0x3D/255.0, alpha: 1.0)
    }))

    // MARK: Status
    //
    // One colour per lifecycle meaning. Views switch on state and read
    // a token; they never pick a hue themselves.

    /// Not running, nothing pending. Deliberately neutral — an idle
    /// server is not a warning.
    static let statusIdle = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x8A/255.0, green: 0x90/255.0, blue: 0x99/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0x8A/255.0, green: 0x86/255.0, blue: 0x7E/255.0, alpha: 1.0)
    }))

    /// Starting, downloading, benchmarking — anything in flight. Amber,
    /// which is also the brand colour: progress is on-brand by design.
    static let statusWorking = brandPrimary

    /// Ready / success. The only role green plays.
    static let statusReady = green

    /// Error / failure. The only role red plays.
    static let statusError = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0xFF/255.0, green: 0x6B/255.0, blue: 0x5E/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xC0/255.0, green: 0x39/255.0, blue: 0x2B/255.0, alpha: 1.0)
    }))

    /// Tinted backing for an error surface (inline notices, failure rows).
    static let statusErrorTint = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x33/255.0, green: 0x1C/255.0, blue: 0x1A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xFB/255.0, green: 0xEC/255.0, blue: 0xEA/255.0, alpha: 1.0)
    }))

    /// Needs-attention-but-not-broken: a drive that went away, a config
    /// edit that has not been applied, a connector that did not come up.
    ///
    /// Amber, deliberately the same family as ``statusWorking`` — the
    /// product only has one "look here" hue and inventing a second
    /// (system orange, as five Settings call sites did) put two
    /// near-identical ambers on the same window. It is a SEPARATE token
    /// from the brand one because the meanings are separate: if warning
    /// ever needs to diverge from brand amber, it moves here and no call
    /// site changes.
    static let statusWarning = brandPrimaryDeep

    /// Tinted backing for a warning surface.
    static let statusWarningTint = brandPrimaryTint

    /// Tinted backing for a success surface (a completed delete, a saved
    /// key). Pairs with ``statusReady`` the way ``statusErrorTint``
    /// pairs with ``statusError``.
    static let statusReadyTint = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x18/255.0, green: 0x2A/255.0, blue: 0x22/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xEC/255.0, green: 0xF5/255.0, blue: 0xF0/255.0, alpha: 1.0)
    }))

    // MARK: Text
    //
    // Named text roles so a view says what a string IS rather than how
    // dim it should look. They resolve to the system's dynamic label
    // colours — which already track appearance, increased contrast, and
    // the vibrancy of whatever surface they sit on — so this is about
    // giving the roles a shared home, not about repainting text.

    /// Default reading colour: row labels, values, body copy.
    static let textPrimary = Color.primary

    /// Supporting copy — a row's explanatory line, a section subtitle,
    /// a caption. Never used for a control's own label.
    static let textSecondary = Color.secondary

    /// The quietest tier: legends, footers, and counts that must not
    /// compete with the content they describe.
    static let textTertiary = Color(nsColor: .tertiaryLabelColor)

    /// Text belonging to a control that is switched off. Distinct from
    /// ``textTertiary``: this one means "unavailable", not "quiet", and
    /// pairs with ``disabledOpacity`` on the control around it.
    static let textDisabled = Color(nsColor: .disabledControlTextColor)

    // MARK: Actions

    /// Fill for the single highest-emphasis action on a surface.
    static let primaryActionFill = brandPrimary
    /// Label on ``primaryActionFill``.
    static let primaryActionLabel = onBrandPrimary
    /// Label/icon for a secondary (outlined) action. Neutral graphite.
    ///
    /// v1.0.1: was ``brandSecondary``. Painting every secondary control
    /// steel blue turned a *supporting* colour into the most repeated
    /// hue on the surface — three filled blue "Copy config" buttons, a
    /// blue glyph on every endpoint row, a blue icon per tool. Steel
    /// blue is now rare by default and earned on hover.
    static let secondaryActionLabel = Color.primary
    /// Fill for a destructive action.
    static let destructiveActionFill = statusError
    /// Label on ``destructiveActionFill``.
    ///
    /// Adaptive, because the fill is: ``statusError`` is a deep brick
    /// (#C0392B) in Light, where white measures ~5.9:1, but a much
    /// lighter coral (#FF6B5E) in Dark, where white falls to ~2.4:1 —
    /// the same failure mode as white-on-amber, on the one button whose
    /// label a user most needs to read before pressing. Dark mode
    /// therefore takes near-black ink, which measures ~7:1 on that
    /// coral.
    static let destructiveActionLabel = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x24/255.0, green: 0x0C/255.0, blue: 0x0A/255.0, alpha: 1.0)
                          : NSColor.white
    }))
    /// A quiet, borderless action — present but not competing.
    static let quietActionLabel = Color.secondary

    // MARK: Utility controls
    //
    // Copy glyphs, reveal toggles, per-row actions: things that are
    // genuinely useful but must not read as calls to action. Neutral at
    // rest, steel blue under the pointer (the hover is where the
    // secondary brand colour earns its place), ready-green on success.

    /// Resting colour for a utility icon/label.
    static let utilityActionLabel = Color.secondary
    /// Hover colour for a utility icon/label.
    ///
    /// v1.0.2: amber, not steel blue. An active control lighting up is
    /// a brand moment; routing every hover through the secondary colour
    /// made steel blue the most frequently-seen accent in the app,
    /// which is the opposite of "rare supporting colour".
    static let utilityActionHover = brandPrimaryDeep
    /// A utility action that just succeeded (copied, saved).
    static let utilityActionSuccess = statusReady

    /// Genuine text links. After v1.0.2 this is essentially the ENTIRE
    /// remaining budget for steel blue: not buttons, not icons, not
    /// hover, not notices — links, and the occasional single deliberate
    /// technical accent on a surface that has earned one.
    static let linkLabel = brandSecondary

    /// Keyboard-focus ring. Amber, per the brand hierarchy: focus is a
    /// primary-attention signal.
    static let focusRing = brandPrimary

    // MARK: Interaction

    /// Fill behind a SELECTED row.
    ///
    /// Direction D decision 02: neutral, not amber. The v1.0 treatment
    /// was an amber tint plus an amber label, and on the warm rail the
    /// two were within a few percent of the surface they sat on — the
    /// selected row was, in practice, invisible. The signal now comes
    /// from three things that do not depend on a tint being legible: a
    /// 3pt amber leading bar (``selectionBarWidth``), this neutral fill,
    /// and a graphite SEMIBOLD label. Weight and a hard-edged bar survive
    /// Increase Contrast, colour-blindness, and a glance; a 4% wash does
    /// not.
    static let selectionFill = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x24/255.0, green: 0x27/255.0, blue: 0x2B/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xEF/255.0, green: 0xEF/255.0, blue: 0xEC/255.0, alpha: 1.0)
    }))

    /// The leading bar that marks a selected row. Amber, and the one
    /// amber element the rail is allowed.
    ///
    /// A deeper step of the same hue in Light, raw ``brandPrimary`` in
    /// Dark — and this is the token where that split is load-bearing
    /// rather than a nicety. Measured against the rail:
    ///
    ///   * Dark: #EFA23A on #191B1E is 8.1:1. Nothing to fix.
    ///   * Light: #EFA23A on #F3F3F1 is **1.9:1** — a light amber on a
    ///     near-white plane. The bar carries the whole selection signal
    ///     under the refined rule, so shipping it at 1.9:1 would have
    ///     replaced one invisible selected row with another.
    ///
    /// #B87317 measures 3.4:1 against the rail and 3.3:1 against the
    /// selected row's own fill, clearing the 3:1 non-text threshold with
    /// margin at 3pt wide. It is deeper than ``brandPrimaryDeep``
    /// (#C9821F, 2.8:1 here) because that token was tuned for TEXT, where
    /// the glyph's own stroke width and the reader's foveal attention do
    /// some of the work; a 3pt rule caught in peripheral vision has
    /// neither.
    static let selectionBar = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0xEF/255.0, green: 0xA2/255.0, blue: 0x3A/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xB8/255.0, green: 0x73/255.0, blue: 0x17/255.0, alpha: 1.0)
    }))

    /// Hover wash over a neutral row or control.
    ///
    /// Phase UI-2: a concrete mineral value rather than
    /// ``Color.primary.opacity(0.055)``. The opacity form composited
    /// differently on each of the four surface planes, so the same hover
    /// landed a visibly different colour in the sidebar than on a card —
    /// and in Dark it lifted toward blue-white while everything around it
    /// stayed slate. One value keeps hover reading as the same gesture
    /// wherever the pointer is, and keeps it a clear step BELOW
    /// ``selectionFill`` so hovering a row never looks like selecting it.
    static let hoverFill = Color(nsColor: .init(name: nil, dynamicProvider: { appearance in
        appearance.isDark ? NSColor(deviceRed: 0x21/255.0, green: 0x24/255.0, blue: 0x29/255.0, alpha: 1.0)
                          : NSColor(deviceRed: 0xF0/255.0, green: 0xF0/255.0, blue: 0xED/255.0, alpha: 1.0)
    }))
    /// Hover wash for controls that draw it ON TOP of their own fill.
    ///
    /// ``hoverFill`` is an opaque plane colour, which is right for the rows
    /// and chips that paint it BEHIND their content — that is what Phase UI-2
    /// made it, so the same hover lands the same colour on all four surfaces.
    /// A button cannot use it: ``RapidButtonSurface`` composites hover over an
    /// amber or white fill it must not replace, so an opaque value there
    /// covers the label instead of shading it. The button surface hovered to a
    /// solid grey block with no text, and pressing it brought the text back —
    /// ``pressedFill`` beside this is still translucent, so the same overlay
    /// read as a wash on press and as a lid on hover.
    ///
    /// This is the value ``hoverFill`` carried before it became a plane, kept
    /// here for the over-content case only. It composites differently across
    /// surfaces — the reason Phase UI-2 moved off it — but a wash sitting on
    /// the control's own fill is exactly where that behaviour is wanted.
    static let hoverWash = Color.primary.opacity(0.055)
    /// Pressed wash — one step firmer than ``hoverWash``, and drawn in the
    /// same over-content position, which is why it is an opacity rather than
    /// a plane colour.
    static let pressedFill = Color.primary.opacity(0.10)
    /// Multiplier applied to a control's opacity when disabled.
    ///
    /// v1.0.2: raised 0.40 → 0.62. A disabled control still has to be
    /// READ — a disabled "Copy config" explains, via its tooltip, what
    /// would make it available, and at 0.40 on a
    /// warm canvas the label was close to invisible. 0.62 keeps the
    /// unmistakable "not right now" signal while leaving the text
    /// legible.
    static let disabledOpacity: Double = 0.62

    // MARK: - Spacing
    //
    // One rhythm for the whole app: 2 / 4 / 8 / 12 / 16 / 24 / 32. Page
    // margins, section gaps, and control padding all come from here so
    // they can never drift apart per-view.

    enum Space {
        /// 2 — a label and the line directly under it; the tightest gap
        /// that still reads as two lines rather than one block.
        static let xxs: CGFloat = 2
        /// 4 — icon-to-label, tight glyph gaps.
        static let xs: CGFloat = 4
        /// 8 — inside a control, between related chips.
        static let sm: CGFloat = 8
        /// 12 — row padding, gap between grouped rows.
        static let md: CGFloat = 12
        /// 16 — card padding, gap between cards.
        static let lg: CGFloat = 16
        /// 24 — page margin, gap between sections.
        static let xl: CGFloat = 24
        /// 32 — major vertical separation.
        static let xxl: CGFloat = 32
        /// 48 — the gap around a hero composition (the chat empty state's
        /// breathing room). Above ``xxl`` the rhythm stops being "space
        /// between things" and becomes "space a thing sits in".
        static let xxxl: CGFloat = 48
        /// 72 — full-window compositions only.
        static let huge: CGFloat = 72
    }

    // MARK: - Radii
    //
    // One radius per shape ROLE, not per view. The pre-v1.0 surface had
    // 6/8/10/12/16/18/22 in play simultaneously, which is why nothing
    // felt like it belonged to one system.

    /// v1.0.1: tightened again. 12pt cards over 8pt buttons over 12pt
    /// inputs still read as soft — three roundnesses competing. Cards,
    /// buttons, and rows now share 8; inputs get 10 so a field is
    /// distinguishable from a button at a glance; only chat bubbles
    /// stay soft, because a message is the one thing that should feel
    /// like an object rather than a control.
    enum Radius {
        /// Cards and grouped containers.
        static let card: CGFloat = 8
        /// Text fields and the composer.
        static let input: CGFloat = 10
        /// Buttons.
        static let button: CGFloat = 8
        /// Selectable rows — sidebar, lists, menu rows.
        static let row: CGFloat = 8
        /// A segment inside a segmented control. One step tighter than
        /// ``row`` so the selected segment reads as sitting INSIDE its
        /// track rather than as a free-floating pill.
        static let segment: CGFloat = 6
        /// Chat bubbles. Deliberately the one soft shape left.
        static let bubble: CGFloat = 14
        /// Inset code / endpoint blocks. Tighter than the card that
        /// contains them so the inset reads as recessed, not nested.
        static let code: CGFloat = 6
        /// A panel that frames a whole composition rather than a control
        /// — the Images stage, the aspect preview. Softer than ``card``
        /// because it holds content, not rows.
        static let panel: CGFloat = 12
    }

    // MARK: - Control heights
    //
    // Accessibility floor is 28pt for anything clickable; a primary
    // action is 36pt. Both are enforced by the shared button styles.

    enum ControlHeight {
        /// 24 — inline icon buttons inside dense rows. Only for
        /// controls with a larger surrounding hit target.
        static let mini: CGFloat = 24
        /// 28 — the standard minimum tappable control.
        static let small: CGFloat = 28
        /// 32 — comfortable default for secondary buttons.
        static let medium: CGFloat = 32
        /// 36 — primary actions.
        static let large: CGFloat = 36
        /// 30 — sidebar / list row height.
        static let row: CGFloat = 30
        /// 30 — segmented controls. macOS's own segmented control is
        /// 28–32pt at the regular control size; the pre-refinement
        /// ``.pickerStyle(.segmented)`` rendered nearer 40 because it
        /// inherited the scene tint and the panel's larger type.
        static let segmented: CGFloat = 30
    }

    // MARK: - Layout

    enum Layout {
        /// Reading measure for chat + prose. Content never stretches
        /// edge-to-edge on a wide window.
        static let contentMaxWidth: CGFloat = 720
        /// Max width for a settings/tool page's content column.
        static let pageMaxWidth: CGFloat = 640
        /// Max width for a block asking the user to decide one thing.
        /// Narrower than a reading measure on purpose — a choice should
        /// be takeable in one eye movement.
        static let decisionMaxWidth: CGFloat = 460
        /// Fixed leading slot every row icon occupies, so labels align
        /// down a column regardless of glyph width.
        static let iconSlot: CGFloat = 18
        /// Width of the amber bar marking a selected row.
        static let selectionBarWidth: CGFloat = 3
        /// Height of that bar. Shorter than the 30pt row so it reads as a
        /// marker beside the label rather than a full-height divider.
        static let selectionBarHeight: CGFloat = 18

        /// The detail-pane widths produced at the three window widths the
        /// layout is verified at (720 / 1000 / 1440), after the 200pt ideal
        /// sidebar is removed.
        ///
        /// These are not CSS media queries — SwiftUI has no such thing,
        /// and the app resizes continuously. They are the widths the
        /// design was drawn and reviewed at, named so a view that DOES
        /// need to change proportion at a threshold reads that threshold
        /// from one place instead of hard-coding it.
        enum Breakpoint {
            /// 520 — detail pane at the 720pt compact window.
            static let floor: CGFloat = 520
            /// 800 — detail pane at the 1000pt working window.
            static let mid: CGFloat = 800
            /// 1240 — detail pane at the 1440pt wide window.
            static let wide: CGFloat = 1240
        }
    }
}

// MARK: - User-facing model naming

/// Turns an internal alias + lifecycle state into copy a person can read.
///
/// Exists because internal placeholders were reaching the surface as if
/// they were model names. ``DownloadProgress.StartupActivity`` has a
/// ``.loading`` phase whose label is the bare word "Loading", and an
/// unresolved alias is the empty string — neither is a model, and
/// neither should ever be rendered where a user expects one.
///
/// Pure and free-standing so every surface (chat empty state, the model
/// picker, settings) answers "what model am I using?" the same
/// way, and so the mapping is unit-testable without SwiftUI.
enum ModelDisplayName {
    /// Internal placeholder strings that must never surface as a name.
    /// Compared case-insensitively.
    private static let placeholders: Set<String> = [
        "loading", "starting", "warming up", "downloading", "unknown", "none",
    ]

    /// True when ``alias`` is absent or is an internal placeholder.
    static func isUnresolved(_ alias: String) -> Bool {
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        return placeholders.contains(trimmed.lowercased())
    }

    /// The name to show in running prose ("Chatting with …").
    ///
    /// While the server is coming up we deliberately do NOT show the
    /// alias, even when we know it: the honest statement is that the
    /// model is being prepared, not that you are talking to it.
    static func conversational(alias: String, state: ServerState) -> String {
        if case .starting = state { return "Preparing your local model…" }
        return isUnresolved(alias) ? "your local model" : alias
    }

    /// The name to show in a config value slot (Connect Tools' `Model`
    /// row, copied snippets). ``nil`` means "no real value yet" — the
    /// caller must not present a placeholder as a working config.
    static func configValue(alias: String) -> String? {
        isUnresolved(alias) ? nil : alias
    }
}

// MARK: - Typography

/// The type ramp. Eight roles, native SF throughout.
///
/// Monospaced is reserved for code, endpoints, keys, and metrics —
/// never for prose. A monospaced sentence reads as terminal output,
/// which is precisely the "this is a CLI with a window around it"
/// impression the product is trying to shed.
///
/// Sizes are fixed rather than `Font.TextStyle`-relative because the
/// desktop density target is tighter than the system defaults. Views
/// that must honour Dynamic Type keep using ``scaledSystemFont``; this
/// ramp covers the chrome.
enum RapidFont {
    /// The one display line on a surface that has nothing else on it —
    /// the chat empty state's "Ask anything", and nothing else in this
    /// slice. 34pt semibold over a 40pt leading.
    ///
    /// The ramp previously topped out at ``pageTitle`` (20pt), which is a
    /// *page heading* size: correct above a list of settings rows, far
    /// too small to be the only object in a 1440pt window. An empty
    /// surface has no hierarchy to sit inside, so its title has to
    /// establish one on its own.
    static let displayTitle = Font.system(size: 34, weight: .semibold)

    /// Optical tracking for ``displayTitle``. SF tightens as it grows;
    /// at 34pt the default spacing reads loose, and -0.022em is where the
    /// word closes up without the letters touching.
    static let displayTitleTracking: CGFloat = -0.75

    /// Supporting line under ``displayTitle``. 14pt — one step above
    /// ``body``, because at display scale a 12pt subtitle reads as a
    /// footnote attached to the wrong heading.
    static let displaySubtitle = Font.system(size: 14)

    /// The model name inside the lifecycle band. Between ``pageTitle``
    /// and ``displayTitle``: the band is a priority surface, but it is
    /// still chrome around a transcript, not the subject of the window.
    static let bandTitle = Font.system(size: 21, weight: .semibold)

    /// The all-caps eyebrow over a band or stage title, and the
    /// monospaced counters beside it. Monospaced because these values
    /// tick live (bytes, percent, step counts) and proportional digits
    /// make a progress line jitter its own width.
    static let bandEyebrow = Font.system(size: 10, weight: .semibold, design: .monospaced)

    /// The big amber percentage in the lifecycle band.
    static let bandMetric = Font.system(size: 34, weight: .semibold, design: .default)

    /// Window / toolbar title.
    static let windowTitle = Font.system(size: 15, weight: .semibold)
    /// The one big title on a page. Chat empty state, page headers,
    /// the title at the top of a Settings category.
    static let pageTitle = Font.system(size: 20, weight: .semibold)
    /// A titled division WITHIN a page — "Models folder", "Web search",
    /// "Approvals". The tier between ``pageTitle`` and ``groupLabel``.
    ///
    /// This is the role Settings was missing, and its absence is why the
    /// window carried four heading sizes: panels reached for `.title2`
    /// (22pt) or `.title3` (17pt) because the ramp offered nothing
    /// between a 20pt page title and an 11pt group label. 15pt semibold
    /// sits a clear step under ``pageTitle`` without shouting.
    static let sectionTitle = Font.system(size: 15, weight: .semibold)
    /// The quiet organisational label directly above a group of rows.
    /// Structural, not announcing — it should recede behind the content
    /// it files.
    ///
    /// Formerly named ``sectionTitle``; renamed when the real section
    /// tier above was added, because two roles cannot share one name.
    static let groupLabel = Font.system(size: 11, weight: .semibold)
    /// Default body copy and row labels.
    static let body = Font.system(size: 13)
    /// Emphasised body — a row's primary label.
    static let bodyEmphasis = Font.system(size: 13, weight: .medium)
    /// Supporting copy under a title or row label.
    static let secondary = Font.system(size: 12)
    /// The smallest supporting text: hints, footnotes, timestamps.
    static let caption = Font.system(size: 11)
    /// A number meant to be compared or watched. Monospaced digits so
    /// a live-updating value doesn't jitter its own layout.
    static let metric = Font.system(size: 11, design: .monospaced)
    /// Code, endpoints, and API keys.
    static let code = Font.system(size: 11, design: .monospaced)
}

private extension NSAppearance {
    /// True if this appearance asks for a dark surface. Covers
    /// the two macOS dark names (aqua-dark and accessibility
    /// high-contrast dark). ``bestMatch`` returns the first
    /// matching name so we can ask "is this aqua-light or
    /// aqua-dark?" without enumerating every variant.
    var isDark: Bool {
        let match = bestMatch(from: [.aqua, .darkAqua, .accessibilityHighContrastDarkAqua])
        return match == .darkAqua || match == .accessibilityHighContrastDarkAqua
    }
}
