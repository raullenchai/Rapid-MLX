# Desktop UI language: an in-app switch, and what actually drives it

Status: implemented (`apps/rapid-mac`)
Owner: Pixel (UI/UX); reaches `RapidApp` wiring, so Atlas was consulted on the
two-mechanism split below.

## What was wrong

The Desktop app compiled a `zh-Hans` table and shipped
`Contents/Resources/zh-Hans.lproj`, but **nothing could select it**. macOS picks
an app's language from the system setting, so a user on an English Mac saw
English copy and had no way to ask for anything else. Settings had no Language
row — the nine categories are Model Management, System Prompt, Memory, Tools,
Performance, Experimental, Appearance, Privacy, App.

The catalog was also far smaller than the interface. `xcstringstool print`
listed 80 keys; the compiler sees 553 localizable sites. And the gap was not
only missing translations:

- **Copy typed as `String` never localizes.** `Text(someString)` is the verbatim
  initialiser. `SectionHeader.title`, `SettingsSection.title`,
  `SettingsRowLabel.title`, `EmptyState.title`, `InstructionTextEditor.placeholder`
  and the sidebar's `row(title:)` were all `String`, so ~75 static call sites —
  the whole Settings shell, the rail, the chat empty state — bypassed the
  catalog no matter how complete it was.
- **`ModelReadiness`** built the entire readiness vocabulary by concatenation
  (`"\(a) isn't downloaded yet"`), which leaves the alias outside the
  translatable unit.
- **The tray menu and every AppKit surface** (`NSAlert`, `NSMenuItem`, window
  titles) were English literals.
- **`MarkdownCodeBlockView` hard-coded `"复制"` / `"已复制"`**, so the code-block
  Copy button read Chinese in *every* language, including English. Its baselines
  had captured that, which is how it survived.
- **`SettingsView.Category.title`** was `String` too, so the Settings *rail* —
  the one piece of chrome visible on every Settings screen — stayed English
  while every panel behind it was translated. This one survived a full review
  pass and a screenshot check of the panel area; it only showed up in a capture
  that included the rail. `LocalizationTests` now pins a `zh-Hans` entry for
  every category title, because a missing entry is invisible in a diff.

## Decision 1 — how the switch takes effect

There are two halves, and a new surface has to be on the right one:

| Surface | Mechanism |
| --- | --- |
| SwiftUI copy (`Text("…")`, `Label`, composer placeholders) | `\.locale` environment, published from `LanguageConfig.resolvedLocale` |
| Foundation + AppKit (`String(localized:)`, `NSMenuItem`, `NSAlert`, window titles) | `BundleLanguageOverride`, a one-time swizzle of `Bundle.localizedString(forKey:value:table:)` |

**Why not `AppleLanguages`.** Writing it into `UserDefaults` is the documented
Apple route, but it only takes effect on the next launch. That reads as a broken
control: the user picks 简体中文, the picker snaps to it, and nothing on screen
changes.

**Why `\.locale` carries SwiftUI rather than the swizzle.** Measured, not
assumed: rendering the *same* `Text` value twice — once before and once after
activating a bundle redirect — produces identical pixels, while
`Text("New chat").environment(\.locale, …)` renders differently. SwiftUI
resolves a `LocalizedStringKey` against the locale in scope and caches the
result in the view value. An environment change is the thing that invalidates
it.

That distinction decides the UX. Forcing the switch through view identity
(`.id(language)` on each window's content) would have worked and is the common
recipe, but it destroys every `@State` in the tree — including
`ChatView.draft`, a half-typed message. Riding `\.locale` re-renders what reads
it and leaves view-local state alone.

The bundle redirect stays for the calls SwiftUI does not own. It is scoped to
`Bundle.main` on purpose: a blanket redirect would also rewrite lookups inside
Sparkle and MarkdownUI, whose string tables are not ours. An unknown `.lproj`
code degrades to "no override" rather than blanking every string.

## Decision 2 — where the picker lives

Settings → **Appearance**, beside the theme. A dedicated Settings category would
be marginally more discoverable, but the sidebar category list is captured in 12
committed AX baselines; adding a row invalidates all 12 for one control. The
Appearance panel is the same "how the app presents itself" surface and is not
itself baselined.

The picker offers **Follow the system / English / 简体中文**. Language names are
rendered in their own language — a translated "English" is unfindable to someone
stranded in a language they cannot read — so only the "follow the system" row is
a catalog key. The menu mirrors `Auto (follow system)` from the theme picker so
the two rows cannot drift.

A fresh install defaults to **`.system`**, unlike the light-first theme default:
silently overriding the macOS language for every new user is not ours to decide.

## Decision 3 — the baselines are English, so the harness says so

Once the copy is localizable, `gui-golden-flows.sh` had a latent dependency on
the *runner's* macOS language: the same build renders different copy on a
Chinese machine. The harness now launches with `-AppleLanguages "(en)"` unless a
flow sets `RAPID_GUI_APP_LANGUAGE` (only `flow_localized_photo_hint` does, for
`zh-Hans`). Whether a comparison passes depends on the product, not the host.

`chat-depth.*` baselines moved `title="复制"` → `title="Copy"` for the same
reason: the button resolves through the catalog now, and English is what the
harness asks for.

## One translation caveat worth knowing

The catalog key `Memory` is **overloaded**, and the two senses need different
Chinese words:

- The Settings pane (entries learned from conversations) and its rail row:
  **记忆**.
- The onboarding hardware row (`OnboardingDirectionD.row("Memory", …)`,
  reporting installed RAM): **内存**.

The value is currently `记忆`, which is right for every site that resolves it
today — the onboarding row passes a literal into a `String` parameter and is not
plumbed yet, so it renders English and ignores the catalog. **Whoever plumbs
onboarding must give that row its own key** rather than reusing this one, or the
RAM line will read 记忆.

## Rebuilding the catalog

The catalog is generated, not hand-maintained. Reproduce with:

```bash
cd apps/rapid-mac
# 1. Compiler-emitted stringsdata — authoritative, and the only source that
#    spells interpolated keys correctly (`%lld`, `%@`; xcstringstool's own
#    extract mode emits a generic `%arg` that does not match the runtime key).
swift build -Xswiftc -emit-localized-strings \
            -Xswiftc -emit-localized-strings-path -Xswiftc /tmp/loc
# 2. Merge into the catalog, preserving existing translations.
xcrun xcstringstool sync Sources/Rapid/Resources/Localizable.xcstrings \
      --stringsdata /tmp/loc/*.stringsdata
```

Notes that cost time to rediscover:

- Typing a component's copy parameter as `LocalizedStringKey` is what makes the
  compiler *extract* the literal at its call sites. That is why the catalog grew
  from 80 to 808 keys with no call-site churn.
- `sync` marks dynamically-constructed keys `stale` — `speculative_status.*`
  (built from an enum's `.help` suffix) and `image_input.unavailable.*` (built
  from `PhotoHint.rawValue`). They are live and translated. **Do not delete
  them**, and do not "fix" the staleness by hand.
- `xcstringstool sync` rewrites the file; re-serialising it with `json.dumps`
  changes the `"key" : value` spacing and the empty-entry shape. Match Xcode's
  format (`separators=(',', ' : ')`, empty dicts as `{\n\n<indent>}`) or the
  next sync produces a whole-file diff.
- The `""` key (from `Toggle("", isOn:)` with `labelsHidden()`) is deliberately
  left untranslated; a `zh-Hans` block with an empty value fails
  `LocalizationTests`.
