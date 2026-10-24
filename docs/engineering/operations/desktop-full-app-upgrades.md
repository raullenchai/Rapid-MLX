# Full-app upgrade packaging

Desktop releases must include their bundled engine. Signed and official builds
now reject `SKIP_SIDECAR=1` before compilation or staging. This prevents a release
configuration from intentionally omitting the replacement engine needed by an
older slim installation. The normal full-app build remains the default.

Use the normal whole-app update or install the full Desktop app from the release
download. Replacing the app supplies its bundled runtime without deleting model
caches, preferences, or the old Application Support runtime directory. A failed
update retains the existing app through the updater's replacement/rollback
behavior. This packaging guard does not change runtime selection or add a new
installer.

Local ad-hoc builds may still use `SKIP_SIDECAR=1`. Developers can explicitly
select their local engine with `RAPID_BIN` in the app's launch environment; this
is not a release packaging option.

`DesktopFullAppReleaseTests` executes the real assembly script and verifies that
both official-release and signing configurations refuse an omitted engine
before building. Existing sidecar assembly tests cover the normal bundled path.
