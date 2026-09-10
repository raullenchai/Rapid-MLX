import Foundation
import Testing
@testable import Rapid

/// Contract cases for ``UpdateChecker``. Migrated from
/// ``TestDriver.runUpdateCheck`` — we inject a stub fetcher so the
/// test runs offline and doesn't depend on whatever version the
/// real worker is reporting at test time.
@MainActor
@Suite("UpdateChecker contract")
struct UpdateCheckerTests {
    /// Shared helper. Keeps each ``@Test`` case to the one shape that
    /// varies (current version + stub-returned version).
    private func release(_ version: String) -> UpdateChecker.Release {
        UpdateChecker.Release(
            schemaVersion: 1,
            version: version,
            tagName: "v\(version)",
            htmlURL: "https://github.com/machinefi/rapid-desktop/releases/tag/v\(version)",
            notes: "test notes",
            publishedAt: "2026-06-09T00:00:00Z",
            dmgURL: "https://example.invalid/Rapid-\(version).dmg"
        )
    }

    @Test("Newer remote version surfaces an availableUpdate")
    func newerRemote() async {
        let release = release("0.3.1")
        let checker = UpdateChecker(currentVersion: "0.3.0") { release }
        _ = await checker.check()
        #expect(checker.availableUpdate?.version == "0.3.1")
    }

    @Test("Same remote version does not surface an update")
    func sameVersion() async {
        let release = release("0.3.1")
        let checker = UpdateChecker(currentVersion: "0.3.1") { release }
        _ = await checker.check()
        #expect(checker.availableUpdate == nil)
    }

    @Test("Older remote version does not offer a downgrade-nag")
    func olderRemote() async {
        let release = release("0.2.9")
        let checker = UpdateChecker(currentVersion: "0.3.1") { release }
        _ = await checker.check()
        #expect(checker.availableUpdate == nil)
    }

    @Test("Transient fetcher failure preserves the previous latest")
    func transientFailurePreservesLatest() async {
        // Helper actor that flips ``fail`` between calls so a single
        // ``UpdateChecker`` can experience a success then a failure.
        actor Flag {
            var fail = false
            func setFail(_ v: Bool) { fail = v }
            func current() -> Bool { fail }
        }
        let flag = Flag()
        let success = release("0.3.2")
        let checker = UpdateChecker(currentVersion: "0.3.0") {
            if await flag.current() {
                throw UpdateError.transport("simulated outage")
            }
            return success
        }
        _ = await checker.check()
        #expect(checker.availableUpdate?.version == "0.3.2")
        // Now simulate the upstream being briefly 5xx.
        await flag.setFail(true)
        _ = await checker.check()
        // Previous ``latest`` must be preserved so the UI doesn't
        // flap between "update available" and "no update" on a blip.
        #expect(checker.lastError != nil)
        #expect(checker.availableUpdate?.version == "0.3.2")
        // Pin LocalizedError conformance — without it the Settings →
        // App panel would render the developer-only Apple default
        // "(Rapid.UpdateError error 1.)" instead of the prose
        // ``UpdateError.description`` declares. Surfaced during the
        // v0.6.7 polish walk.
        #expect(checker.lastError?.contains("transport error") == true)
        #expect(checker.lastError?.contains("simulated outage") == true)
    }

    @Test("Semver: 0.3.10 is newer than 0.3.9 (numeric, not lexicographic)")
    func semverNumeric() {
        #expect(UpdateChecker.isNewer("v0.3.10", than: "v0.3.9"))
        #expect(!UpdateChecker.isNewer("v0.3.9", than: "v0.3.10"))
    }

    @Test("Semver: pre-release suffixes coerce to numeric prefix")
    func semverPrerelease() {
        // "0.3.0-rc1" is treated as "0.3.0" — same numeric prefix,
        // therefore NOT newer than the GA "0.3.0".
        #expect(!UpdateChecker.isNewer("0.3.0-rc1", than: "0.3.0"))
    }

    @Test("Semver: major bumps win over high minor/patch")
    func semverMajorBump() {
        #expect(UpdateChecker.isNewer("1.0.0", than: "0.99.99"))
    }

    // MARK: - Release payload validation (codex audit r1 UpdateChecker.swift:161)

    /// Build a happy-path Release; tests then mutate one field to
    /// prove the validator rejects each invariant violation.
    private func goodRelease() -> UpdateChecker.Release {
        UpdateChecker.Release(
            schemaVersion: 1,
            version: "0.5.13",
            tagName: "v0.5.13",
            htmlURL: "https://github.com/machinefi/rapid-desktop/releases/tag/v0.5.13",
            notes: "ok",
            publishedAt: "2026-06-12T00:00:00Z",
            dmgURL: "https://github.com/machinefi/rapid-desktop/releases/download/v0.5.13/Rapid-0.5.13.dmg"
        )
    }

    @Test("validateReleasePayload accepts the happy-path worker shape")
    func validatorAcceptsGood() throws {
        try UpdateChecker.validateReleasePayload(goodRelease())
    }

    @Test("validateReleasePayload rejects an unexpected schema_version")
    func validatorRejectsSchema() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: 2,
            version: r.version, tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    @Test("validateReleasePayload rejects a non-numeric version string")
    func validatorRejectsVersion() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion,
            version: "latest", tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    // PR #26 codex meta-review finding 6 (P2): the old grammar gate
    // used compactMap which silently dropped non-numeric segments,
    // so a server (or a tampered-with response) could ship "0.5.13.evil"
    // and we would happily display "Update available 0.5.13.evil" in
    // the menu. Each of these cases must now throw.
    @Test("validateReleasePayload rejects partially numeric versions")
    func validatorRejectsPartiallyNumericVersions() {
        let r0 = goodRelease()
        for bad in ["1.evil", "0.5.13.evil", "1..2", "1.2.", ".1.2", "1.2.bad.3"] {
            let r = UpdateChecker.Release(
                schemaVersion: r0.schemaVersion,
                version: bad, tagName: r0.tagName, htmlURL: r0.htmlURL,
                notes: r0.notes, publishedAt: r0.publishedAt, dmgURL: r0.dmgURL
            )
            #expect(throws: UpdateError.self) {
                try UpdateChecker.validateReleasePayload(r)
            }
        }
    }

    @Test("strictNumericParts accepts well-formed semver-ish versions")
    func strictParserAcceptsCleanVersions() {
        #expect(UpdateChecker.strictNumericParts(of: "0.5.13") == [0, 5, 13])
        #expect(UpdateChecker.strictNumericParts(of: "v0.5.13") == [0, 5, 13])
        // Pre-release tag is stripped before grammar check.
        #expect(UpdateChecker.strictNumericParts(of: "0.5.13-beta") == [0, 5, 13])
        #expect(UpdateChecker.strictNumericParts(of: "1") == [1])
    }

    @Test("strictNumericParts rejects malformed versions")
    func strictParserRejectsMalformed() {
        #expect(UpdateChecker.strictNumericParts(of: "1.evil") == nil)
        #expect(UpdateChecker.strictNumericParts(of: "0.5.13.evil") == nil)
        #expect(UpdateChecker.strictNumericParts(of: "1..2") == nil)
        #expect(UpdateChecker.strictNumericParts(of: "1.2.") == nil)
        #expect(UpdateChecker.strictNumericParts(of: ".1.2") == nil)
        #expect(UpdateChecker.strictNumericParts(of: "abc") == nil)
        #expect(UpdateChecker.strictNumericParts(of: "") == nil)
    }

    @Test("validateReleasePayload rejects a release URL on a foreign host")
    func validatorRejectsForeignReleaseHost() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName,
            htmlURL: "https://attacker.example.com/release",
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    @Test("validateReleasePayload rejects a DMG URL on a foreign host")
    func validatorRejectsForeignDMGHost() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt,
            dmgURL: "https://attacker.example.com/Rapid.dmg"
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    @Test("validateReleasePayload rejects HTTP (non-TLS) release URL")
    func validatorRejectsHTTPReleaseURL() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName,
            htmlURL: "http://github.com/machinefi/rapid-desktop/releases/tag/v0.5.13",
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    @Test("validateReleasePayload accepts a release with no DMG URL (open-release-page-only)")
    func validatorAcceptsNilDMG() throws {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: nil
        )
        try UpdateChecker.validateReleasePayload(r)
    }

    // MARK: - README L94-96 audit-batch-2 unpinned surfaces

    /// Pins the ``URL.user == nil`` userinfo guard in isolation.
    /// Codex r1 BLOCKING: the prior fixture used a foreign host as
    /// the URL.host, so the foreign-host allowlist check caught the
    /// regression first and the user guard could be deleted with the
    /// test still failing. The new fixture puts an ALLOWLISTED host
    /// at URL.host (github.com) so the only thing keeping
    /// ``https://attacker@github.com/release`` from passing is the
    /// userinfo guard — drop the line at UpdateChecker.swift:215 and
    /// this test goes red.
    ///
    /// Apple's URL parser maps RFC 3986 §3.2.1 userinfo separately
    /// from host: for ``https://attacker@github.com/release``,
    /// ``URL.user == "attacker"``, ``URL.host == "github.com"``.
    @Test("validateReleasePayload rejects a release URL carrying a userinfo component (allowlisted host)")
    func validatorRejectsUserinfoInReleaseURL() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName,
            // user=attacker, host=github.com — host is allowlisted,
            // so only the URL.user == nil guard at validator L215
            // catches this case.
            htmlURL: "https://attacker@github.com/release",
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    /// Same defence for the DMG URL path with an allowlisted host so
    /// the userinfo guard at UpdateChecker.swift:230 is the
    /// load-bearing check. The download allowlist is broader (adds
    /// githubusercontent CDN hosts) — pick one of those as host so a
    /// regression dropping the user guard would let a userinfo-bearing
    /// CDN-host URL through.
    @Test("validateReleasePayload rejects a DMG URL carrying a userinfo component (allowlisted host)")
    func validatorRejectsUserinfoInDMGURL() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt,
            dmgURL: "https://attacker@objects.githubusercontent.com/Rapid.dmg"
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    /// Pins the **combined** "no userinfo" contract — user AND
    /// password components must both be nil. Codex r1 NIT: dropping
    /// only the password check wouldn't actually leak this fixture
    /// (Apple parses ``x:p@github.com`` as user=x + password=p, so
    /// the user guard would still reject); the value of the test is
    /// that it pins the union contract holds across the
    /// user:password two-field RFC 3986 §3.2.1 shape that production
    /// servers also accept.
    @Test("validateReleasePayload rejects a release URL carrying user:password userinfo (allowlisted host)")
    func validatorRejectsUserPasswordInReleaseURL() {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName,
            // user=x, password=p, host=github.com.
            htmlURL: "https://x:p@github.com/release",
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: r.dmgURL
        )
        #expect(throws: UpdateError.self) {
            try UpdateChecker.validateReleasePayload(r)
        }
    }

    /// Pins case-insensitive host matching. RFC 3986 §3.2.2: host
    /// component is case-insensitive. A worker that ever rendered
    /// the URL with uppercase letters (or an HTTP redirect to one)
    /// must still pass — without lowercasing, the allowlist would
    /// reject legitimate traffic in production and the user sees a
    /// "no update found" toast despite an update being available.
    @Test("validateReleasePayload accepts case-mixed release host (lowercased before allowlist check)")
    func validatorAcceptsUppercaseHost() throws {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName,
            htmlURL: "https://GITHUB.com/machinefi/rapid-desktop/releases/tag/v0.5.13",
            notes: r.notes, publishedAt: r.publishedAt, dmgURL: nil
        )
        try UpdateChecker.validateReleasePayload(r)
    }

    /// Codex r1 NIT 3: also pin case-insensitivity for the dmgURL
    /// path. The validator lowercases both htmlURL and dmgURL hosts;
    /// a regression dropping one .lowercased() call could surface
    /// only on the path the test doesn't exercise.
    @Test("validateReleasePayload accepts case-mixed DMG host (lowercased before allowlist check)")
    func validatorAcceptsUppercaseDMGHost() throws {
        var r = goodRelease()
        r = UpdateChecker.Release(
            schemaVersion: r.schemaVersion, version: r.version,
            tagName: r.tagName, htmlURL: r.htmlURL,
            notes: r.notes, publishedAt: r.publishedAt,
            dmgURL: "https://OBJECTS.GITHUBUSERCONTENT.com/foo/Rapid.dmg"
        )
        try UpdateChecker.validateReleasePayload(r)
    }

    /// Pins the exact membership of ``updateReleaseHostAllowlist``.
    /// A refactor that adds a new entry — even one that looks
    /// innocuous like a typo'd ``"githubusercontent.com"`` — would
    /// silently widen the phishing CTA surface. Pinning the
    /// set-equal contract surfaces the addition as a deliberate
    /// decision instead of a silent drift.
    ///
    /// v0.6.12: `rapidmlx.com` and `www.rapidmlx.com` joined the set
    /// when the updater stopped proxying GitHub Releases and started
    /// reading a static R2 manifest. The manifest's `html_url` now
    /// points at the public landing page (`https://rapidmlx.com/desktop`)
    /// — the rapid-desktop repo is private, so a GitHub release URL
    /// would 404 for most users.
    @Test("updateReleaseHostAllowlist contains exactly the four documented hosts")
    func releaseAllowlistMembershipPinned() {
        #expect(updateReleaseHostAllowlist == [
            "github.com",
            "www.github.com",
            "rapidmlx.com",
            "www.rapidmlx.com",
        ])
    }

    /// Pins the exact membership of ``updateDownloadHostAllowlist``.
    /// The download set is broader than the release set by design
    /// (GH 302s the asset URL to objects.githubusercontent.com).
    /// Adding an arbitrary CDN here would let the in-app installer
    /// land a DMG fetched from a non-GitHub origin.
    ///
    /// v0.5.22 (#164): ``dl.rapidmlx.com`` joined the set as
    /// forward-compat for the R2 mirror.
    ///
    /// v0.6.12: `rapidmlx.com` and `www.rapidmlx.com` joined to
    /// preserve the release⊆download superset invariant — the
    /// manifest's `html_url` (release-notes link) now points at the
    /// landing page. DMGs are still only served from `dl.rapidmlx.com`;
    /// these two entries are bookkeeping, not new download surfaces.
    @Test("updateDownloadHostAllowlist contains exactly the seven documented hosts")
    func downloadAllowlistMembershipPinned() {
        #expect(updateDownloadHostAllowlist == [
            "github.com",
            "www.github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
            "dl.rapidmlx.com",
            "rapidmlx.com",
            "www.rapidmlx.com",
        ])
    }

    /// Pins the invariant that every host on the release allowlist
    /// also appears on the download allowlist. The "Update available"
    /// CTA flow goes release-page → "Install in app" → DMG download;
    /// if a host can be the landing page but not the asset source the
    /// flow breaks with a confusing "release URL is OK but DMG URL
    /// failed" surface. The current downloadAllowlist is a strict
    /// superset of releaseAllowlist by intent — pin it.
    @Test("Every release allowlist host is also on the download allowlist (superset invariant)")
    func releaseAllowlistIsDownloadSubset() {
        for host in updateReleaseHostAllowlist {
            #expect(
                updateDownloadHostAllowlist.contains(host),
                "Release host \(host) is not in download allowlist — Install-in-app flow would break for that host"
            )
        }
    }

    // MARK: - P3 slice δ: optional bootstrapper / quickstart fields

    @Test("Release decodes a payload without sidecar/model fields (pre-slice-γ wire shape — backward compat)")
    func releaseDecodesWithoutBootstrapperFields() throws {
        // Pre-slice-γ ``latest.json`` had only dmg_url / dmg_sha256 /
        // dmg_size on top of the core fields. The Release Codable
        // type MUST keep decoding that shape — older clients in the
        // wild also publish manifests via the same shape during a
        // staged rollout, and Codable optional-field default tolerance
        // is what makes the schema_version=1 contract additive.
        let payload = """
        {
          "schema_version": 1,
          "version": "0.6.11",
          "tag_name": "v0.6.11",
          "html_url": "https://rapidmlx.com/desktop",
          "notes": "test",
          "published_at": "2026-06-16T16:38:34Z",
          "dmg_url": "https://dl.rapidmlx.com/rapid-mlx-desktop-0.6.11.dmg"
        }
        """
        let data = payload.data(using: .utf8)!
        let release = try JSONDecoder().decode(UpdateChecker.Release.self, from: data)
        #expect(release.version == "0.6.11")
        #expect(release.dmgURL?.hasSuffix("0.6.11.dmg") == true)
        // Eight optional fields default to nil. Pin the most important
        // four (the slice δ additions) — without this, a future
        // refactor that promotes one of them to required (or renames
        // a CodingKey) would silently break the pre-slice-γ decode
        // contract.
        #expect(release.modelURL == nil)
        #expect(release.modelSHA256 == nil)
        #expect(release.modelSize == nil)
        #expect(release.modelAlias == nil)
    }

    @Test("Release decodes a payload WITH all four model_* fields (slice δ activation)")
    func releaseDecodesWithModelFields() throws {
        // Once slice δ ships, latest.json carries the four model_*
        // fields. The Release Codable type MUST surface them
        // populated — even though v0.8.x UpdateChecker doesn't take
        // action on them, the forward-compat invariant is what
        // protects a future client migration (Sparkle, bootstrapper-
        // backed in-app reinstall, etc.) from a silent type drift.
        let payload = """
        {
          "schema_version": 1,
          "version": "0.9.0",
          "tag_name": "v0.9.0",
          "html_url": "https://rapidmlx.com/desktop",
          "notes": "test",
          "published_at": "2026-06-25T00:00:00Z",
          "dmg_url": "https://dl.rapidmlx.com/rapid-mlx-desktop-0.9.0.dmg",
          "sidecar_url": "https://dl.rapidmlx.com/rapid-mlx-sidecar-0.9.0.tar.gz",
          "sidecar_sha256": "6e7ad75c945993a23101b2c0ae98a8079f2719ebb54bada8fcd185f0d9fdca12",
          "sidecar_size": 125969942,
          "sidecar_version": "0.8.18",
          "model_url": "https://dl.rapidmlx.com/quickstart-bonsai-1.7b-2bit-0.9.0.tar.gz",
          "model_sha256": "6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a22",
          "model_size": 303104000,
          "model_alias": "bonsai-1.7b-2bit"
        }
        """
        let data = payload.data(using: .utf8)!
        let release = try JSONDecoder().decode(UpdateChecker.Release.self, from: data)
        #expect(release.modelURL == "https://dl.rapidmlx.com/quickstart-bonsai-1.7b-2bit-0.9.0.tar.gz")
        #expect(release.modelSHA256 == "6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a22")
        #expect(release.modelSize == 303104000)
        #expect(release.modelAlias == "bonsai-1.7b-2bit")
        // The bootstrapper-only sidecar fields also surface — same
        // tolerance contract.
        #expect(release.sidecarURL == "https://dl.rapidmlx.com/rapid-mlx-sidecar-0.9.0.tar.gz")
        #expect(release.sidecarVersion == "0.8.18")
        #expect(release.sidecarSize == 125969942)
        // Schema_version stays at 1 — additive optional fields don't
        // bump it. A v0.8.x client decoding this still passes
        // validateReleasePayload (which enforces == 1 exactly).
        #expect(release.schemaVersion == 1)
    }

    @Test("validateReleasePayload still accepts the slice δ payload (additive fields invisible to the gate)")
    func validateReleasePayloadAcceptsSliceDeltaShape() throws {
        // The validate gate only inspects core fields (schema_version /
        // version / html_url / dmg_url). Slice δ's additions live
        // outside that surface — pin that the gate's acceptance is
        // unchanged.
        let release = UpdateChecker.Release(
            schemaVersion: 1,
            version: "0.9.0",
            tagName: "v0.9.0",
            htmlURL: "https://rapidmlx.com/desktop",
            notes: "test",
            publishedAt: "2026-06-25T00:00:00Z",
            dmgURL: "https://dl.rapidmlx.com/rapid-mlx-desktop-0.9.0.dmg",
            sidecarURL: "https://dl.rapidmlx.com/rapid-mlx-sidecar-0.9.0.tar.gz",
            sidecarSHA256: "6e7ad75c945993a23101b2c0ae98a8079f2719ebb54bada8fcd185f0d9fdca12",
            sidecarSize: 125969942,
            sidecarVersion: "0.8.18",
            modelURL: "https://dl.rapidmlx.com/quickstart-bonsai-1.7b-2bit-0.9.0.tar.gz",
            modelSHA256: "6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a2254795d6a22",
            modelSize: 303104000,
            modelAlias: "bonsai-1.7b-2bit"
        )
        do {
            try UpdateChecker.validateReleasePayload(release)
        } catch {
            Issue.record("validateReleasePayload rejected a slice δ shape: \(error). The gate must remain unchanged by additive optional fields.")
        }
    }

    // MARK: - Update-ping endpoint + off switch

    @Test("endpointURL appends the app version as a percent-encoded ?v= query")
    func endpointCarriesVersion() throws {
        let url = try #require(UpdateChecker.endpointURL(forVersion: "0.10.9"))
        #expect(url.absoluteString == "https://rapidmlx.com/api/desktop-update?v=0.10.9")
        // Base is the landing-page worker, not the raw R2 object.
        #expect(UpdateChecker.endpoint == "https://rapidmlx.com/api/desktop-update")
    }

    @Test("endpointURL carries optional os + chip labels when supplied")
    func endpointCarriesOSAndChip() throws {
        let url = try #require(
            UpdateChecker.endpointURL(forVersion: "0.14.0", os: "26.1", chip: "Apple M4 Pro")
        )
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let items = try #require(comps.queryItems)
        #expect(items.count == 3)
        #expect(items.first(where: { $0.name == "v" })?.value == "0.14.0")
        #expect(items.first(where: { $0.name == "os" })?.value == "26.1")
        #expect(items.first(where: { $0.name == "chip" })?.value == "Apple M4 Pro")
    }

    @Test("endpointURL omits a label it could not read")
    func endpointOmitsUnknownLabels() throws {
        // An unreadable value is left off the wire rather than sent as a
        // placeholder, which would mint a bucket meaning "we did not know".
        let url = try #require(UpdateChecker.endpointURL(forVersion: "0.14.0", os: "26.1"))
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let items = try #require(comps.queryItems)
        #expect(items.count == 2)
        #expect(items.contains { $0.name == "os" })
        #expect(!items.contains { $0.name == "chip" })
    }

    @Test("pollOSLabel keeps major.minor and drops the patch level")
    func pollOSLabelIsCoarse() {
        // Dropped at the client, not the endpoint: the poll is on by default,
        // so it should never carry what the server does not keep.
        #expect(UpdateChecker.pollOSLabel(
            OperatingSystemVersion(majorVersion: 26, minorVersion: 1, patchVersion: 3)
        ) == "26.1")
        #expect(UpdateChecker.pollOSLabel(
            OperatingSystemVersion(majorVersion: 15, minorVersion: 0, patchVersion: 0)
        ) == "15.0")
    }

    @Test("pollChipLabel reports Apple silicon only")
    func pollChipLabelRejectsIntel() {
        #expect(UpdateChecker.pollChipLabel("Apple M4 Max") == "Apple M4 Max")
        // Intel brand strings carry the exact SKU and clock — more identifying
        // than the tier this is meant to report.
        #expect(UpdateChecker.pollChipLabel("Intel") == nil)
        #expect(UpdateChecker.pollChipLabel("Intel(R) Core(TM) i9-9880H @ 2.30GHz") == nil)
        #expect(UpdateChecker.pollChipLabel(nil) == nil)
    }

    @Test("endpointURL defensively encodes a malformed version string")
    func endpointEncodesMalformedVersion() throws {
        // A hostile/garbled CFBundleShortVersionString must not be able
        // to inject extra query pairs or break out of the ?v= value.
        let url = try #require(UpdateChecker.endpointURL(forVersion: "1 & evil=x"))
        let comps = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let items = try #require(comps.queryItems)
        #expect(items.count == 1)
        #expect(items.first?.name == "v")
        #expect(items.first?.value == "1 & evil=x")
    }

    @Test("updateChecksEnabled defaults to true with no env override and no stored preference")
    func offSwitchDefaultsOn() {
        #expect(UpdateChecker.updateChecksEnabled(
            environment: [:],
            defaults: freshDefaults()
        ))
    }

    @Test("RAPIDMLX_NO_UPDATE_CHECK truthy value disables the check")
    func offSwitchEnvRapidmlx() {
        #expect(!UpdateChecker.updateChecksEnabled(
            environment: ["RAPIDMLX_NO_UPDATE_CHECK": "1"],
            defaults: freshDefaults()
        ))
    }

    @Test("DO_NOT_TRACK truthy value disables the check")
    func offSwitchEnvDoNotTrack() {
        #expect(!UpdateChecker.updateChecksEnabled(
            environment: ["DO_NOT_TRACK": "1"],
            defaults: freshDefaults()
        ))
    }

    @Test("DO_NOT_TRACK=0 keeps the check enabled (DNT convention)")
    func offSwitchDoNotTrackZeroAllows() {
        #expect(UpdateChecker.updateChecksEnabled(
            environment: ["DO_NOT_TRACK": "0"],
            defaults: freshDefaults()
        ))
    }

    @Test("UserDefaults toggle set to false disables the check")
    func offSwitchUserDefaults() {
        let d = freshDefaults()
        d.set(false, forKey: UpdateChecker.updateCheckEnabledKey)
        #expect(!UpdateChecker.updateChecksEnabled(environment: [:], defaults: d))
    }

    @Test("Off switch skips the fetcher entirely — check() makes no poll")
    func offSwitchSkipsFetch() async {
        // Inject the gate as OFF at the instance level — do NOT mutate
        // process-global ``UserDefaults.standard``, which under Swift
        // Testing's parallel execution races any other test whose
        // ``check()`` reads the same key.
        actor Counter { var n = 0; func bump() { n += 1 }; func count() -> Int { n } }
        let calls = Counter()
        // Both closures passed by explicit label — no trailing-closure
        // ambiguity between ``fetcher`` and ``checksEnabled``.
        let checker = UpdateChecker(
            currentVersion: "0.3.0",
            fetcher: {
                await calls.bump()
                return UpdateChecker.Release(
                    schemaVersion: 1, version: "0.3.1", tagName: "v0.3.1",
                    htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "",
                    dmgURL: nil
                )
            },
            checksEnabled: { false }
        )
        let result = await checker.check()
        // No fetch happened, and no update surfaced.
        #expect(await calls.count() == 0)
        #expect(result == nil)
        #expect(checker.availableUpdate == nil)
    }

    // MARK: - Worker-primary → R2 fallback

    /// Records the URLs a fetch primitive was asked for, in order.
    /// Actor so the ``@Sendable`` fetch closure can mutate it safely.
    private actor FetchSpy {
        private(set) var urls: [URL] = []
        func record(_ u: URL) { urls.append(u) }
        func requested() -> [URL] { urls }
    }

    @Test("Worker failure falls back to the R2 manifest (updates still surface)")
    func fallbackToR2OnWorkerFailure() async throws {
        let spy = FetchSpy()
        let workerURL = try #require(UpdateChecker.endpointURL(forVersion: "0.10.9"))
        let r2URL = try #require(URL(string: UpdateChecker.fallbackEndpoint))
        let release = try await UpdateChecker.fetchWithFallback(
            primary: workerURL,
            fallback: r2URL
        ) { url in
            await spy.record(url)
            // The Worker is down; R2 still serves a valid manifest.
            if url == workerURL { throw UpdateError.httpStatus(503) }
            return UpdateChecker.Release(
                schemaVersion: 1, version: "9.9.9", tagName: "v9.9.9",
                htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "",
                dmgURL: nil
            )
        }
        // Worker tried first, then R2 — and the R2 manifest is surfaced.
        #expect(await spy.requested() == [workerURL, r2URL])
        #expect(release.version == "9.9.9")
    }

    @Test("Worker success never touches the R2 fallback")
    func noFallbackWhenWorkerSucceeds() async throws {
        let spy = FetchSpy()
        let workerURL = try #require(UpdateChecker.endpointURL(forVersion: "0.10.9"))
        let r2URL = try #require(URL(string: UpdateChecker.fallbackEndpoint))
        let release = try await UpdateChecker.fetchWithFallback(
            primary: workerURL,
            fallback: r2URL
        ) { url in
            await spy.record(url)
            return UpdateChecker.Release(
                schemaVersion: 1, version: "1.0.0", tagName: "v1.0.0",
                htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "",
                dmgURL: nil
            )
        }
        #expect(await spy.requested() == [workerURL])   // R2 never hit
        #expect(release.version == "1.0.0")
    }

    @Test("Cancellation is rethrown, not retried against R2")
    func cancellationNotRetried() async {
        let spy = FetchSpy()
        let workerURL = UpdateChecker.endpointURL(forVersion: "0.10.9")!
        let r2URL = URL(string: UpdateChecker.fallbackEndpoint)!
        do {
            _ = try await UpdateChecker.fetchWithFallback(
                primary: workerURL,
                fallback: r2URL
            ) { url in
                await spy.record(url)
                throw CancellationError()
            }
            Issue.record("expected cancellation to propagate")
        } catch is CancellationError {
            // expected — scene teardown must preserve state, not double-fetch.
        } catch {
            Issue.record("expected CancellationError, got \(error)")
        }
        #expect(await spy.requested() == [workerURL])   // R2 NOT tried
    }

    @Test("A nil primary URL throws invalidURL without any fetch")
    func nilPrimaryThrowsInvalidURL() async {
        let spy = FetchSpy()
        do {
            _ = try await UpdateChecker.fetchWithFallback(
                primary: nil,
                fallback: URL(string: UpdateChecker.fallbackEndpoint)
            ) { url in
                await spy.record(url)
                return UpdateChecker.Release(
                    schemaVersion: 1, version: "1.0.0", tagName: "v1.0.0",
                    htmlURL: "https://rapidmlx.com/desktop", notes: "", publishedAt: "",
                    dmgURL: nil
                )
            }
            Issue.record("expected invalidURL")
        } catch UpdateError.invalidURL {
            // expected
        } catch {
            Issue.record("expected UpdateError.invalidURL, got \(error)")
        }
        #expect(await spy.requested().isEmpty)
    }

    /// A throwaway ``UserDefaults`` suite so toggle tests never touch
    /// (or leak into) the real ``standard`` domain.
    private func freshDefaults() -> UserDefaults {
        let suite = "test.updatechecker.\(UUID().uuidString)"
        return UserDefaults(suiteName: suite)!
    }
}
