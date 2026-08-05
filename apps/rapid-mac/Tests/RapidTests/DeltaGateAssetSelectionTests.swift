import Foundation
import Testing

/// Pin the asset-selection grammar of release.yml's "Bundle size
/// delta gate" so a future regression to the old endswith-only
/// filter can't reach CI again.
///
/// Background: from v0.8.6 onward, GH Releases carry TWO ``.dmg``
/// assets — the canonical full DMG (``rapid-mlx-desktop.dmg``,
/// ~157 MB) and the slim bootstrapper DMG (``rapid-mlx-desktop-
/// bootstrapper-X.Y.Z.dmg``, ~5.6 MB) added by slice ε.1 as a
/// preview asset. The old gate logic
/// ``select(.name | endswith(".dmg")) | head -1`` picked whichever
/// asset the GH API returned first, which on v0.8.6 happened to be
/// the slim bootstrapper DMG — making v0.8.7's full DMG (166 MB)
/// look like it grew by +160 MB and failing the +50 MB delta cap.
/// The fix pins the filter to exact name ``rapid-mlx-desktop.dmg``
/// so any future preview / variant DMG cannot be picked by mistake.
@Suite("Bundle size delta gate asset selection — v0.8.7 release-fail regression pin")
struct DeltaGateAssetSelectionTests {

    private static var releaseYamlPath: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent(".github")
            .appendingPathComponent("workflows")
            .appendingPathComponent("release.yml")
    }

    private static func loadYaml() throws -> String {
        try String(contentsOf: Self.releaseYamlPath, encoding: .utf8)
    }

}
