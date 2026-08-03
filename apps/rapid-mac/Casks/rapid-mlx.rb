cask "rapid-mlx" do
  version "0.11.0"
  # TODO(release): the real DMG sha256 is filled in per release by the
  # cask-bump flow in Casks/README.md (the workflow prints the notarised
  # DMG's sha256). The all-zeros value below is an obvious placeholder so
  # an un-bumped cask fails loudly on install rather than silently
  # verifying the wrong bytes.
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

  # The tag prefix is "rapid-mac-v" to match the release workflow
  # (.github/workflows/rapid-mac-release.yml triggers on rapid-mac-v*), and
  # the DMG asset name is "rapid-mlx-desktop.dmg" from scripts/dmg.sh.
  url "https://github.com/raullenchai/Rapid-MLX/releases/download/rapid-mac-v#{version}/rapid-mlx-desktop.dmg",
      verified: "github.com/raullenchai/Rapid-MLX/"
  name "Rapid-MLX"
  desc "Native SwiftUI Mac client for rapid-mlx local inference"
  homepage "https://rapidmlx.com/"

  livecheck do
    url :url
    # The monorepo publishes BOTH engine and app releases; the regex
    # keeps livecheck locked to the app's "rapid-mac-v<semver>" tags so an
    # engine release can't be misread as a desktop version.
    regex(/^rapid-mac[._-]v?(\d+(?:\.\d+)+)$/i)
    strategy :github_latest
  end

  auto_updates true
  depends_on macos: :sonoma

  app "Rapid-MLX Desktop.app"

  zap trash: [
    "~/Library/Application Support/Rapid",
    "~/Library/Caches/com.rapidmlx.rapid",
    "~/Library/Preferences/com.rapidmlx.rapid.plist",
    "~/Library/Saved Application State/com.rapidmlx.rapid.savedState",
  ]
end
