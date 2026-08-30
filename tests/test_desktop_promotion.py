import json
from pathlib import Path

import pytest

from scripts.desktop_promotion import create_manifest, verify_manifest

SHA = "a" * 40
VERSION = "0.13.2"
TAG = f"rapid-mac-v{VERSION}"
REPO = "raullenchai/Rapid-MLX"
WORKFLOW = ".github/workflows/auto-release.yml"


def _bundle(tmp_path: Path, *, version: str = VERSION) -> tuple[Path, Path]:
    tag = f"rapid-mac-v{version}"
    bundle = tmp_path / "candidate"
    sparkle = bundle / "sparkle"
    sparkle.mkdir(parents=True)
    dmg = bundle / "rapid-mlx-desktop.dmg"
    dmg.write_bytes(b"signed notarized dmg")
    zip_path = sparkle / f"Rapid-MLX-Desktop-{version}.zip"
    zip_path.write_bytes(b"signed sparkle zip")
    (bundle / "release-notes.md").write_text(f"## [{version}]\n")
    (bundle / "rapid-mlx-desktop.manifest.json").write_text(
        json.dumps(
            {
                "source_sha": SHA,
                "version": version,
                "app_tag": tag,
                "signed": True,
                "artifacts": [
                    {
                        "filename": dmg.name,
                        "size": dmg.stat().st_size,
                        "sha256": __import__("hashlib")
                        .sha256(dmg.read_bytes())
                        .hexdigest(),
                    }
                ],
            }
        )
    )
    (sparkle / "appcast.xml").write_text(
        f'''<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel><item><sparkle:version>132</sparkle:version><sparkle:shortVersionString>{version}</sparkle:shortVersionString><enclosure url="https://dl.rapidmlx.com/{zip_path.name}" length="{zip_path.stat().st_size}" sparkle:edSignature="signature" /></item></channel></rss>'''
    )
    manifest = bundle / "desktop-promotion-manifest.json"
    manifest.write_text(
        json.dumps(
            create_manifest(
                bundle=bundle,
                repository=REPO,
                workflow=WORKFLOW,
                run_id=123,
                run_attempt=2,
                source_sha=SHA,
                version=version,
                app_tag=tag,
            )
        )
    )
    return bundle, manifest


def _verify(bundle: Path, manifest: Path, *, version: str = VERSION):
    return verify_manifest(
        bundle=bundle,
        manifest_path=manifest,
        repository=REPO,
        workflow=WORKFLOW,
        run_id=123,
        run_attempt=2,
        source_sha=SHA,
        version=version,
        app_tag=f"rapid-mac-v{version}",
    )


def test_exact_bundle_round_trip(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    assert _verify(bundle, manifest)["producer"]["run_attempt"] == 2


def test_rc_item_level_identity_round_trip(tmp_path: Path):
    version = "0.13.2-rc1"
    bundle, manifest = _bundle(tmp_path, version=version)
    assert _verify(bundle, manifest, version=version)["release"]["version"] == version


def test_legacy_enclosure_identity_is_accepted(tmp_path: Path):
    bundle, _ = _bundle(tmp_path)
    appcast = bundle / "sparkle/appcast.xml"
    text = appcast.read_text()
    text = text.replace("<sparkle:version>132</sparkle:version>", "")
    text = text.replace(
        f"<sparkle:shortVersionString>{VERSION}</sparkle:shortVersionString>", ""
    )
    text = text.replace(
        "<enclosure ",
        f'<enclosure sparkle:version="132" sparkle:shortVersionString="{VERSION}" ',
    )
    appcast.write_text(text)

    manifest = create_manifest(
        bundle=bundle,
        repository=REPO,
        workflow=WORKFLOW,
        run_id=123,
        run_attempt=2,
        source_sha=SHA,
        version=VERSION,
        app_tag=TAG,
    )
    assert manifest["release"]["version"] == VERSION


def test_conflicting_item_and_enclosure_identity_is_rejected(tmp_path: Path):
    bundle, _ = _bundle(tmp_path)
    appcast = bundle / "sparkle/appcast.xml"
    appcast.write_text(
        appcast.read_text().replace(
            "<enclosure ",
            '<enclosure sparkle:shortVersionString="0.13.3" ',
        )
    )
    with pytest.raises(ValueError, match="shortVersionString values conflict"):
        create_manifest(
            bundle=bundle,
            repository=REPO,
            workflow=WORKFLOW,
            run_id=123,
            run_attempt=2,
            source_sha=SHA,
            version=VERSION,
            app_tag=TAG,
        )


@pytest.mark.parametrize(
    "relative",
    [
        "rapid-mlx-desktop.dmg",
        "rapid-mlx-desktop.manifest.json",
        "release-notes.md",
        "sparkle/appcast.xml",
        "sparkle/Rapid-MLX-Desktop-0.13.2.zip",
    ],
)
def test_mutated_payload_is_rejected(tmp_path: Path, relative: str):
    bundle, manifest = _bundle(tmp_path)
    with (bundle / relative).open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="does not match|bytes"):
        _verify(bundle, manifest)


def test_wrong_exact_run_is_rejected(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        verify_manifest(
            bundle=bundle,
            manifest_path=manifest,
            repository=REPO,
            workflow=WORKFLOW,
            run_id=124,
            run_attempt=2,
            source_sha=SHA,
            version=VERSION,
            app_tag=TAG,
        )


def test_missing_or_extra_zip_is_rejected(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    (bundle / "sparkle/extra.zip").write_bytes(b"extra")
    with pytest.raises(ValueError, match="exactly one"):
        _verify(bundle, manifest)


def test_unrecorded_payload_is_rejected(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    (bundle / "surprise.txt").write_text("not in the signed roster")
    with pytest.raises(ValueError, match="unrecorded payload"):
        _verify(bundle, manifest)


def test_unsafe_manifest_path_is_rejected(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    data = json.loads(manifest.read_text())
    data["artifacts"][0]["path"] = "../rapid-mlx-desktop.dmg"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unsafe"):
        _verify(bundle, manifest)


def test_symlinked_promotion_manifest_is_rejected(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    outside = tmp_path / "outside.json"
    manifest.rename(outside)
    manifest.symlink_to(outside)
    with pytest.raises(ValueError, match="regular file"):
        _verify(bundle, manifest)


def test_appcast_identity_is_rejected_even_if_manifest_is_rehashed(tmp_path: Path):
    bundle, manifest = _bundle(tmp_path)
    appcast = bundle / "sparkle/appcast.xml"
    appcast.write_text(
        appcast.read_text().replace(
            f"<sparkle:shortVersionString>{VERSION}</sparkle:shortVersionString>",
            "<sparkle:shortVersionString>0.13.3</sparkle:shortVersionString>",
        )
    )
    data = json.loads(manifest.read_text())
    item = next(x for x in data["artifacts"] if x["path"] == "sparkle/appcast.xml")
    item["size"] = appcast.stat().st_size
    item["sha256"] = __import__("hashlib").sha256(appcast.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="short version"):
        _verify(bundle, manifest)
