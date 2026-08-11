from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "apps" / "rapid-mac"


def test_xcui_target_is_checked_in_and_runs_in_gui_ci():
    project = MAC / "Tests/RapidUITests/RapidUITests.xcodeproj/project.pbxproj"
    source = MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    workflow = (ROOT / ".github/workflows/rapid-mac-ci.yml").read_text()

    assert project.is_file()
    assert "com.apple.product-type.bundle.ui-testing" in project.read_text()
    assert source.is_file()
    assert "./scripts/run-xcui-tests.sh" in workflow
    assert "RapidUITests.xcresult" in workflow


def test_pixel_assertion_uses_element_screenshots_and_crops_chrome():
    source = (
        MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    ).read_text()

    assert 'element("Images.Gallery.Thumb.1", in: app)' in source
    assert 'element("Images.Gallery.Thumb.2", in: app)' in source
    assert "newest.screenshot()" in source
    assert "older.screenshot()" in source
    assert 'element("Images.Result.Save", in: app)' in source
    assert "waitForNonExistence" in source
    assert "XCTUnwrap" in source
    assert "XCTSkip" not in source
    assert "source.cropping(to: rect)" in source
    assert "XCTAssertGreaterThan(" in source


def test_xcui_runner_launches_production_bundle_with_fake_sidecar():
    runner = (MAC / "scripts/run-xcui-tests.sh").read_text()
    source = (
        MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    ).read_text()

    assert "build/Rapid-MLX Desktop.app" in runner
    assert "lsregister" in runner
    assert 'XCUIApplication(bundleIdentifier: "com.rapidmlx.rapid")' in source
    assert source.count('"CFFIXED_USER_HOME": testHome.path') == 2
    assert '"RAPID_BIN"' in source
    assert "fake-rapid-mlx.sh" in source
    assert "isExecutableFile" in source
    assert "RapidUITests-$(date +%s)-$$.xcresult" in runner


def test_swift_source_parent_traversal_resolves_rapid_mac_fixture():
    source = MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"

    # Swift starts with the file URL and deletes four path components:
    # Tests, RapidUITests, Tests, then the filename's rapid-mac root.
    resolved = source
    for _ in range(4):
        resolved = resolved.parent

    assert resolved == MAC
    assert (resolved / "scripts/fake-rapid-mlx.sh").is_file()
    assert not (resolved.parent / "scripts/fake-rapid-mlx.sh").exists()
