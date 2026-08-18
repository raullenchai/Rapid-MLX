from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "apps" / "rapid-mac"


def test_xcui_target_is_checked_in_and_runs_in_gui_ci():
    project = MAC / "Tests/RapidUITests/RapidUITests.xcodeproj/project.pbxproj"
    source = MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    workflow_path = ROOT / ".github/workflows/rapid-mac-ci.yml"
    workflow = workflow_path.read_text()
    gui_steps = yaml.safe_load(workflow)["jobs"]["gui-golden-flows"]["steps"]
    named_steps = {
        step.get("name"): (index, step) for index, step in enumerate(gui_steps)
    }

    assert project.is_file()
    assert "com.apple.product-type.bundle.ui-testing" in project.read_text()
    assert source.is_file()
    assert "./scripts/run-xcui-tests.sh" in workflow
    assert "RapidUITests.xcresult" in workflow
    xcui_index, xcui = named_steps["XCUITest: image-generation pixels"]
    upload_index, upload = named_steps["Upload XCUITest evidence"]
    verdict_index, verdict = named_steps["Require native XCUITest"]
    assert xcui["id"] == "xcui"
    assert xcui["continue-on-error"] is True
    assert upload["if"] == "steps.xcui.outcome == 'failure'"
    assert verdict["if"] == "always()"
    assert "steps.xcui.outcome" in verdict["env"]["XCUI_OUTCOME"]
    assert xcui_index < upload_index < verdict_index


def test_pixel_assertion_uses_element_screenshots_and_crops_chrome():
    source = (
        MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    ).read_text()

    assert 'element("Images.Gallery.Thumb.1", in: app)' in source
    assert 'element("Images.Gallery.Thumb.2", in: app)' in source
    assert "newest.screenshot()" in source
    assert "older.screenshot()" in source
    assert 'element("Images.ModelPicker", in: app)' in source
    assert 'picker.label.contains("fake-image-alias")' in source
    assert 'events.contains(#""event": "server_started""#)' in source
    assert 'events.contains(#""alias": "fake-image-alias""#)' in source
    assert "imageResponseCount(in: eventLog) == 1" in source
    assert "imageResponseCount(in: eventLog) == 2" in source
    assert "waitForNonExistence" not in source
    assert "XCTUnwrap" in source
    assert "XCTSkip" not in source
    assert "source.cropping(to: rect)" in source
    assert "centerRGBSamples" in source
    assert "meanSquaredDistance.squareRoot()" in source
    assert "XCTAssertGreaterThan(" in source


def test_xcui_runner_launches_production_bundle_with_fake_sidecar():
    runner = (MAC / "scripts/run-xcui-tests.sh").read_text()
    source = (
        MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    ).read_text()

    assert "build/Rapid-MLX Desktop.app" in runner
    assert "lsregister" in runner
    assert "xcodebuild -version" in runner
    assert "XCUIApplication(url: appURL)" in source
    assert 'appendingPathComponent("build/Rapid-MLX Desktop.app")' in source
    assert source.count('"CFFIXED_USER_HOME": testHome.path') == 1
    assert '"RAPID_BIN"' in source
    assert "fake-rapid-mlx.sh" in source
    assert 'appendingPathComponent(".rapid-golden-fake.json")' in source
    assert '"FAKE_EVENT_LOG": eventLog.path' in source
    assert '"RAPID_DESKTOP_PORT": "65000"' in source
    assert '"RAPID_DESKTOP_NO_PORT_SWEEP": "1"' in source
    assert (
        'terminateFakeSidecars(recordedIn: eventLog, alias: "fake-image-alias")'
        in source
    )
    assert "isExecutableFile" in source
    assert "RapidUITests-$(date +%s)-$$.xcresult" in runner


def test_swift_source_parent_traversal_resolves_rapid_mac_fixture():
    source = MAC / "Tests/RapidUITests/Tests/ImageGenerationPixelTests.swift"
    source_text = source.read_text()
    traversal_expression = source_text.split(
        "let rapidMacRoot = URL(fileURLWithPath: #filePath)", 1
    )[1].split("let fakeSidecar", 1)[0]
    traversal_count = traversal_expression.count(".deletingLastPathComponent()")

    # Replay the traversal count from the actual Swift expression, starting
    # with the file itself just as URL(fileURLWithPath: #filePath) does.
    resolved = source
    for _ in range(traversal_count):
        resolved = resolved.parent

    assert resolved == MAC
    assert (resolved / "scripts/fake-rapid-mlx.sh").is_file()
    assert not (resolved.parent / "scripts/fake-rapid-mlx.sh").exists()
