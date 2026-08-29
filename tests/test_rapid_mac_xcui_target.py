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
    assert (MAC / "Tests/RapidUITests/Tests/ChatAttachmentJourneyTests.swift").is_file()
    assert "./scripts/run-xcui-tests.sh" in workflow
    assert "RapidImageUITests.xcresult" in workflow
    assert "RapidChatAttachmentUITests.xcresult" in workflow
    xcui_index, xcui = named_steps["XCUITest: image-generation pixels"]
    upload_index, upload = named_steps["Upload XCUITest evidence"]
    verdict_index, verdict = named_steps["Require native XCUITest"]
    assert xcui["id"] == "xcui"
    assert "image-generation" in xcui["if"]
    assert xcui["continue-on-error"] is True
    chat_xcui_index, chat_xcui = named_steps["XCUITest: chat attachment ownership"]
    assert chat_xcui["id"] == "chat_xcui"
    assert "chat-multimodal-attachments" in chat_xcui["if"]
    assert "chat-document-attachment" in chat_xcui["if"]
    assert chat_xcui["continue-on-error"] is True
    assert upload["if"] == (
        "steps.xcui.outcome == 'failure' || steps.chat_xcui.outcome == 'failure'"
    )
    assert verdict["if"] == "always()"
    assert "steps.xcui.outcome" in verdict["env"]["XCUI_OUTCOME"]
    assert "steps.chat_xcui.outcome" in verdict["env"]["CHAT_XCUI_OUTCOME"]
    assert "image-generation" in verdict["env"]["IMAGE_SELECTED"]
    assert "IMAGE_SELECTED" in verdict["run"]
    assert "CHAT_SELECTED" in verdict["run"]
    assert xcui_index < chat_xcui_index < upload_index < verdict_index


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
    harness = (MAC / "Tests/RapidUITests/Tests/RapidUITestHarness.swift").read_text()
    chat_source = (
        MAC / "Tests/RapidUITests/Tests/ChatAttachmentJourneyTests.swift"
    ).read_text()

    assert "build/Rapid-MLX Desktop.app" in runner
    assert "lsregister" in runner
    assert "xcodebuild -version" in runner
    assert "CODE_SIGN_STYLE=Manual" in runner
    assert "CODE_SIGNING_ALLOWED=YES" in runner
    assert "CODE_SIGNING_REQUIRED=YES" in runner
    assert "CODE_SIGN_IDENTITY=-" in runner
    assert "CODE_SIGNING_ALLOWED=NO" not in runner
    assert "XCUIApplication(url: appURL)" in source
    assert 'appendingPathComponent("build/Rapid-MLX Desktop.app")' in source
    assert source.count('"CFFIXED_USER_HOME": testHome.path') == 1
    assert '"RAPID_BIN"' in source
    assert "fake-rapid-mlx.sh" in source
    assert 'appendingPathComponent(".rapid-golden-fake.json")' in source
    assert '"FAKE_EVENT_LOG": eventLog.path' in source
    assert 'config["FAKE_PID_FILE"] = sidecarPIDFile.path' in harness
    assert "String(contentsOf: sidecarPIDFile" in harness
    assert 'element("MemoryWarning.Confirm")' in harness
    assert "let priorServerStartCount = serverStartCount()" in harness
    assert "self.serverStartCount() > priorServerStartCount" in harness
    assert (
        "func waitForConversationPersistence(containing markers: [String])" in harness
    )
    assert (
        'app.launchEnvironment["RAPID_DESKTOP_PORT"] = String(reservedPort.port)'
        in harness
    )
    assert 'app.dialogs["open-panel"].buttons["OKButton"]' in harness
    assert (
        'XCUIApplication(bundleIdentifier: "com.rapidmlx.rapid-uitest-host")' in harness
    )
    assert 'matching(identifier: "RapidUITests.FileDragSource")' in harness
    assert 'let dropTarget = element("rapid.chat.compose")' in harness
    assert "click(forDuration: 1, thenDragTo: dropTarget)" in harness
    assert "func testDragPasteAndRemovalPreserveWireIdentity()" in chat_source
    assert "NSImage(data: data)" in harness
    assert "pasteboard.writeObjects([image])" in harness
    assert "XCTAssertNotNil(NSImage(pasteboard: pasteboard))" in harness
    assert 'composer.typeKey("v", modifierFlags: .command)' in harness
    assert "reserveLoopbackPort" in harness
    assert "reservationTransferred" in harness
    assert "Darwin.close(reservedPort.descriptor)" in harness
    assert "releasePortReservation()" in harness
    assert "restorePasteboardIfOwned" in harness
    assert "pasteboard.changeCount == ownedPasteboardChangeCount" in harness
    assert "originalPasteboardItems == nil || !stillOwnsPasteboard" in harness
    assert "func relaunch()" in harness
    relaunch_index = chat_source.index("harness.relaunch()")
    restart_index = chat_source.index("harness.startModel()", relaunch_index)
    assert relaunch_index < restart_index
    assert "func staticText(valuePrefix prefix: String)" in harness
    assert "app.staticTexts.matching(" in harness
    assert 'NSPredicate(format: "value BEGINSWITH %@", prefix)' in harness
    assert "terminateFakeSidecars()" in harness
    assert 'messageAction("Retry")' in harness
    assert "testDragPasteAndRemovalPreserveWireIdentity" in chat_source
    assert "testRetryAndRelaunchPreserveSentAttachmentIdentity" in chat_source
    assert "assertCombinedIdentity" in chat_source
    assert 'element(label: "Persist both attachments")' in chat_source
    assert 'element("Sidebar.NewChat").click()' in chat_source
    assert 'element("ChatView.Attachment.Remove.Pasted image.png")' in chat_source
    assert "port: 65_001" not in chat_source
    assert "port: 65_002" not in chat_source
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
