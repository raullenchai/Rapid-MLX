import SwiftUI
import AppKit

/// DEV-ONLY visual snapshot harness.
///
/// When the `RAPID_DEV_SNAPSHOT_DIR` environment variable is set, this
/// renders the real SwiftUI screens to PNG via `ImageRenderer` and then
/// quits. It needs **no Screen-Recording permission** and works over
/// SSH / headless — `ImageRenderer` rasterises the actual view hierarchy
/// in-process, so it is the reliable way to eyeball the UI when
/// `screencapture` can only see the wallpaper.
///
/// Entirely gated on the env var: absent it, `runIfRequested` returns
/// immediately and nothing here runs in normal use. No product behaviour
/// change, no version bump.
enum DevSnapshot {
    @MainActor
    static func runIfRequested(
        server: ServerManager,
        downloads: DownloadManager,
        chat: ChatViewModel,
        updater: UpdateChecker,
        sampling: SamplingConfig,
        appearance: AppearanceConfig,
        settingsRouter: SettingsRouter,
        installTracker: InstallTracker,
        quickstart: QuickstartCoordinator,
        dockPromptStore: DockVisibilityPromptStore
    ) async {
        guard let dir = ProcessInfo.processInfo.environment["RAPID_DEV_SNAPSHOT_DIR"],
              !dir.isEmpty else { return }

        // Let @State init, first layout, and any cheap sync work settle.
        try? await Task.sleep(nanoseconds: 1_400_000_000)
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)

        // Erase to AnyView so the long environment chain stays cheap to
        // type-check and the render call is monomorphic.
        func contentView(width: CGFloat, height: CGFloat) -> AnyView {
            AnyView(
                ContentView()
                    .tint(RapidTheme.brandAmber)
                    .environment(server)
                    .environment(downloads)
                    .environment(chat)
                    .environment(updater)
                    .environment(sampling)
                    .environment(appearance)
                    .environment(settingsRouter)
                    .environment(installTracker)
                    .environment(quickstart)
                    .environment(dockPromptStore)
                    .frame(width: width, height: height)
            )
        }

        /// The Launch page inside the real split-view chrome, so the
        /// captured frame shows what the user actually sees (sidebar +
        /// page) rather than the page in isolation.
        ///
        /// ``ContentView`` owns its ``SidebarSection`` in private
        /// ``@State``, so the harness cannot drive it to ``.launch``
        /// from outside. Re-composing the same two views here is the
        /// only way to capture that route; the scaffold deliberately
        /// mirrors ``ContentView``'s ``NavigationSplitView`` shape so
        /// the screenshot stays representative.
        /// An `HStack`, deliberately, NOT a ``NavigationSplitView``.
        ///
        /// A hosted ``NavigationSplitView`` renders its DETAIL pane
        /// correctly offscreen but leaves the SIDEBAR column blank —
        /// AppKit's split-view controller wants a real on-screen window
        /// to populate it. Since the point of this scene is to review
        /// the rail's width and density against the detail pane, the
        /// scaffold reproduces the split geometry manually so both
        /// columns actually appear.
        ///
        /// Consequence to keep in mind when reading the image: the
        /// system's sidebar toolbar/collapse chrome is absent, and the
        /// divider is drawn here rather than by AppKit.
        /// ``readiness`` is threaded through so the capture exercises the
        /// SHARED value the real ``ContentView`` supplies, not
        /// ``ConnectToolsView``'s nil-fallback sentence. Without it this
        /// scene could not show that Chat and Launch render the same
        /// banner, with the same words and the same action, for the same
        /// state — which is the whole point of the readiness work.
        func launchView(
            width: CGFloat,
            height: CGFloat,
            readiness: ModelReadiness? = .needsStart(alias: "bonsai-1.7b-2bit")
        ) -> AnyView {
            AnyView(
                HStack(spacing: 0) {
                    SidebarView(
                        selection: .constant(.launch),
                        chat: chat,
                        onNewChat: {},
                        onSelectConversation: { _ in }
                    )
                    .frame(width: SidebarView.columnIdealWidth)
                    .background(RapidTheme.surfaceSidebar)

                    Rectangle()
                        .fill(RapidTheme.hairline)
                        .frame(width: 1)

                    LaunchView(
                        server: server,
                        alias: "bonsai-1.7b-2bit",
                        readiness: readiness,
                        onReadinessAction: { _ in }
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(RapidTheme.surfaceCanvas)
                }
                .tint(RapidTheme.brandAmber)
                .environment(server)
                .environment(downloads)
                .environment(chat)
                .environment(updater)
                .environment(sampling)
                .environment(appearance)
                .environment(settingsRouter)
                .environment(installTracker)
                .environment(quickstart)
                .environment(dockPromptStore)
                .frame(width: width, height: height)
            )
        }

        // Scenario 1: the app as launched (idle / first-run, depending on
        // whether HF_HUB_CACHE points at a populated cache).
        render(contentView(width: 900, height: 640), to: "\(dir)/content-idle.png")
        render(contentView(width: 640, height: 560), to: "\(dir)/content-min.png")

        // Scenario 1b (v1.0 visual foundation): the Light/Dark × surface
        // matrix the Phase-1 review runs on. Chat and Launch are the two
        // surfaces this phase repaints, so both are captured at the
        // 900x640 review size in both appearances, plus one shot each at
        // the 640x560 window floor to prove the layout survives it.
        let reviewSize = CGSize(width: 900, height: 640)
        let floorSize = CGSize(width: 640, height: 560)

        renderHosted(contentView(width: 900, height: 640), size: reviewSize,
                     appearance: .aqua, to: "\(dir)/chat-900x640-light.png")
        renderHosted(contentView(width: 900, height: 640), size: reviewSize,
                     appearance: .darkAqua, to: "\(dir)/chat-900x640-dark.png")
        renderHosted(contentView(width: 640, height: 560), size: floorSize,
                     appearance: .aqua, to: "\(dir)/chat-640x560-light.png")
        renderHosted(contentView(width: 640, height: 560), size: floorSize,
                     appearance: .darkAqua, to: "\(dir)/chat-640x560-dark.png")
        renderHosted(launchView(width: 900, height: 640), size: reviewSize,
                     appearance: .aqua, to: "\(dir)/launch-900x640-light.png")
        renderHosted(launchView(width: 900, height: 640), size: reviewSize,
                     appearance: .darkAqua, to: "\(dir)/launch-900x640-dark.png")
        renderHosted(launchView(width: 640, height: 560), size: floorSize,
                     appearance: .aqua, to: "\(dir)/launch-640x560-light.png")
        renderHosted(launchView(width: 640, height: 560), size: floorSize,
                     appearance: .darkAqua, to: "\(dir)/launch-640x560-dark.png")

        // The readiness matrix: every ``ModelReadiness`` case rendered as
        // the user sees it, with the three copy channels that must agree
        // printed underneath.
        //
        // A live ``ContentView`` capture can only ever show whichever
        // state the harness happens to be in (``noModel``, with no
        // catalog and autostart off). The lifecycle states that matter
        // most for review — mid-download, starting, failed — need a real
        // server doing real work, which a snapshot run cannot stage. This
        // renders the same view the composer renders, driven directly by
        // the state values, so the banner / action / placeholder /
        // tooltip / send-enabled contract is reviewable in one image.
        func readinessMatrix() -> AnyView {
            let states: [ModelReadiness] = [
                .noModel,
                .needsDownload(alias: "qwen3.5-9b-4bit", sizeText: "5.0 GB"),
                .needsStart(alias: "bonsai-1.7b-2bit"),
                .downloading(
                    alias: "qwen3.5-9b-4bit",
                    detail: "1.2 GB / 5.0 GB · 24% · 8.4 MB/s · 7 min left",
                    fraction: 0.24
                ),
                .starting(alias: "bonsai-1.7b-2bit", detail: "Loading the model into memory…"),
                .failed(
                    alias: "qwen3.5-9b-4bit",
                    message: FailureDiagnoser.diagnosis(for: .modelLoadFailed).message,
                    action: .retry(alias: "qwen3.5-9b-4bit")
                ),
                .engineMissing,
                .ready(alias: "bonsai-1.7b-2bit"),
            ]
            return AnyView(
                VStack(alignment: .leading, spacing: RapidTheme.Space.lg) {
                    ForEach(Array(states.enumerated()), id: \.offset) { _, state in
                        VStack(alignment: .leading, spacing: RapidTheme.Space.xs) {
                            ReadinessBanner(readiness: state, onAction: { _ in })
                            Text(
                                "send=\(state.sendAllowed ? "ENABLED" : "disabled")"
                                + "  ·  placeholder: “\(state.composerPlaceholder)”"
                                + "  ·  tooltip: “\(state.sendTooltip)”"
                            )
                            .font(RapidFont.code)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(RapidTheme.Space.xl)
                .frame(width: 900, alignment: .leading)
                .background(RapidTheme.surfaceCanvas)
                .tint(RapidTheme.brandAmber)
            )
        }
        let matrixSize = CGSize(width: 900, height: 900)
        renderHosted(readinessMatrix(), size: matrixSize,
                     appearance: .aqua, to: "\(dir)/readiness-matrix-light.png")
        renderHosted(readinessMatrix(), size: matrixSize,
                     appearance: .darkAqua, to: "\(dir)/readiness-matrix-dark.png")

        // The rail on its own, at its shipping width, with seeded
        // history so row density / truncation / the amber selected
        // state are all reviewable. Needed because the hosted
        // ``NavigationSplitView`` captures above render the sidebar
        // column blank.
        func sidebarOnly() -> AnyView {
            AnyView(
                SidebarView(
                    selection: .constant(.launch),
                    chat: chat,
                    onNewChat: {},
                    onSelectConversation: { _ in }
                )
                .frame(width: SidebarView.columnIdealWidth, height: 640)
                .background(RapidTheme.surfaceSidebar)
                .tint(RapidTheme.brandAmber)
            )
        }
        let sidebarSize = CGSize(width: SidebarView.columnIdealWidth, height: 640)
        renderHosted(sidebarOnly(), size: sidebarSize,
                     appearance: .aqua, to: "\(dir)/sidebar-light.png")
        renderHosted(sidebarOnly(), size: sidebarSize,
                     appearance: .darkAqua, to: "\(dir)/sidebar-dark.png")

        // Scenario 2: a populated chat transcript, so we can eyeball the
        // streaming bubble / markdown render path that an empty transcript
        // never exercises.
        chat.devSeedMessages([
            ChatMessage(role: .user, content: "What can you help me with?"),
            ChatMessage(
                role: .assistant,
                content: """
                I run entirely on your Mac — no data leaves the machine. \
                I can answer questions, help with **code**, and explain \
                things. Here's a quick example:

                ```swift
                let greeting = "Hello from Rapid-MLX"
                print(greeting)
                ```

                Ask me anything.
                """,
                status: .complete,
                stats: MessageStats(
                    elapsedSeconds: 0.69,
                    charCount: 232,
                    promptTokens: 12,
                    completionTokens: 58
                )
            ),
            // A failed turn, so the transcript scene actually exercises
            // the error branch of ``MessageRow``. Without it the failure
            // caption's colour had no render path at all and could only
            // be reviewed by reading the source.
            ChatMessage(
                role: .assistant,
                content: "",
                status: .failed,
                errorMessage: "The model couldn't complete that request."
            ),
        ])
        // Let the transcript layout settle before capturing.
        try? await Task.sleep(nanoseconds: 500_000_000)
        render(contentView(width: 900, height: 640), to: "\(dir)/content-chat.png")

        // Chat transcript bubbles, rendered without the ScrollView so the
        // seeded messages are actually visible.
        render(
            AnyView(
                ChatView(viewModel: chat, server: server,
                         alias: .constant("bonsai-1.7b-2bit"),
                         readiness: .ready(alias: "bonsai-1.7b-2bit"))
                    .transcriptRows
                    .frame(width: 900)
                    .background(RapidTheme.canvas)
                    .tint(RapidTheme.brand)
            ),
            to: "\(dir)/chat-bubbles.png"
        )

        // Scenario 3: the "Connect your tools" sheet (pure SwiftUI, so it
        // renders faithfully — unlike the NSViewRepresentable composer).
        render(
            AnyView(
                ConnectToolsView(
                    host: "127.0.0.1", port: 8000,
                    bearer: "rapid-sk-demo1234567890abcdef",
                    alias: "bonsai-1.7b-2bit", onClose: {}
                ).cardContent
                    .frame(width: 460)
                    .background(RapidTheme.canvas)
                    .tint(RapidTheme.brand)
            ),
            to: "\(dir)/connect-tools.png"
        )

        // Scenario 4: the "Speed on this Mac" benchmark card, result state.
        let benchRunner = BenchmarkRunner()
        benchRunner.devSeed(phase: .done(BenchmarkResult(
            alias: "bonsai-1.7b-2bit", chip: "Apple M3 Ultra",
            throughputTPS: 836, tokensPerSecond: 781
        )))
        render(
            AnyView(
                BenchmarkView(
                    binary: nil, alias: "bonsai-1.7b-2bit",
                    hardware: MacHardware.detect(), onClose: {}, runner: benchRunner
                ).content
                    .frame(width: 440)
                    .padding(20)
                    .background(RapidTheme.canvas)
                    .tint(RapidTheme.brand)
            ),
            to: "\(dir)/benchmark-result.png"
        )

        // Scenario 5: the telemetry consent sheet (the privacy "gate").
        render(
            AnyView(
                TelemetryConsentView(onDecision: { _ in })
                    .frame(width: 460)
                    .background(RapidTheme.canvas)
                    .tint(RapidTheme.brand)
            ),
            to: "\(dir)/consent.png"
        )

        log("wrote PNGs to \(dir)")

        // LIVE mode: when RAPID_DEV_SERVE_ALIAS is set, actually start the
        // sidecar (resolved by ServerLocator — the bundled engine unless
        // RAPID_BIN overrides it), send one chat turn, and snapshot the
        // REAL streamed output — the runtime path static renders can't
        // reach. Then the normal terminate exercises clean teardown.
        if let liveAlias = ProcessInfo.processInfo.environment["RAPID_DEV_SERVE_ALIAS"],
           !liveAlias.isEmpty {
            await runLiveChat(
                alias: liveAlias, server: server, chat: chat,
                downloads: downloads, quickstart: quickstart, dir: dir
            )
        }

        // One-shot: quit so the dogfood harness gets a clean exit.
        NSApp.terminate(nil)
    }

    @MainActor
    private static func runLiveChat(
        alias: String, server: ServerManager, chat: ChatViewModel,
        downloads: DownloadManager, quickstart: QuickstartCoordinator, dir: String
    ) async {
        log("live: starting sidecar for \(alias)…")
        await server.start(alias: alias)
        guard case .ready = server.state else {
            log("live: server did not reach ready (state=\(server.state)) — skipping")
            return
        }
        log("live: ready on port \(server.activePort); sending a chat turn")
        chat.send("Say hello and name one thing you can help with, in one sentence.",
                  alias: alias)
        // Wait for the stream to finish (cap ~90s).
        for _ in 0..<180 {
            if !chat.isStreaming { break }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        try? await Task.sleep(nanoseconds: 400_000_000)
        if let msg = chat.messages.last(where: { $0.role == .assistant }) {
            log("live: streaming=\(chat.isStreaming) status=\(msg.status) "
                + "content=\(msg.content.count)ch reasoning=\(msg.reasoning.count)ch "
                + "err=\(msg.errorMessage ?? "-")")
            log("live: content='\(msg.content.prefix(160))'")
            if !msg.reasoning.isEmpty {
                log("live: reasoning='\(msg.reasoning.prefix(160))'")
            }
        } else {
            log("live: no assistant message (isStreaming=\(chat.isStreaming), lastError=\(chat.lastError ?? "-"))")
        }
        render(
            liveContentView(
                server: server, chat: chat,
                downloads: downloads, quickstart: quickstart
            ),
            to: "\(dir)/content-chat-live.png"
        )
        log("live: wrote content-chat-live.png; stopping sidecar")
        await server.stop()
    }

    @MainActor
    private static func liveContentView(
        server: ServerManager, chat: ChatViewModel,
        downloads: DownloadManager, quickstart: QuickstartCoordinator
    ) -> AnyView {
        // ChatView reads DownloadManager + QuickstartCoordinator from the
        // environment (the Ollama-layout composer/quickstart affordances);
        // inject both or ImageRenderer traps with "No Observable object of
        // type DownloadManager found". The real app supplies them from
        // RapidApp's scene — this render path must mirror that.
        AnyView(
            ChatView(
                viewModel: chat,
                server: server,
                alias: .constant(server.servingAlias ?? ""),
                readiness: .ready(alias: server.servingAlias ?? "bonsai-1.7b-2bit")
            )
            .environment(downloads)
            .environment(quickstart)
            .frame(width: 900, height: 640)
            .tint(RapidTheme.brand)
            .background(RapidTheme.canvas)
        )
    }

    /// Render at the host's current appearance via ``ImageRenderer``.
    ///
    /// Retained unchanged for the pre-v1.0 component scenes (Connect
    /// Tools card body, Benchmark result, Consent, chat bubbles), which
    /// are plain view trees that ``ImageRenderer`` rasterises correctly
    /// and which benefit from its ``scale`` support.
    ///
    /// Full-window compositions must use ``renderHosted`` instead — see
    /// the note there about ``NavigationSplitView``.
    @MainActor
    private static func render(_ view: AnyView, to path: String) {
        let renderer = ImageRenderer(content: view)
        renderer.scale = 2.0
        guard let image = renderer.nsImage,
              let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff),
              let png = rep.representation(using: .png, properties: [:]) else {
            log("FAILED to render \(path)")
            return
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
        } catch {
            log("FAILED to write \(path): \(error)")
        }
    }

    /// Render a full-window composition at a pinned appearance.
    ///
    /// **Why not ``ImageRenderer``.** ``ImageRenderer`` cannot rasterise
    /// ``NavigationSplitView``: it emits a "prohibited" placeholder
    /// glyph instead of the view tree. That is not new — every
    /// `content-idle.png` / `content-min.png` this harness has ever
    /// written was that placeholder, verified by rendering from a
    /// pristine build of the parent commit. Any main-window screenshot
    /// taken from the old path was therefore worthless, which also
    /// means the split view has never actually been under visual
    /// regression review.
    ///
    /// Hosting the view in a real (offscreen, borderless) ``NSWindow``
    /// and calling ``cacheDisplay`` drives genuine AppKit layout, which
    /// the split view needs. The window also gives us:
    ///
    ///   * a correct ``NSAppearance`` for the whole tree, which is what
    ///     ``NSColor(name:dynamicProvider:)`` — i.e. every
    ///     ``RapidTheme`` colour — resolves against, and
    ///   * the display's backing scale, so the capture is 2x on Retina
    ///     rather than the 1x a window-less ``NSHostingView`` yields.
    ///
    /// ``\.colorScheme`` is set alongside the appearance to cover the
    /// SwiftUI-native side (materials, `.primary`/`.secondary`).
    @MainActor
    private static func renderHosted(
        _ view: AnyView,
        size: CGSize,
        appearance appearanceName: NSAppearance.Name,
        to path: String
    ) {
        let scheme: ColorScheme = appearanceName == .darkAqua ? .dark : .light
        let hosting = NSHostingView(
            rootView: view.environment(\.colorScheme, scheme)
        )
        hosting.frame = CGRect(origin: .zero, size: size)

        let window = NSWindow(
            contentRect: CGRect(origin: .zero, size: size),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.appearance = NSAppearance(named: appearanceName)
        window.contentView = hosting
        window.setFrame(CGRect(origin: .zero, size: size), display: true)
        hosting.layoutSubtreeIfNeeded()
        window.displayIfNeeded()

        // Let SwiftUI's first layout pass + any .task/.onAppear that
        // affects layout settle before we snapshot. Spinning the
        // runloop (rather than sleeping) lets those callbacks actually
        // run — they are main-actor bound.
        RunLoop.current.run(until: Date().addingTimeInterval(0.35))
        hosting.layoutSubtreeIfNeeded()
        window.displayIfNeeded()

        guard let rep = hosting.bitmapImageRepForCachingDisplay(in: hosting.bounds) else {
            log("FAILED to allocate bitmap for \(path)")
            return
        }
        hosting.cacheDisplay(in: hosting.bounds, to: rep)

        guard let png = rep.representation(using: .png, properties: [:]) else {
            log("FAILED to encode \(path)")
            return
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
        } catch {
            log("FAILED to write \(path): \(error)")
        }
    }

    private static func log(_ message: String) {
        FileHandle.standardError.write(Data("[dev-snapshot] \(message)\n".utf8))
    }
}
