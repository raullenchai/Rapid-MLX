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

        // ``ContentView`` reads this from the environment (the browse
        // tool's per-fetch approval dialog). Without it every capture
        // below traps with "No Observable object of type
        // BrowseApprovalStore found" before the first PNG is written —
        // the harness had been dead since that dependency landed. A
        // throwaway instance is right here: the snapshot never approves
        // anything, it only needs the object to exist.
        let browseApproval = BrowseApprovalStore()
        // ``ContentView`` also reads ``ImageGenViewModel`` from the
        // environment (the Images tab). Same rule as ``browseApproval``: a
        // throwaway instance so the view can be evaluated without trapping.
        let imageGen = ImageGenViewModel(server: server)
        let audio = AudioViewModel(server: server)
        // Same rule again, for the connectors stack (issue #1716).
        // ``ContentView`` reads ``MCPCatalog`` and ``MCPToolApprovalStore``
        // for its tool-approval sheet, and ``SettingsConnectorsPanel``
        // additionally reads ``MCPConfigStore`` and ``MCPToolRegistry``.
        // Without these the harness trapped on the FIRST capture with
        // "No Observable object of type MCPCatalog found" and wrote zero
        // PNGs — it had been dead since connectors landed, the same way it
        // was dead before ``browseApproval`` was added above.
        //
        // The config store is pointed at a scratch file inside the snapshot
        // directory so a capture run can never read or write the real
        // ``mcp.json``. The catalog is given a nil endpoint provider, so it
        // never polls anything.
        let snapshotMCPConfig = MCPConfigStore(
            fileURL: URL(fileURLWithPath: dir).appendingPathComponent("snapshot-mcp.json")
        )
        let snapshotMCPCatalog = MCPCatalog { nil }
        let snapshotMCPApproval = MCPToolApprovalStore()
        let snapshotMCPTools = MCPToolRegistry(
            catalog: snapshotMCPCatalog,
            approval: snapshotMCPApproval
        )
        let snapshotSparkleUpdater = SparkleUpdateController(infoDictionary: [:])
        let snapshotWebSearch = WebSearchConfig()
        let snapshotPerfDefaults = UserDefaults(suiteName: "rapid.dev-snapshot.perf")!
        snapshotPerfDefaults.removePersistentDomain(forName: "rapid.dev-snapshot.perf")
        let snapshotPerfConfig = ModelPerfConfigStore(defaults: snapshotPerfDefaults)

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
                    // ``FailedReplaceBanner`` (rendered by ContentView when a
                    // Finder Replace silently failed) hands off to Sparkle, so
                    // the controller has to be in this chain too — SwiftUI
                    // traps on a missing observable the first time it renders.
                    .environment(snapshotSparkleUpdater)
                    .environment(quickstart)
                    .environment(dockPromptStore)
                    .environment(browseApproval)
                    .environment(imageGen)
                    .environment(audio)
                    .environment(snapshotMCPCatalog)
                    .environment(snapshotMCPApproval)
                    .frame(width: width, height: height)
            )
        }

        // LIVE mode must run before the static matrix. The matrix below
        // renders ~200 full-size views and temporarily raises process memory;
        // running live last can trip ServerManager's real pre-load safety
        // gate even for a 0.6B model, leaving state=.idle and silently skipping
        // the only end-to-end sidecar check. Use a separate non-persisting
        // chat model so the real turn cannot leak into the static fixtures.
        if let liveAlias = ProcessInfo.processInfo.environment["RAPID_DEV_SERVE_ALIAS"],
           !liveAlias.isEmpty {
            let liveChat = ChatViewModel(
                sampling: sampling,
                customInstructions: chat.customInstructions,
                server: server,
                persistsConversations: false
            )
            await runLiveChat(
                alias: liveAlias, server: server, chat: liveChat,
                downloads: downloads, quickstart: quickstart, dir: dir
            )
            await server.stop()
            server.dismissTerminalState()
        }

        // The Images tab, rendered standalone — ``ContentView`` owns its
        // ``SidebarSection`` privately, so (like ``launchView``) the detail
        // surface is captured directly rather than by driving navigation.
        func imagesView(width: CGFloat, height: CGFloat) -> AnyView {
            AnyView(
                ImagesView(viewModel: imageGen, server: server)
                    .tint(RapidTheme.brandAmber)
                    // ``ImagesView`` deep-links to Settings → Model
                    // Management from its readiness banner, so it reads the
                    // router. Missing it trapped the harness here, one
                    // capture after the MCP fix above.
                    .environment(settingsRouter)
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
        render(contentView(width: 720, height: 560), to: "\(dir)/content-min.png")
        // Narrow window: the status footer sheds its readouts through
        // ViewThatFits rather than squeezing them to ellipses, and the version
        // pill must stay on one line. Only a width this small exercises it.
        render(contentView(width: 380, height: 560), to: "\(dir)/content-narrow.png")
        // Images tab (empty state — no results, catalog not yet resolved).
        render(imagesView(width: 700, height: 640), to: "\(dir)/images-empty.png")
        // Narrow composer: the canvas controls wrap to the two-row
        // ViewThatFits layout, which only this width exercises.
        render(imagesView(width: 420, height: 640), to: "\(dir)/images-narrow.png")
        renderHosted(imagesView(width: 700, height: 640),
                     size: CGSize(width: 700, height: 640),
                     appearance: .darkAqua, to: "\(dir)/images-dark.png")

        // Image-edit mode at regular and narrow widths. Use the bundled mascot
        // as a deterministic source image; this exercises the stage actions,
        // source strip, edit-capable picker, and wrapped composer without weights.
        imageGen.imageModels = [
            ModelEntry(
                alias: "flux2-klein-4b", hfRepo: "snapshot/generate",
                sizeOnDisk: "4.3 GiB", cached: true, kind: .image,
                imageCapability: .generationAndEditing
            ),
        ]
        imageGen.selectedAlias = "flux2-klein-4b"
        if let mascot = CheetahLogo.load(forSize: 120),
           let tiff = mascot.tiffRepresentation,
           let rep = NSBitmapImageRep(data: tiff),
           let png = rep.representation(using: .png, properties: [:]) {
            imageGen.beginEdit(GeneratedImage(
                pngData: png, prompt: "Cheetah", isEdit: false
            ))
            imageGen.prompt = "Change the background to a bright photo studio"
            renderHosted(imagesView(width: 700, height: 640),
                         size: CGSize(width: 700, height: 640),
                         appearance: .aqua, to: "\(dir)/images-edit.png")
            renderHosted(imagesView(width: 420, height: 640),
                         size: CGSize(width: 420, height: 640),
                         appearance: .aqua, to: "\(dir)/images-edit-narrow.png")
        }

        // Scenario 1b (v1.0 visual foundation): the Light/Dark × surface
        // matrix the Phase-1 review runs on. Chat and Launch are the two
        // surfaces this phase repaints, so both are captured at the
        // 900x640 review size in both appearances, plus one shot each at
        // the 720x560 window floor to prove the layout survives it.
        let reviewSize = CGSize(width: 900, height: 640)
        let floorSize = CGSize(width: 720, height: 560)

        renderHosted(contentView(width: 900, height: 640), size: reviewSize,
                     appearance: .aqua, to: "\(dir)/chat-900x640-light.png")
        renderHosted(contentView(width: 900, height: 640), size: reviewSize,
                     appearance: .darkAqua, to: "\(dir)/chat-900x640-dark.png")
        renderHosted(contentView(width: 720, height: 560), size: floorSize,
                     appearance: .aqua, to: "\(dir)/chat-720x560-light.png")
        renderHosted(contentView(width: 720, height: 560), size: floorSize,
                     appearance: .darkAqua, to: "\(dir)/chat-720x560-dark.png")
        renderHosted(launchView(width: 900, height: 640), size: reviewSize,
                     appearance: .aqua, to: "\(dir)/launch-900x640-light.png")
        renderHosted(launchView(width: 900, height: 640), size: reviewSize,
                     appearance: .darkAqua, to: "\(dir)/launch-900x640-dark.png")
        renderHosted(launchView(width: 720, height: 560), size: floorSize,
                     appearance: .aqua, to: "\(dir)/launch-720x560-light.png")
        renderHosted(launchView(width: 720, height: 560), size: floorSize,
                     appearance: .darkAqua, to: "\(dir)/launch-720x560-dark.png")

        // Scenario 1c (Paper 05.2): the Step 2 model-selection review matrix.
        //
        // Every micro-stage and every state its footer can be in, at the three
        // documented widths, in both appearances. Rendered from the REAL
        // ``QuickstartView`` against a synthetic catalogue rather than a live
        // engine, so the states that need a failed or still-loading catalogue
        // are reachable at all — and so nothing here touches a server, a
        // download, or the user's model cache.
        captureStep2Matrix(
            quickstart: quickstart,
            downloads: downloads,
            server: server,
            settingsRouter: settingsRouter,
            dir: dir
        )

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
                .unknownModel(alias: "mlx-community/Some-Custom-Repo"),
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

        // MARK: UI-2 Slice 1 — the core-workspace review matrix
        //
        // The captures above review Chat and Launch at 900x640 and at the
        // 720x560 window floor, which were the sizes the v1.0 pass was
        // signed off at. Slice 1 is reviewed at three DIFFERENT widths —
        // 1440, 1000 and 720 — because that is where the lifecycle band
        // changes shape, and a 900pt capture would silently review only
        // its middle step.
        //
        // Chat and Images are captured at every width in both
        // appearances. The band is captured separately at all three
        // widths: a live ``ContentView`` can only show whichever state
        // the harness is actually in, and staging a real multi-gigabyte
        // download is not something a capture run can do.
        let ui2Widths: [CGFloat] = [1440, 1000, 720]
        let ui2Appearances: [(NSAppearance.Name, String)] = [(.aqua, "light"), (.darkAqua, "dark")]

        for windowWidth in ui2Widths {
            // 900 tall at every width: the point of the sweep is the
            // horizontal behaviour, and holding height fixed makes the
            // three images directly comparable.
            let size = CGSize(width: windowWidth, height: 900)
            for (appearance, mode) in ui2Appearances {
                renderHosted(
                    contentView(width: windowWidth, height: 900), size: size,
                    appearance: appearance,
                    to: "\(dir)/ui2-chat-\(Int(windowWidth))-\(mode).png"
                )
                renderHosted(
                    imagesView(width: windowWidth, height: 900), size: size,
                    appearance: appearance,
                    to: "\(dir)/ui2-images-result-\(Int(windowWidth))-\(mode).png"
                )
            }
        }

        // Images again with the stage cleared, because the capture above
        // inherits the edit-mode seeding from Scenario 1 and therefore
        // only ever shows a RESULT. The empty stage is the surface this
        // slice actually changed — it is where the mascot was replaced by
        // the aspect preview — so it needs its own sweep.
        //
        // Safe to mutate here: every other Images capture has already
        // been taken by this point in the run.
        imageGen.cancelEdit()
        imageGen.prompt = ""
        // ``activeImage`` is derived from ``results``, so clearing the
        // list is what empties the stage.
        imageGen.results = []
        for windowWidth in ui2Widths {
            let size = CGSize(width: windowWidth, height: 900)
            for (appearance, mode) in ui2Appearances {
                renderHosted(
                    imagesView(width: windowWidth, height: 900), size: size,
                    appearance: appearance,
                    to: "\(dir)/ui2-images-empty-\(Int(windowWidth))-\(mode).png"
                )
            }
        }
        // The aspect preview tracks the live selection, so sweep the
        // three aspects too — a square-only capture would not show that
        // the frame actually changes shape.
        for aspect in ImageGenViewModel.Aspect.allCases {
            imageGen.aspect = aspect
            renderHosted(
                imagesView(width: 1000, height: 900),
                size: CGSize(width: 1000, height: 900),
                appearance: .aqua,
                to: "\(dir)/ui2-images-aspect-\(aspect.rawValue).png"
            )
        }
        imageGen.aspect = .square

        // The band, driven directly by the two states that open it, at
        // each of its three heights. Rendered over a stub transcript so
        // the graphite-against-canvas separation is reviewable rather
        // than the band floating on nothing.
        func bandProof(_ readiness: ModelReadiness, width: CGFloat) -> AnyView {
            AnyView(
                VStack(spacing: 0) {
                    LifecycleBand(readiness: readiness, width: width)
                    Spacer(minLength: 0)
                }
                .frame(width: width, height: LifecycleBand.height(for: width) + 80)
                .background(RapidTheme.surfaceCanvas)
                .tint(RapidTheme.brandAmber)
            )
        }
        let bandStates: [(String, ModelReadiness)] = [
            ("downloading", .downloading(
                alias: "qwen3.5-9b-4bit",
                detail: "1.2 GB of 5.0 GB · 8.4 MB/s · 7 min left",
                fraction: 0.24
            )),
            // No fraction: the indeterminate case, which must render an
            // honest bare track rather than a bar pinned at zero.
            ("starting", .starting(
                alias: "bonsai-1.7b-2bit",
                detail: "Loading the model into memory…"
            )),
        ]
        let ui2DetailWidths: [(window: CGFloat, detail: CGFloat)] = [
            (1440, RapidTheme.Layout.Breakpoint.wide),
            (1000, RapidTheme.Layout.Breakpoint.mid),
            (720, RapidTheme.Layout.Breakpoint.floor),
        ]
        for (label, state) in bandStates {
            for widths in ui2DetailWidths {
                let size = CGSize(
                    width: widths.detail,
                    height: LifecycleBand.height(for: widths.detail) + 80
                )
                for (appearance, mode) in ui2Appearances {
                    renderHosted(
                        bandProof(state, width: widths.detail), size: size,
                        appearance: appearance,
                        to: "\(dir)/ui2-band-\(label)-\(Int(widths.window))-\(mode).png"
                    )
                }
            }
        }

        // NOTE ON REDUCE MOTION. There is deliberately no capture for it
        // here, and the reason is a real constraint rather than an
        // oversight: `\.accessibilityReduceMotion` is a READ-ONLY
        // environment key on macOS — it mirrors the system setting, and
        // SwiftUI offers no writable keypath — so a harness cannot stage
        // the reduced rendering the way it stages an appearance or a
        // width. Injecting it would need a seam through every call site,
        // which is a refactor this visual slice has no business making.
        //
        // The contract is held instead by ``RapidMotion.shouldPulse`` and
        // by the source guard asserting both perpetual loops in the
        // Images HUD route through it. Whether the reduced frame LOOKS
        // right stays a manual check against the system setting.

        // The rail with a MID-LIST row selected, which the capture above
        // cannot show: it selects ``.launch``, the last row, so it proves
        // the bar renders but not that the bar leaves its neighbours
        // alone. Selecting Images puts a marked row between two unmarked
        // ones, which is where a leading bar that took layout width —
        // rather than overlaying it — would visibly step the labels in
        // and out.
        //
        // ``.chat`` is deliberately NOT used here: it marks no nav row at
        // all (it highlights a conversation row, and this fixture seeds
        // no history), so it would capture an empty rail and look like
        // the selection had regressed.
        func sidebarSelectionProof() -> AnyView {
            AnyView(
                SidebarView(
                    selection: .constant(.images),
                    chat: chat,
                    onNewChat: {},
                    onSelectConversation: { _ in }
                )
                .frame(width: SidebarView.columnIdealWidth, height: 640)
                .background(RapidTheme.surfaceSidebar)
                .tint(RapidTheme.brandAmber)
            )
        }
        renderHosted(sidebarSelectionProof(), size: sidebarSize,
                     appearance: .aqua, to: "\(dir)/ui2-sidebar-selected-light.png")
        renderHosted(sidebarSelectionProof(), size: sidebarSize,
                     appearance: .darkAqua, to: "\(dir)/ui2-sidebar-selected-dark.png")

        // Settings → Model Management. The panel seeds its catalog
        // synchronously from ``ModelCatalogCache``'s mirror, so warm the
        // cache first or every capture is the spinner.
        if let binary = server.binaryPath {
            _ = await ModelCatalogCache.shared.entries(
                binary: binary, generation: downloads.cacheGeneration
            )
        }
        func modelManagement(width: CGFloat) -> AnyView {
            AnyView(
                SettingsModelManagementPanel()
                    .environment(server)
                    .environment(downloads)
                    .frame(width: width, alignment: .top)
                    .padding(20)
                    .background(RapidTheme.surfaceCanvas)
                    .tint(RapidTheme.brandAmber)
            )
        }
        // Both the Settings window's default content width and its 720pt
        // floor: the recommended card's action button clipped at BOTH, so
        // a single wide capture would not have shown the bug or its fix.
        renderHosted(modelManagement(width: 660), size: CGSize(width: 700, height: 1100),
                     appearance: .aqua, to: "\(dir)/model-management-light.png")
        renderHosted(modelManagement(width: 480), size: CGSize(width: 520, height: 1100),
                     appearance: .aqua, to: "\(dir)/model-management-narrow.png")

        // The table heading in every state the filter + search can put it
        // in. A running panel can only ever be captured in one of them,
        // and the heading is what this pass changed.
        func headingStates() -> AnyView {
            let cases: [(String, ModelCacheActions.ListHeading)] = [
                ("no filter", ModelCacheActions.listHeading(
                    filter: .all, query: "", visibleCount: 175, totalCount: 175)),
                ("search \"qwen3.6\"", ModelCacheActions.listHeading(
                    filter: .all, query: "qwen3.6", visibleCount: 4, totalCount: 175)),
                ("Cached segment", ModelCacheActions.listHeading(
                    filter: .cached, query: "", visibleCount: 3, totalCount: 175)),
                ("Not cached + search", ModelCacheActions.listHeading(
                    filter: .notCached, query: "gemma", visibleCount: 6, totalCount: 175)),
                ("no matches", ModelCacheActions.listHeading(
                    filter: .all, query: "zzz", visibleCount: 0, totalCount: 175)),
            ]
            return AnyView(
                VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                    ForEach(Array(cases.enumerated()), id: \.offset) { _, item in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(item.0).font(.caption2).foregroundStyle(.tertiary)
                            ModelsTableHeading(heading: item.1)
                        }
                    }
                }
                .padding(RapidTheme.Space.xl)
                .frame(width: 420, alignment: .leading)
                .background(RapidTheme.surfaceCanvas)
                .tint(RapidTheme.brandAmber)
            )
        }
        renderHosted(headingStates(), size: CGSize(width: 420, height: 300),
                     appearance: .aqua, to: "\(dir)/model-management-heading-states.png")

        // ── UI-1: the whole Settings window, every category, both
        // appearances, at the three sizes the phase has to hold.
        //
        // Panel-at-a-time captures (like the Model Management pair above)
        // cannot show what this phase is actually about: whether the rail,
        // the page titles, the section treatment and the row rhythm agree
        // ACROSS categories. So this renders the real ``SettingsView``
        // shell and walks the category rail.
        //
        // The stores Settings binds that ``runIfRequested`` isn't handed
        // are built here rather than threaded through the call site — a
        // dev-only capture should not widen a production signature. The
        // MCP config store is pointed at a scratch file inside the
        // snapshot directory so a capture run cannot touch the real one.
        func settingsShell(category: SettingsView.Category, size: CGSize) -> AnyView {
            settingsRouter.requestedCategory = category
            return AnyView(
                SettingsView()
                    .environment(chat)
                    .environment(sampling)
                    .environment(chat.customInstructions)
                    .environment(appearance)
                    .environment(settingsRouter)
                    .environment(server)
                    // The capture loop walks Category.allCases, which in a
                    // debug build includes Developer — and that panel reads
                    // the coordinator.
                    .environment(quickstart)
                    .environment(downloads)
                    .environment(updater)
                    .environment(snapshotSparkleUpdater)
                    .environment(dockPromptStore)
                    .environment(snapshotWebSearch)
                    .environment(browseApproval)
                    .environment(snapshotMCPConfig)
                    .environment(snapshotMCPCatalog)
                    .environment(snapshotMCPApproval)
                    .environment(snapshotMCPTools)
                    .environment(snapshotPerfConfig)
                    .frame(width: size.width, height: size.height)
                    .tint(RapidTheme.brandAmber)
            )
        }

        // Default (the scene's own ``defaultSize``), the size the brief
        // names, and the hard floor. 480pt tall is the floor and is
        // deliberately short — a category that does not scroll cleanly
        // there is the defect this phase had to fix.
        let settingsSizes: [(label: String, size: CGSize)] = [
            ("default", CGSize(width: 900, height: 720)),
            ("900x640", CGSize(width: 900, height: 640)),
            ("720x480", CGSize(width: 720, height: 480)),
        ]
        for category in SettingsView.Category.allCases {
            for (label, size) in settingsSizes {
                for (mode, name) in [("light", NSAppearance.Name.aqua),
                                     ("dark", NSAppearance.Name.darkAqua)] {
                    renderHosted(
                        settingsShell(category: category, size: size),
                        size: size,
                        appearance: name,
                        to: "\(dir)/settings-\(category.rawValue)-\(label)-\(mode).png"
                    )
                }
            }
        }

        // ── UI-1 refinement proof sheets ──────────────────────────
        //
        // The shell captures above show each category in its DEFAULT
        // state. The refinement review is about states a static capture
        // of the default cannot reach: a disclosure open, a segmented
        // control on each segment, a switch on AND off, a disabled
        // button, each web-search backend. These render those states
        // side by side, the same way ``headingStates()`` above renders
        // every filter heading at once.

        func toolsPanel(expanded: Set<String>, width: CGFloat) -> AnyView {
            AnyView(
                SettingsToolsPanel(initiallyExpanded: expanded)
                    .environment(chat)
                    .environment(snapshotWebSearch)
                    .environment(browseApproval)
                    .frame(width: width, alignment: .top)
                    .padding(RapidTheme.Space.xl)
                    .background(RapidTheme.surfaceCanvas)
                    .tint(RapidTheme.brandAmber)
            )
        }
        for (mode, name) in [("light", NSAppearance.Name.aqua),
                             ("dark", NSAppearance.Name.darkAqua)] {
            renderHosted(toolsPanel(expanded: [], width: 620),
                         size: CGSize(width: 660, height: 1000),
                         appearance: name, to: "\(dir)/tools-collapsed-\(mode).png")
            renderHosted(toolsPanel(expanded: ["web_search", "browse", "weather"], width: 620),
                         size: CGSize(width: 660, height: 1500),
                         appearance: name, to: "\(dir)/tools-expanded-\(mode).png")
        }

        // Each web-search backend, so the radio group and the key field
        // that appears for the keyed backends are both reviewable.
        for provider in WebSearchProvider.allCases {
            snapshotWebSearch.provider = provider
            renderHosted(toolsPanel(expanded: [], width: 620),
                         size: CGSize(width: 660, height: 1000),
                         appearance: .aqua,
                         to: "\(dir)/tools-backend-\(provider.id).png")
        }
        snapshotWebSearch.provider = .duckduckgo

        // Connectors with the master switch ON — the state that reveals
        // the servers list, the empty hint and the approvals card.
        snapshotMCPConfig.isEnabled = true
        func connectorsPanel(width: CGFloat) -> AnyView {
            AnyView(
                SettingsConnectorsPanel()
                    .environment(snapshotMCPConfig)
                    .environment(snapshotMCPCatalog)
                    .environment(snapshotMCPApproval)
                    .environment(snapshotMCPTools)
                    .environment(server)
                    .frame(width: width, alignment: .top)
                    .padding(RapidTheme.Space.xl)
                    .background(RapidTheme.surfaceCanvas)
                    .tint(RapidTheme.brandAmber)
            )
        }
        for (mode, name) in [("light", NSAppearance.Name.aqua),
                             ("dark", NSAppearance.Name.darkAqua)] {
            renderHosted(connectorsPanel(width: 620),
                         size: CGSize(width: 660, height: 900),
                         appearance: name, to: "\(dir)/connectors-enabled-\(mode).png")
        }
        snapshotMCPConfig.isEnabled = false

        // The Add-connector sheet, both appearances.
        func connectorSheet() -> AnyView {
            AnyView(
                MCPServerEditorSheet(original: nil, onSave: { _ in }, onCancel: {})
                    .tint(RapidTheme.brandAmber)
            )
        }
        for (mode, name) in [("light", NSAppearance.Name.aqua),
                             ("dark", NSAppearance.Name.darkAqua)] {
            renderHosted(connectorSheet(), size: CGSize(width: 520, height: 760),
                         appearance: name, to: "\(dir)/connector-sheet-\(mode).png")
        }

        // Buttons, toggles and segmented controls in every state the
        // review asks to see, on one sheet per appearance.
        for (mode, name) in [("light", NSAppearance.Name.aqua),
                             ("dark", NSAppearance.Name.darkAqua)] {
            renderHosted(AnyView(SettingsControlProofSheet()),
                         size: CGSize(width: 560, height: 620),
                         appearance: name, to: "\(dir)/controls-proof-\(mode).png")
        }

        // The menu-bar mark, at the size the bar actually draws it, on the
        // three backgrounds it has to survive. A template image renders
        // black-on-transparent in isolation, so these composite it the way
        // AppKit will: ink on a light bar, on a dark bar, and knocked out
        // on the selection fill while the menu is open.
        renderHosted(
            AnyView(MenuBarMarkProofSheet()),
            size: CGSize(width: 360, height: 150),
            appearance: .aqua,
            to: "\(dir)/menubar-mark.png"
        )

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

        // Scenario 3: the "Connect your agents" sheet (pure SwiftUI, so it
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

        // Scenario 4: the telemetry consent sheet (the privacy "gate").
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
    /// Tools card body, Consent, chat bubbles), which
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
    // MARK: - Step 2 model selection (Paper 05.2)

    /// A synthetic chat catalogue big enough to scroll, with a realistic mix of
    /// cached and uncached rows.
    ///
    /// Synthetic on purpose. The states this matrix exists to prove — catalogue
    /// error, still-loading, no results, empty cache — are precisely the ones a
    /// healthy live engine will not produce on demand, and the alternative
    /// (breaking a real engine to photograph the wreckage) touches things this
    /// harness must not touch.
    @MainActor
    private static func step2Catalog() -> [ModelEntry] {
        let cached: [(String, String)] = [
            ("gemma3-1b-qat-4bit", "1.1 GiB"),
            ("qwen3.5-4b-4bit", "2.4 GiB"),
        ]
        let uncached = [
            "lfm2.5-1b-4bit", "lfm2.5-2.6b-4bit", "qwen3-0.6b-4bit",
            "qwen3.5-9b-4bit", "qwen3.5-14b-4bit", "qwen3.5-32b-4bit",
            "llama3.3-8b-4bit", "llama3.3-70b-4bit", "mistral-7b-4bit",
            "mixtral-8x7b-4bit", "phi4-14b-4bit", "gemma3-4b-4bit",
            "gemma3-12b-4bit", "gemma3-27b-4bit", "deepseek-r1-7b-4bit",
            "ling-3.0-tiny-fp8", "smollm3-3b-4bit", "olmo2-13b-4bit",
        ]
        return cached.map {
            ModelEntry(alias: $0.0, hfRepo: "mlx-community/\($0.0)",
                       sizeOnDisk: $0.1, cached: true, kind: .chat)
        } + uncached.map {
            ModelEntry(alias: $0, hfRepo: "mlx-community/\($0)",
                       sizeOnDisk: nil, cached: false, kind: .chat)
        }
        // An image row, to prove approved default D4 filters it out of a
        // catalogue whose job is the first CHAT model.
        + [ModelEntry(alias: "flux2-klein-4b", hfRepo: "mlx-community/flux2",
                      sizeOnDisk: "4.3 GiB", cached: true, kind: .image)]
    }

    /// Render every Step 2 state at every documented width, in both
    /// appearances.
    @MainActor
    private static func captureStep2Matrix(
        quickstart: QuickstartCoordinator,
        downloads: DownloadManager,
        server: ServerManager,
        settingsRouter: SettingsRouter,
        dir: String
    ) {
        let catalog = step2Catalog()

        func step2(
            width: CGFloat,
            height: CGFloat,
            catalog: [ModelEntry],
            loaded: Bool
        ) -> AnyView {
            AnyView(
                QuickstartView(
                    coordinator: quickstart,
                    downloads: downloads,
                    server: server,
                    cachedModels: catalog,
                    catalogLoaded: loaded,
                    onSkip: {},
                    onSeedWelcome: { true },
                    onCompleted: {},
                    // A fixed probe so the Review screen's free-space row is
                    // deterministic across runs rather than reporting whatever
                    // this machine happens to have free.
                    freeBytesProbe: { 96_000_000_000 }
                )
                .environment(settingsRouter)
                .frame(width: width, height: height)
            )
        }

        /// The three documented review widths.
        let sizes: [(label: String, size: CGSize)] = [
            ("1440x900", CGSize(width: 1440, height: 900)),
            ("1000x700", CGSize(width: 1000, height: 700)),
            ("720x560", CGSize(width: 720, height: 560)),
        ]
        let appearances: [(String, NSAppearance.Name)] = [
            ("light", .aqua), ("dark", .darkAqua),
        ]

        /// Drive the coordinator into one state, then photograph it everywhere.
        func capture(_ name: String, catalog: [ModelEntry], loaded: Bool,
                     _ arrange: () -> Void) {
            for (sizeLabel, size) in sizes {
                arrange()
                for (schemeLabel, appearance) in appearances {
                    renderHosted(
                        step2(width: size.width, height: size.height,
                              catalog: catalog, loaded: loaded),
                        size: size,
                        appearance: appearance,
                        to: "\(dir)/step2-\(name)-\(sizeLabel)-\(schemeLabel).png"
                    )
                }
            }
        }

        let starter = QuickstartCoordinator.defaultChoice
        let cachedChoice = QuickstartView.choice(
            forCatalogEntry: catalog.first { $0.alias == "gemma3-1b-qat-4bit" }!
        )
        let uncachedChoice = QuickstartView.choice(
            forCatalogEntry: catalog.first { $0.alias == "qwen3.5-9b-4bit" }!
        )

        // 2b — recommendation loading. The catalogue has not landed, so the
        // shortlist cannot say what is cached and the footer is disabled.
        capture("finding-fit", catalog: [], loaded: false) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.resolveRecommendationLoading(catalogLoaded: false)
        }

        // 2c — the recommended shortlist, with two models already on this Mac.
        capture("shortlist", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.resolveRecommendationLoading(catalogLoaded: true)
            quickstart.select(starter)
        }

        // 2c — a cached model selected: the primary reads Start existing model.
        capture("shortlist-cached-selection", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.resolveRecommendationLoading(catalogLoaded: true)
            quickstart.select(cachedChoice)
        }

        // 2d-i — the catalogue, still loading.
        capture("browse-loading", catalog: [], loaded: false) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
        }

        // 2d — the catalogue, populated, with an uncached pick selected.
        capture("browse-uncached-selection", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(uncachedChoice)
        }

        // 2d — the catalogue with a cached pick selected.
        capture("browse-cached-selection", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(cachedChoice)
        }

        // 2d — the pick is searched away. The alias is retained; the primary is
        // disabled and still shows the neutral verb. This is the "no valid
        // visible selection" state and the disabled-CTA state at once.
        capture("browse-no-selection-visible", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(uncachedChoice)
            quickstart.catalogQuery = "gemma"
        }

        // 2d — a search that matches nothing.
        capture("browse-no-results", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.catalogQuery = "zzzz-no-such-model"
        }

        // 2d — the catalogue subprocess failed (ModelCatalog's `[]` sentinel).
        capture("browse-catalog-error", catalog: [], loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
        }

        // 2d — an empty cache under the Cached filter. Note this is a healthy
        // catalogue with nothing downloaded, NOT the phantom "No" row: that
        // parser bug is fixed upstream (#1920) and is deliberately not worked
        // around here.
        let emptyCache = catalog
            .filter { $0.kind == .chat }
            .map {
                ModelEntry(alias: $0.alias, hfRepo: $0.hfRepo, sizeOnDisk: nil,
                           cached: false, kind: .chat)
            }
        capture("browse-empty-cache", catalog: emptyCache, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.catalogFilter = .cached
        }

        // 2e — Review download for an uncached model, opened from the
        // catalogue. Primary reads Download & start; Back names the catalogue.
        capture("review-uncached", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(uncachedChoice)
            quickstart.beginReviewDownload(origin: .catalogue)
        }

        // 2e — Review for a model already on disk. Primary reads Start existing
        // model, and no download size is quoted as a cost.
        capture("review-cached", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.select(cachedChoice)
            quickstart.beginReviewDownload(origin: .shortlist)
        }

        // Back restoration — the state the user lands in after Review → Back
        // from the catalogue: catalogue re-shown, query/filter/sort intact,
        // pick still selected.
        capture("back-restores-catalogue", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.catalogQuery = "qwen"
            quickstart.catalogSort = .nameAscending
            quickstart.select(uncachedChoice)
            quickstart.rememberCatalogAnchor(uncachedChoice.alias)
            quickstart.beginReviewDownload(origin: .catalogue)
            quickstart.backFromReviewDownload()
        }

        // Back restoration onto the shortlist, where a catalogue pick has to
        // come back as YOUR PICK (approved default D2) rather than vanishing.
        //
        // Deliberately mistral, not qwen3.5-9b: the 9B is one of the four
        // native shortlist rows, so picking it in the catalogue and coming back
        // correctly shows NO extra group — which proves the "the group
        // disappears when the selection is native" half of D2, and nothing at
        // all about the half this capture is for.
        let offShortlistChoice = QuickstartView.choice(
            forCatalogEntry: catalog.first { $0.alias == "mistral-7b-4bit" }!
        )
        capture("back-restores-shortlist-your-pick", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(offShortlistChoice)
            quickstart.backToRecommendedModels()
        }

        // The other half of D2, so both are on the canvas: a native shortlist
        // row picked in the catalogue comes back selected in place, with no
        // YOUR PICK group added.
        capture("back-restores-shortlist-native", catalog: catalog, loaded: true) {
            quickstart._testingReset()
            quickstart.advanceToChooseModel()
            quickstart.beginBrowsingCatalog()
            quickstart.select(uncachedChoice)
            quickstart.backToRecommendedModels()
        }

        // Leave the coordinator as the rest of the harness found it.
        quickstart._testingReset()
    }

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

/// DEV-ONLY proof sheet for the menu-bar template mark.
///
/// A template ``NSImage`` is ink plus alpha, so rendering it on its own
/// says nothing about how it will look — the whole question is what
/// AppKit does with it. This composites the shipped image the three ways
/// the menu bar will: black on a light bar, white on a dark bar, and
/// white on the selection fill while the menu is open.
private struct MenuBarMarkProofSheet: View {
    private var mark: Image? {
        RapidRMark.menuBarTemplateImage().map { Image(nsImage: $0) }
    }

    var body: some View {
        VStack(spacing: 0) {
            bar(background: Color(white: 0.96), ink: .black, label: "Light bar")
            bar(background: Color(white: 0.13), ink: .white, label: "Dark bar")
            bar(background: Color(red: 0, green: 0.48, blue: 1), ink: .white, label: "Menu open")
        }
        .frame(width: 360)
    }

    private func bar(background: Color, ink: Color, label: String) -> some View {
        HStack(spacing: RapidTheme.Space.md) {
            Text(label)
                .font(RapidFont.caption)
                .foregroundStyle(ink.opacity(0.6))
            Spacer()
            if let mark {
                // `.template` + `foregroundStyle` is what AppKit does to a
                // template image; rendering it any other way would be
                // testing something the app never does.
                mark
                    .renderingMode(.template)
                    .foregroundStyle(ink)
            }
        }
        .padding(.horizontal, RapidTheme.Space.lg)
        .frame(height: 50)
        .background(background)
    }
}

/// DEV-ONLY proof sheet for the shared Settings controls.
///
/// Every button tier, both switch states, and each segmented control on
/// each of its segments — in one frame, so "do the heights match, is the
/// ink on amber dark, is the disabled state legible" can be answered by
/// looking rather than by reading modifier chains.
private struct SettingsControlProofSheet: View {
    private enum Segment: Hashable { case first, second, third }

    @State private var segA: Segment = .first
    @State private var segB: Segment = .second
    @State private var toggleOn = true
    @State private var toggleOff = false

    var body: some View {
        VStack(alignment: .leading, spacing: RapidTheme.Space.xl) {
            SectionHeader("Buttons", emphasis: .section)
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                // Specimen controls: rendered only offscreen via
                // ``ImageRenderer`` for the visual harness, never AX-driven.
                // They still carry identifiers so the AX-identifier gate can
                // hold its zero-exemption line — see the gate's doc comment.
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Primary") {}.buttonStyle(.rapidPrimary)
                        .accessibilityIdentifier("DevSnapshot.Specimen.Primary")
                    Button("Secondary") {}.buttonStyle(.rapidSecondary)
                        .accessibilityIdentifier("DevSnapshot.Specimen.Secondary")
                    Button("Destructive") {}.buttonStyle(.rapidDestructive)
                        .accessibilityIdentifier("DevSnapshot.Specimen.Destructive")
                    Button("Tertiary") {}.buttonStyle(.rapidTertiary)
                        .accessibilityIdentifier("DevSnapshot.Specimen.Tertiary")
                }
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Primary") {}.buttonStyle(.rapidPrimaryCompact)
                        .accessibilityIdentifier("DevSnapshot.Specimen.PrimaryCompact")
                    Button("Secondary") {}.buttonStyle(.rapidSecondaryCompact)
                        .accessibilityIdentifier("DevSnapshot.Specimen.SecondaryCompact")
                    Button("Destructive") {}.buttonStyle(.rapidDestructiveCompact)
                        .accessibilityIdentifier("DevSnapshot.Specimen.DestructiveCompact")
                    QuietIconButton(symbol: "trash", label: "Delete",
                                    tint: RapidTheme.statusError) {}
                        .accessibilityIdentifier("DevSnapshot.Specimen.Icon.Delete")
                    QuietIconButton(symbol: "arrow.down.circle", label: "Download") {}
                        .accessibilityIdentifier("DevSnapshot.Specimen.Icon.Download")
                }
                HStack(spacing: RapidTheme.Space.sm) {
                    Button("Disabled primary") {}.buttonStyle(.rapidPrimary).disabled(true)
                        .accessibilityIdentifier("DevSnapshot.Specimen.DisabledPrimary")
                    Button("Disabled secondary") {}.buttonStyle(.rapidSecondary).disabled(true)
                        .accessibilityIdentifier("DevSnapshot.Specimen.DisabledSecondary")
                }
                Button("Wide primary") {}.buttonStyle(.rapidPrimaryWide)
                    .accessibilityIdentifier("DevSnapshot.Specimen.WidePrimary")
            }
            .settingsGroupedCard()

            SectionHeader("Switches", emphasis: .section)
            VStack(alignment: .leading, spacing: 0) {
                Toggle(isOn: $toggleOn) {
                    SettingsRowLabel(
                        title: "On, with a description",
                        description: "A three-line description exists to prove the switch stays pinned to the title line and never gets crowded by the copy beside it, however long that copy runs."
                    )
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityIdentifier("DevSnapshot.Specimen.ToggleOn")
                SettingsRowDivider()
                Toggle(isOn: $toggleOff) {
                    SettingsRowLabel(title: "Off, no description")
                }
                .toggleStyle(TrailingSettingsToggleStyle())
                .accessibilityIdentifier("DevSnapshot.Specimen.ToggleOff")
            }
            .settingsGroupedCard()

            SectionHeader("Segmented", emphasis: .section)
            VStack(alignment: .leading, spacing: RapidTheme.Space.md) {
                RapidSegmentedControl(
                    selection: $segA,
                    options: [
                        .init(value: .first, title: "Chat models"),
                        .init(value: .second, title: "Audio models"),
                    ],
                    accessibilityLabel: "Two-way"
                )
                RapidSegmentedControl(
                    selection: $segB,
                    options: [
                        .init(value: .first, title: "All"),
                        .init(value: .second, title: "Cached"),
                        .init(value: .third, title: "Not cached"),
                    ],
                    accessibilityLabel: "Three-way"
                )
            }
            .settingsGroupedCard()
            Spacer(minLength: 0)
        }
        .padding(RapidTheme.Space.xl)
        .frame(width: 560, alignment: .leading)
        .background(RapidTheme.surfaceCanvas)
        .tint(RapidTheme.brandAmber)
    }
}
