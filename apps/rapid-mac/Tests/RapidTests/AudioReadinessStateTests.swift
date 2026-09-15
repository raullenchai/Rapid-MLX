import Testing

@testable import Rapid

@Suite("Audio readiness state")
struct AudioReadinessStateTests {
    @Test(
        "selected model follows the complete readiness lifecycle",
        arguments: [
            (
                snapshot(cached: false),
                AudioReadinessState.notDownloaded(alias: alias, sizeText: "1.5 GiB")
            ),
            (
                snapshot(
                    cached: false,
                    download: .init(
                        alias: alias,
                        status: .running(detail: "30%", fraction: 0.3)
                    )),
                .downloading(alias: alias, detail: "30%", fraction: 0.3)
            ),
            (snapshot(cached: true), .downloaded(alias: alias)),
            (
                snapshot(
                    cached: true, loading: .init(alias: alias, detail: "Loading the audio model…")),
                .loading(alias: alias, detail: "Loading the audio model…")
            ),
            (snapshot(cached: true, readyAlias: alias), .ready(alias: alias)),
            (
                snapshot(
                    cached: true,
                    readyAlias: alias,
                    activity: .init(alias: alias, activity: .transcribing)
                ),
                .active(alias: alias, activity: .transcribing)
            ),
            (
                snapshot(
                    cached: false,
                    download: .init(alias: alias, status: .failed(message: "network lost"))),
                .failed(alias: alias, message: "network lost")
            ),
        ]
    )
    func lifecycle(snapshot: AudioReadinessState.Snapshot, expected: AudioReadinessState) {
        #expect(AudioReadinessState.resolve(snapshot) == expected)
    }

    @Test("catalog success is verified before downloaded is published")
    func completedPullWaitsForCatalogProof() {
        let state = AudioReadinessState.resolve(
            Self.snapshot(
                cached: false,
                download: .init(alias: Self.alias, status: .completed)
            ))

        #expect(state == AudioReadinessState.verifyingDownload(alias: Self.alias))
        #expect(
            state.modelReadinessOverride
                == ModelReadiness.starting(
                    alias: Self.alias,
                    detail: "Finishing the download…"
                ))
    }

    @Test("terminal download result outranks an activation task until retry begins")
    func failedPullIsNotHiddenByActivation() {
        let state = AudioReadinessState.resolve(
            Self.snapshot(
                cached: false,
                download: .init(alias: Self.alias, status: .failed(message: "network lost")),
                loading: .init(alias: Self.alias, detail: "Downloading or loading the audio model…")
            ))

        #expect(state == .failed(alias: Self.alias, message: "network lost"))
    }

    @Test("cancelled downloads return to a retryable not-downloaded state")
    func cancelledDownloadIsRetryable() {
        let state = AudioReadinessState.resolve(
            Self.snapshot(
                cached: false,
                download: .init(alias: Self.alias, status: .cancelled)
            ))

        #expect(state == AudioReadinessState.notDownloaded(alias: Self.alias, sizeText: "1.5 GiB"))
        #expect(
            state.modelReadinessOverride
                == ModelReadiness.needsDownload(
                    alias: Self.alias,
                    sizeText: "1.5 GiB"
                ))
    }

    @Test("events for a previous selection cannot advance the current model")
    func staleEventsAreRejected() {
        let previous = "qwen3-tts-4bit"
        let state = AudioReadinessState.resolve(
            Self.snapshot(
                cached: false,
                download: .init(alias: previous, status: .completed),
                loading: .init(alias: previous, detail: "Loading the audio model…"),
                readyAlias: previous,
                activity: .init(alias: previous, activity: .synthesizing)
            ))

        #expect(state == AudioReadinessState.notDownloaded(alias: Self.alias, sizeText: "1.5 GiB"))
    }

    @Test("selection and catalog boundaries are explicit")
    func selectionAndCatalogBoundaries() {
        #expect(
            AudioReadinessState.resolve(
                .init(
                    alias: "",
                    catalogLoaded: true,
                    cached: nil
                )) == .noModel)
        #expect(
            AudioReadinessState.resolve(
                .init(
                    alias: Self.alias,
                    catalogLoaded: false,
                    cached: nil
                )) == .catalogPending)
        #expect(
            AudioReadinessState.resolve(
                .init(
                    alias: Self.alias,
                    catalogLoaded: true,
                    cached: nil
                )) == .unknownModel(alias: Self.alias))
    }

    @Test("model selection is blocked only by runtime-owned work")
    func modelSelectionBoundary() {
        #expect(!AudioReadinessState.loading(alias: Self.alias, detail: nil).allowsModelSelection)
        #expect(
            !AudioReadinessState.active(
                alias: Self.alias,
                activity: .recording
            ).allowsModelSelection)
        #expect(
            AudioReadinessState.downloading(
                alias: Self.alias,
                detail: nil,
                fraction: nil
            ).allowsModelSelection)
        #expect(AudioReadinessState.ready(alias: Self.alias).allowsModelSelection)
    }

    private static let alias = "whisper-medium"

    private static func snapshot(
        cached: Bool,
        download: AudioReadinessState.DownloadSnapshot? = nil,
        loading: AudioReadinessState.LoadingSnapshot? = nil,
        readyAlias: String? = nil,
        activity: AudioReadinessState.ActivitySnapshot? = nil
    ) -> AudioReadinessState.Snapshot {
        .init(
            alias: alias,
            catalogLoaded: true,
            cached: cached,
            sizeText: "1.5 GiB",
            download: download,
            loading: loading,
            readyAlias: readyAlias,
            activity: activity
        )
    }
}
