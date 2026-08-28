import Foundation
import Testing
@testable import Rapid

@Suite("Desktop activation telemetry", .serialized)
struct DesktopActivationReporterTests {
    private func temporaryDirectory(_ label: String) -> URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("rapid-desktop-activation-\(label)-\(UUID().uuidString)")
    }

    private func event(_ kind: TelemetryEvent.Activation.Kind) -> TelemetryEvent {
        TelemetryEvent(
            schema_version: 1,
            client_id: "11111111-2222-3333-4444-555555555555",
            session_id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            rapid_mlx_version: "0.13.2",
            event: .activation,
            timestamp: "2026-08-26T17:00:00Z",
            platform: .init(
                app: "rapid-desktop",
                os: "macos",
                os_version: "26.0.0",
                arch: "arm64"
            ),
            error_type: nil,
            error_message: nil,
            stack_frames: nil,
            context: nil,
            activation: .init(activation_kind: kind, surface: .desktop)
        )
    }

    @Test("Activation wire reuses the deployed two-field activation payload")
    func activationWireShape() throws {
        let data = try JSONEncoder().encode(event(.firstChatReply))
        let json = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        #expect(json["event"] as? String == "activation")
        let activation = try #require(json["activation"] as? [String: Any])
        #expect(activation["activation_kind"] as? String == "first_chat_reply")
        #expect(activation["surface"] as? String == "desktop")
        #expect(Set(activation.keys) == ["activation_kind", "surface"])
        #expect(json["request"] == nil)
        #expect(json["session"] == nil)
    }

    @Test("Declined consent touches no event, network, identity, marker, or directory")
    func disabledDoesNothing() async {
        let directory = temporaryDirectory("disabled")
        let probe = ActivationReporterProbe()
        let reporter = DesktopActivationReporter(
            isEnabled: { false },
            buildEvent: { kind in
                probe.didBuild()
                return event(kind)
            },
            sendEvent: { event in
                await probe.didSend(event)
                return .accepted
            },
            markerDirectory: directory
        )

        await reporter.report(.firstChatReply)

        #expect(probe.buildCount == 0)
        #expect(await probe.sentCount == 0)
        #expect(!FileManager.default.fileExists(atPath: directory.path))
    }

    @Test("Accepted activation sends once and a new process observes its marker")
    func acceptedThenMarked() async throws {
        let directory = temporaryDirectory("accepted")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe()

        func makeReporter() -> DesktopActivationReporter {
            DesktopActivationReporter(
                isEnabled: { true },
                buildEvent: { event($0) },
                sendEvent: { event in
                    await probe.didSend(event)
                    return .accepted
                },
                markerDirectory: directory
            )
        }

        let first = makeReporter()
        await first.report(.firstDictation)
        await first.report(.firstDictation)
        #expect(await probe.sentCount == 1)

        let second = makeReporter()
        await second.report(.firstDictation)
        #expect(await probe.sentCount == 1)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_dictation")
                    .path
            )
        )
    }

    @Test("Transport failure leaves the activation retryable")
    func failedSendRetries() async {
        let directory = temporaryDirectory("retry")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe(results: [.retry, .accepted])
        let reporter = DesktopActivationReporter(
            isEnabled: { true },
            buildEvent: { event($0) },
            sendEvent: { event in await probe.nextResult(for: event) },
            markerDirectory: directory
        )

        await reporter.report(.firstImage)
        await reporter.report(.firstImage)

        #expect(await probe.sentCount == 2)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_image")
                    .path
            )
        )
    }

    @Test("A concurrent success retries an in-flight transport failure", .timeLimit(.minutes(1)))
    func concurrentSuccessRetriesFailedInFlightSend() async {
        let directory = temporaryDirectory("concurrent-retry")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = InFlightActivationRetryProbe()
        let reporter = DesktopActivationReporter(
            isEnabled: { true },
            buildEvent: { event($0) },
            sendEvent: { event in await probe.send(event) },
            markerDirectory: directory
        )

        let first = Task { await reporter.report(.firstImage) }
        await probe.waitUntilFirstSendStarted()
        // Actor reentrancy admits this call while the first transport awaits.
        // It must become a retry opportunity rather than disappear.
        await reporter.report(.firstImage)
        await probe.failFirstSend()
        await first.value

        #expect(await probe.sentCount == 2)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_image")
                    .path
            )
        )
    }

    @Test("A permanent rejection is suppressed for this process without claiming the accepted marker")
    func rejectedSendDoesNotRepeatOrClaim() async {
        let directory = temporaryDirectory("rejected")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe()
        let reporter = DesktopActivationReporter(
            isEnabled: { true },
            buildEvent: { event($0) },
            sendEvent: { event in
                await probe.didSend(event)
                return .discard
            },
            markerDirectory: directory
        )

        await reporter.report(.firstChatReply)
        await reporter.report(.firstChatReply)

        #expect(await probe.sentCount == 1)
        #expect(
            !FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_chat_reply")
                    .path
            )
        )
    }

    @Test("Consent revoked during a discarded send stays retryable after Settings opt-in")
    func revokedDiscardCanRetry() async {
        let directory = temporaryDirectory("discarded-revoked")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe(results: [.discard, .accepted])
        let reporter = DesktopActivationReporter(
            isEnabled: { probe.isEnabled },
            buildEvent: { event($0) },
            sendEvent: { event in
                let result = await probe.nextResult(for: event)
                if result == .discard { probe.isEnabled = false }
                return result
            },
            markerDirectory: directory
        )

        await reporter.report(.firstChatReply)
        probe.isEnabled = true
        await reporter.report(.firstChatReply)

        #expect(await probe.sentCount == 2)
        #expect(
            FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_chat_reply")
                    .path
            )
        )
    }

    @Test("Consent revoked during send never burns the once marker")
    func revokeDuringSendDoesNotClaim() async {
        let directory = temporaryDirectory("revoked")
        defer { try? FileManager.default.removeItem(at: directory) }
        let probe = ActivationReporterProbe()
        let reporter = DesktopActivationReporter(
            isEnabled: { probe.isEnabled },
            buildEvent: { event($0) },
            sendEvent: { event in
                await probe.didSend(event)
                probe.isEnabled = false
                return .accepted
            },
            markerDirectory: directory
        )

        await reporter.report(.firstChatReply)

        #expect(await probe.sentCount == 1)
        #expect(
            !FileManager.default.fileExists(
                atPath: directory
                    .appendingPathComponent("activation_seen_desktop_first_chat_reply")
                    .path
            )
        )
    }
}

private actor InFlightActivationRetryProbe {
    private var sends = 0
    private var firstSendStarted = false
    private var startWaiters: [CheckedContinuation<Void, Never>] = []
    private var firstDelivery: CheckedContinuation<TelemetryClient.BatchDelivery, Never>?

    var sentCount: Int { sends }

    func send(_ event: TelemetryEvent) async -> TelemetryClient.BatchDelivery {
        _ = event
        sends += 1
        guard sends == 1 else { return .accepted }
        return await withCheckedContinuation { continuation in
            firstDelivery = continuation
            firstSendStarted = true
            let waiters = startWaiters
            startWaiters.removeAll()
            for waiter in waiters { waiter.resume() }
        }
    }

    func waitUntilFirstSendStarted() async {
        guard !firstSendStarted else { return }
        await withCheckedContinuation { continuation in
            startWaiters.append(continuation)
        }
    }

    func failFirstSend() {
        firstDelivery?.resume(returning: .retry)
        firstDelivery = nil
    }
}

private final class ActivationReporterProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var builds = 0
    private var sent: [TelemetryEvent] = []
    private var results: [TelemetryClient.BatchDelivery]
    private var enabled = true

    init(results: [TelemetryClient.BatchDelivery] = []) {
        self.results = results
    }

    var buildCount: Int {
        lock.withLock { builds }
    }

    var sentCount: Int {
        get async { lock.withLock { sent.count } }
    }

    var isEnabled: Bool {
        get { lock.withLock { enabled } }
        set { lock.withLock { enabled = newValue } }
    }

    func didBuild() {
        lock.withLock { builds += 1 }
    }

    func didSend(_ event: TelemetryEvent) async {
        lock.withLock { sent.append(event) }
    }

    func nextResult(for event: TelemetryEvent) async -> TelemetryClient.BatchDelivery {
        lock.withLock {
            sent.append(event)
            return results.isEmpty ? .accepted : results.removeFirst()
        }
    }
}
