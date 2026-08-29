import Foundation
import Network
import Testing
@testable import Rapid

@Suite("IP-pinned browse transport")
struct IPPinnedHTTPTransportTests {
    @Test("Chunked HTTP/1.1 bodies are decoded once")
    func chunkedResponseBodiesAreDecoded() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))
        let raw = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n4\r\n,123\r\n0\r\n\r\n".utf8)

        let (body, response) = try IPHTTPResponseParser.parse(data: raw, url: url)

        #expect(String(decoding: body, as: UTF8.self) == "abc,123")
        #expect(response.statusCode == 200)
        #expect(response.value(forHTTPHeaderField: "Transfer-Encoding") == "chunked")
    }

    @Test("A response without a complete header block fails closed")
    func incompleteHeaderResponseFailsClosed() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))

        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: Data("HTTP/1.1 200 OK\r\n".utf8), url: url)
        }
    }

    @Test("Cancelling an open HTTP request resumes the caller")
    func cancellingOpenRequestResumesCaller() async throws {
        let server = try #require(HangingTCPServer())
        defer { server.stop() }
        let port = try #require(server.port)
        let url = try #require(URL(string: "http://pinned-cancel.test:\(port)/request"))
        let address = try #require(ParsedIP("127.0.0.1"))

        try await raceTransportAgainstWatchdog {
            try await IPPinnedHTTPTransport.fetch(
                url: url,
                address: address,
                byteLimit: 1024 * 1024
            )
        } watchdogBody: { @Sendable in
            try await Task.sleep(for: .seconds(2))
            throw TestWatchdogDeadline()
        } settle: {
            await server.waitForRequest()
            return true
        }
    }

    @Test("An HTTP request deadline cancels the pinned transport and resumes")
    func httpDeadlineCancelsPinnedTransportAndResumes() async throws {
        let server = try #require(HangingTCPServer())
        defer { server.stop() }
        let port = try #require(server.port)
        let url = try #require(URL(string: "http://pinned-deadline.test:\(port)/request"))
        let address = try #require(ParsedIP("127.0.0.1"))

        try await raceTransportAgainstWatchdog {
            try await BrowseTool.withDeadline(0.2) {
                try await IPPinnedHTTPTransport.fetch(
                    url: url,
                    address: address,
                    byteLimit: 1024 * 1024
                )
            }
        } watchdogBody: { @Sendable in
            try await Task.sleep(for: .seconds(2))
            throw TestWatchdogDeadline()
        } settle: {
            await server.waitForRequest()
            return false
        }
    }
}

private struct TestWatchdogDeadline: Error {}

private enum FetchRaceOutcome {
    case fetch(Error)
    case watchdog(Error)
}

/// Accepts a request and deliberately does not respond, so cancellation has
/// to settle the checked continuation used by the pinned transport.
private final class HangingTCPServer: @unchecked Sendable {
    private let listener: NWListener
    private let requestSignal = RequestSignal()
    private(set) var port: UInt16?

    init?() {
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: .any)
        guard let listener = try? NWListener(using: parameters) else { return nil }
        self.listener = listener
        port = nil

        let ready = DispatchSemaphore(value: 0)
        let requestSignal = requestSignal
        listener.newConnectionHandler = { connection in
            connection.start(queue: .global())
            connection.receive(minimumIncompleteLength: 1, maximumLength: 8 * 1024) {
                data, _, _, _ in
                if let data,
                   Data(data).range(of: Data("\r\n\r\n".utf8)) != nil {
                    requestSignal.signal()
                }
            }
        }

        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed: ready.signal()
            default: break
            }
        }

        listener.start(queue: .global())
        guard ready.wait(timeout: .now() + 2) == .success else {
            listener.cancel()
            return nil
        }
        port = listener.port?.rawValue
    }

    func waitForRequest() async {
        await requestSignal.wait()
    }

    func stop() {
        listener.cancel()
    }
}

private final class RequestSignal: @unchecked Sendable {
    private var continuation: CheckedContinuation<Void, Never>?
    private var acknowledged = false
    private let lock = NSLock()

    func wait() async {
        await withCheckedContinuation { pendingContinuation in
            lock.lock()
            if acknowledged {
                lock.unlock()
                pendingContinuation.resume()
                return
            }
            continuation = pendingContinuation
            lock.unlock()
        }
    }

    func signal() {
        lock.lock()
        let pendingContinuation = continuation
        continuation = nil
        acknowledged = true
        lock.unlock()
        pendingContinuation?.resume()
    }
}

private func raceTransportAgainstWatchdog(
    fetchBody: @escaping @Sendable () async throws -> (Data, HTTPURLResponse),
    watchdogBody: @escaping @Sendable () async throws -> Void,
    settle: @escaping @Sendable () async -> Bool
) async throws {
    try await withThrowingTaskGroup(of: FetchRaceOutcome.self) { group in
        group.addTask {
            do {
                _ = try await fetchBody()
                return .fetch(TestWatchdogDeadline())
            } catch {
                return .fetch(error)
            }
        }
        group.addTask {
            do {
                try await watchdogBody()
                return .watchdog(TestWatchdogDeadline())
            } catch {
                return .watchdog(error)
            }
        }

        let cancelAfterSettled = await settle()
        if cancelAfterSettled {
            group.cancelAll()
        }

        guard let first = try await group.next() else {
            throw TestWatchdogDeadline()
        }
        group.cancelAll()
        guard case .fetch(let error) = first else {
            throw TestWatchdogDeadline()
        }
        if cancelAfterSettled {
            #expect(error is CancellationError, "transport should surface cancellation")
        } else {
            #expect(
                String(describing: error).contains("deadline"),
                "deadline should be the first observed failure"
            )
        }
    }
}
