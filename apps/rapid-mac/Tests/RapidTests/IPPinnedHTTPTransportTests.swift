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

        try await raceTransportAgainstWatchdog(expected: .cancellation) {
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

        try await raceTransportAgainstWatchdog(expected: .deadline) {
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

    @Test("A DNS hostname pinned to IPv6 keeps its Host header unbracketed")
    func dnsHostnamePinnedToIPv6KeepsUnbracketedHostHeader() throws {
        let address = try #require(ParsedIP("2001:db8::1"))

        let hostHeader = IPPinnedHTTPTransport.hostHeader(
            host: "example.com",
            address: address,
            scheme: "https",
            port: 443
        )

        #expect(hostHeader == "example.com")
    }

    @Test("An IPv6 literal host uses the required bracket syntax")
    func ipv6LiteralHostUsesBracketSyntax() throws {
        let address = try #require(ParsedIP("2001:db8::1"))

        let hostHeader = IPPinnedHTTPTransport.hostHeader(
            host: "[2001:db8::1]",
            address: address,
            scheme: "https",
            port: 443
        )

        #expect(hostHeader == "[2001:db8::1]")
    }

    @Test("A chunk size larger than the remaining body fails closed")
    func oversizedChunkFailsClosed() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))
        let raw = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n100\r\nabc".utf8)

        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: raw, url: url)
        }
    }

    @Test("Explicit non-default ports remain in the Host authority")
    func explicitNonDefaultPortsRemainInHostAuthority() throws {
        let address = try #require(ParsedIP("2001:db8::1"))

        let httpOn443 = IPPinnedHTTPTransport.hostHeader(
            host: "example.com",
            address: address,
            scheme: "http",
            port: 443
        )
        let httpsOn80 = IPPinnedHTTPTransport.hostHeader(
            host: "example.com",
            address: address,
            scheme: "https",
            port: 80
        )

        #expect(httpOn443 == "example.com:443")
        #expect(httpsOn80 == "example.com:80")
    }

    @Test("Default ports are omitted from the Host authority")
    func defaultPortsAreOmittedFromHostAuthority() throws {
        let address = try #require(ParsedIP("2001:db8::1"))

        let httpOn80 = IPPinnedHTTPTransport.hostHeader(
            host: "example.com",
            address: address,
            scheme: "http",
            port: 80
        )
        let httpsOn443 = IPPinnedHTTPTransport.hostHeader(
            host: "example.com",
            address: address,
            scheme: "https",
            port: 443
        )

        #expect(httpOn80 == "example.com")
        #expect(httpsOn443 == "example.com")
    }

    @Test("A chunk missing terminator bytes fails closed")
    func missingChunkTerminatorFailsClosed() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))
        let missingBoth = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc".utf8)
        let missingOne = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r".utf8)

        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: missingBoth, url: url)
        }
        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: missingOne, url: url)
        }
    }

    @Test("A parseable Int-sized chunk with no body fails closed")
    func maxIntChunkFailsClosed() throws {
        let url = try #require(URL(string: "http://pinned-name.test/page"))
        let raw = Data("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n7FFFFFFFFFFFFFFF\r\n".utf8)

        #expect(throws: Error.self) {
            _ = try IPHTTPResponseParser.parse(data: raw, url: url)
        }
    }

    @Test("Percent-encoded path and query remain encoded in request-target")
    func percentEncodedRequestTargetRemainsEncoded() throws {
        let url = try #require(URL(string: "http://approved.example/a%20b?q=a%20b"))

        let target = try IPPinnedHTTPTransport.requestTarget(for: url)

        #expect(target == "/a%20b?q=a%20b")
    }

    @Test("Encoded control characters stay encoded in request-target")
    func encodedControlCharactersRemainEncodedInRequestTarget() throws {
        let url = try #require(URL(string: "http://approved.example/%0D%0AHost:%20other.internal"))

        let target = try IPPinnedHTTPTransport.requestTarget(for: url)

        #expect(target == "/%0D%0AHost:%20other.internal")
        #expect(!target.contains("\r"))
        #expect(!target.contains("\n"))
    }
}

private struct TestWatchdogDeadline: Error {}

private enum FetchRaceExpectation {
    case cancellation
    case deadline
}

private enum FetchRaceOutcome: @unchecked Sendable {
    case fetch(Error)
    case watchdog(Error)
}

/// First-result latch for the transport/watchdog race. The fetch must be
/// cancelled independently after the server observes its request; cancelling
/// a shared task group also cancels the watchdog and lets that cancellation
/// win nondeterministically.
private final class FetchRaceSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var outcome: FetchRaceOutcome?
    private var continuation: CheckedContinuation<FetchRaceOutcome, Never>?

    func wait() async -> FetchRaceOutcome {
        await withCheckedContinuation { pendingContinuation in
            lock.lock()
            if let outcome {
                lock.unlock()
                pendingContinuation.resume(returning: outcome)
                return
            }
            continuation = pendingContinuation
            lock.unlock()
        }
    }

    func signal(_ outcome: FetchRaceOutcome) {
        lock.lock()
        guard self.outcome == nil else {
            lock.unlock()
            return
        }
        self.outcome = outcome
        let pendingContinuation = continuation
        continuation = nil
        lock.unlock()
        pendingContinuation?.resume(returning: outcome)
    }
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
    expected: FetchRaceExpectation,
    fetchBody: @escaping @Sendable () async throws -> (Data, HTTPURLResponse),
    watchdogBody: @escaping @Sendable () async throws -> Void,
    settle: @escaping @Sendable () async -> Bool
) async throws {
    let signal = FetchRaceSignal()
    let fetchTask = Task {
        do {
            _ = try await fetchBody()
            signal.signal(.fetch(TestWatchdogDeadline()))
        } catch {
            signal.signal(.fetch(error))
        }
    }
    let watchdogTask = Task {
        do {
            try await watchdogBody()
            signal.signal(.watchdog(TestWatchdogDeadline()))
        } catch {
            signal.signal(.watchdog(error))
        }
    }
    let settleTask = Task {
        if await settle() {
            fetchTask.cancel()
        }
    }

    let first = await signal.wait()
    fetchTask.cancel()
    watchdogTask.cancel()
    settleTask.cancel()

    guard case .fetch(let error) = first else {
        throw TestWatchdogDeadline()
    }
    switch expected {
    case .cancellation:
        #expect(error is CancellationError, "transport should surface cancellation")
    case .deadline:
        #expect(
            !(error is CancellationError) && String(describing: error).contains("deadline"),
            "deadline should be the first observed failure"
        )
    }
}
