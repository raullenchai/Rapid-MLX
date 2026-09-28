import Darwin
import Foundation
import Testing
@testable import Rapid

@Suite("Computer Use model-free server")
struct CUAServerManagerTests {
    @Test("Launch arguments select CUA-only mode without a model")
    func argumentsAreModelFree() {
        let arguments = CUAServerManager.serveArguments(host: "127.0.0.1", port: 7_659)
        #expect(arguments.prefix(2) == ["serve", "--cua-only"])
        #expect(arguments.contains("127.0.0.1"))
        #expect(arguments.contains("7659"))
        #expect(!arguments.contains(where: { $0 == "--model" }))
        #expect(!arguments.contains(where: { $0.contains("qwen") }))
        #expect(!Set(CUAServerManager.candidatePorts).isEmpty)
    }

    @Test("An occupied chat port is skipped without sweeping its listener")
    func occupiedPortIsNeverSwept() throws {
        let listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
        #expect(listener >= 0)
        defer { close(listener) }
        var reuse: Int32 = 1
        setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(listener, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        try #require(bound == 0)
        try #require(listen(listener, 1) == 0)
        var liveAddress = sockaddr_in()
        var liveAddressLength = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &liveAddress) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(listener, $0, &liveAddressLength)
            }
        }
        try #require(named == 0)
        let occupiedPort = Int(UInt16(bigEndian: liveAddress.sin_port))
        let freePort = try #require((60_000...60_100).first {
            PortAllocator.canBind(port: $0, host: "127.0.0.1")
        })

        let selected = CUAServerManager.selectPort(
            candidates: [occupiedPort, freePort], reserved: []
        ) { PortAllocator.canBind(port: $0, host: "127.0.0.1") }

        #expect(selected == freePort)
        #expect(fcntl(listener, F_GETFD) != -1)
        #expect(!PortAllocator.canBind(port: occupiedPort, host: "127.0.0.1"))
    }

    @Test("Repeated panel appearances share one model-free child")
    @MainActor
    func repeatedEnsureRunsOnce() async {
        var launches = 0
        var capturedArguments: [String] = []
        var capturedEnvironment: [String: String] = [:]
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: {
                try? await Task.sleep(nanoseconds: 20_000_000)
                return 7_659
            },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in true },
            launcher: { _, arguments, environment, _, _, _ in
                launches += 1
                capturedArguments = arguments
                capturedEnvironment = environment
                return ProcessGroupChild.testStub()
            }
        )

        async let first: Void = manager.ensureRunning()
        async let second: Void = manager.ensureRunning()
        _ = await (first, second)

        #expect(launches == 1)
        #expect(manager.state == .ready)
        #expect(manager.port == 7_659)
        #expect(manager.client != nil)
        #expect(capturedArguments.first == "serve")
        #expect(capturedArguments.dropFirst().first == "--cua-only")
        #expect(capturedEnvironment["RAPID_MLX_API_KEY"] == "secret")
        #expect(capturedEnvironment["RAPID_MLX_WATCHDOG_PPID"] != nil)

        await manager.ensureRunning()
        #expect(launches == 1)
    }

    @Test("Failed readiness is recoverable without duplicate retry children")
    @MainActor
    func failureCanRetry() async {
        var launches = 0
        var probes = 0
        let manager = CUAServerManager(
            binaryPath: URL(fileURLWithPath: "/usr/bin/true"),
            portProvider: { 7_660 },
            bearerProvider: { "secret" },
            readinessProbe: { _, _, _ in
                probes += 1
                return probes > 1
            },
            launcher: { _, _, _, _, _, _ in
                launches += 1
                return ProcessGroupChild.testStub()
            }
        )

        await manager.ensureRunning()
        #expect(manager.failureMessage?.contains("didn't become ready") == true)
        #expect(manager.client == nil)

        async let first: Void = manager.retry()
        async let second: Void = manager.retry()
        _ = await (first, second)

        #expect(launches == 2)
        #expect(manager.state == .ready)
        #expect(manager.client != nil)
    }

    @Test("Missing engine fails without selecting or downloading a model")
    @MainActor
    func missingEngineFailsClosed() async {
        var launches = 0
        let manager = CUAServerManager(
            binaryPath: nil,
            binaryRefresh: { nil },
            launcher: { _, _, _, _, _, _ in
                launches += 1
                return ProcessGroupChild.testStub()
            }
        )

        await manager.ensureRunning()

        #expect(launches == 0)
        #expect(manager.client == nil)
        #expect(manager.failureMessage?.contains("unavailable") == true)
    }
}
