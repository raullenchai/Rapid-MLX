import Foundation
import Testing
@testable import Rapid

struct MetalMemorySnapshotTests {
    @Test func fetchHandlesMissingUnsupportedAndFailedResponses() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MetalMemoryTestProtocol.self]
        let client = MetalMemoryClient(session: URLSession(configuration: configuration))
        let complete = await client.fetch(port: 7660, bearer: "complete")
        #expect(complete?.active == 1.25)
        #expect(complete?.cache == nil)
        #expect(complete?.numRunning == 2)
        #expect(complete?.numWaiting == 3)
        #expect(complete?.prefixCache?.currentBytes == 1_250_000_000)
        #expect(complete?.prefixCache?.maxBytes == 5_000_000_000)
        #expect(complete?.prefixCache?.utilization == 0.25)
        #expect(PrefixCacheSnapshot.label(complete?.prefixCache) == "1.25 GB / 5.00 GB")
        for scenario in ["missing", "unauthorized", "malformed"] {
            let result = await client.fetch(port: 7660, bearer: scenario)
            #expect(result == nil)
        }
    }

    @Test func statusSignalsDoNotRequireMetalCounters() throws {
        let data = Data(#"{"num_running":0,"num_waiting":4,"cache":{"current_memory_bytes":0,"max_memory_bytes":1000000000}}"#.utf8)
        let snapshot = try #require(try MetalMemoryClient.decodeStatus(data))
        #expect(snapshot.active == nil && snapshot.cache == nil && snapshot.peak == nil)
        #expect(MemoryOverviewFormat.requestCount(snapshot.numRunning) == "0")
        #expect(MemoryOverviewFormat.requestCount(snapshot.numWaiting) == "4")
        #expect(snapshot.prefixCache?.utilization == 0)
        #expect(PrefixCacheSnapshot.label(snapshot.prefixCache) == "0.00 GB / 1.00 GB")
    }

    @Test func unsupportedCacheBudgetsAndMissingCountsUseDash() throws {
        let cacheObjects = [
            "null", "{}", #"{"enabled":false}"#,
            #"{"current_memory_bytes":10}"#,
            #"{"max_memory_bytes":100}"#,
            #"{"current_memory_bytes":null,"max_memory_bytes":100}"#,
            #"{"current_memory_bytes":10,"max_memory_bytes":null}"#,
            #"{"current_memory_bytes":10,"max_memory_bytes":0}"#,
            #"{"current_memory_bytes":-1,"max_memory_bytes":100}"#,
            #"{"current_memory_bytes":10,"max_memory_bytes":-1}"#,
            #"{"enabled":false,"current_memory_bytes":10,"max_memory_bytes":100}"#,
            #"{"used_blocks":4,"max_blocks":16}"#,
            #"{"current_memory_mb":10,"max_memory_mb":100}"#
        ]
        for cache in cacheObjects {
            let data = Data("{\"metal\":{\"active_memory_gb\":1.25},\"cache\":\(cache)}".utf8)
            let snapshot = try #require(try MetalMemoryClient.decodeStatus(data))
            #expect(snapshot.active == 1.25)
            #expect(snapshot.prefixCache?.utilization == nil)
            #expect(PrefixCacheSnapshot.label(snapshot.prefixCache) == "—")
            #expect(MemoryOverviewFormat.requestCount(snapshot.numRunning) == "—")
            #expect(MemoryOverviewFormat.requestCount(snapshot.numWaiting) == "—")
        }
        for body in [#"{"status":"not_loaded","requests":[]}"#,
                     #"{"num_running":null,"num_waiting":null,"metal":{}}"#,
                     #"{"metal":{}}"#] {
            let snapshot = try MetalMemoryClient.decodeStatus(Data(body.utf8))
            #expect(PrefixCacheSnapshot.label(snapshot?.prefixCache) == "—")
            #expect(MemoryOverviewFormat.requestCount(snapshot?.numRunning) == "—")
            #expect(MemoryOverviewFormat.requestCount(snapshot?.numWaiting) == "—")
        }
    }

    @Test func cacheOverCapacityAndInvalidValuesStayHonest() throws {
        let data = Data(#"{"num_running":-1,"num_waiting":0,"cache":{"current_memory_bytes":2000000000,"max_memory_bytes":1000000000}}"#.utf8)
        let snapshot = try #require(try MetalMemoryClient.decodeStatus(data))
        #expect(snapshot.prefixCache?.utilization == 2)
        #expect(PrefixCacheSnapshot.label(snapshot.prefixCache) == "2.00 GB / 1.00 GB")
        #expect(MemoryOverviewFormat.requestCount(snapshot.numRunning) == "—")
        #expect(MemoryOverviewFormat.requestCount(snapshot.numWaiting) == "0")
        for invalid in [Double.nan, .infinity, -1] {
            #expect(PrefixCacheSnapshot(currentBytes: invalid, maxBytes: 100, enabled: nil).utilization == nil)
            #expect(PrefixCacheSnapshot(currentBytes: 10, maxBytes: invalid, enabled: nil).utilization == nil)
        }
        #expect(PrefixCacheSnapshot(currentBytes: .greatestFiniteMagnitude, maxBytes: .leastNonzeroMagnitude,
                                    enabled: nil).utilization == nil)
    }

    @Test func decimalGBAndUnavailableValues() throws {
        let data = Data(#"{"active_memory_gb":1.25,"cache_memory_gb":null,"peak_memory_gb":2.5}"#.utf8)
        let snapshot = try JSONDecoder().decode(MetalMemorySnapshot.self, from: data)
        #expect(MetalMemorySnapshot.label(snapshot.active) == "1.25 GB")
        #expect(MetalMemorySnapshot.label(snapshot.cache) == "—")
        #expect(MetalMemorySnapshot.label(snapshot.peak) == "2.50 GB")
        #expect(MetalMemorySnapshot.label(0) == "0.00 GB")
        for value in [-1.0, Double.infinity, Double.nan] {
            #expect(MetalMemorySnapshot.label(value) == "—")
        }
        let absent = try JSONDecoder().decode(MetalMemorySnapshot.self, from: Data("{}".utf8))
        #expect(absent.active == nil && absent.cache == nil && absent.peak == nil)
    }

    @Test func requestUsesAuthenticatedLoopbackAndActualPort() throws {
        let request = try #require(MetalMemoryClient.request(port: 7660, bearer: "test-only"))
        #expect(request.url?.absoluteString == "http://127.0.0.1:7660/v1/status")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer test-only")
        #expect(MetalMemoryClient.request(port: 7660, bearer: "") == nil)
        #expect(MetalMemoryClient.request(port: 0, bearer: "test-only") == nil)
    }
}

private final class MetalMemoryTestProtocol: URLProtocol, @unchecked Sendable {
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        let scenario = request.value(forHTTPHeaderField: "Authorization")
        let status = scenario == "Bearer unauthorized" ? 401 : 200
        let body: String
        switch scenario {
        case "Bearer complete", "Bearer unauthorized":
            body = #"{"metal":{"active_memory_gb":1.25},"num_running":2,"num_waiting":3,"cache":{"current_memory_bytes":1250000000,"max_memory_bytes":5000000000}}"#
        case "Bearer malformed": body = "invalid json"
        default: body = "{}"
        }
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
