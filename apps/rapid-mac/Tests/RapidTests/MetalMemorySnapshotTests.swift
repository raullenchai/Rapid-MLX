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
        for scenario in ["missing", "unauthorized", "malformed"] {
            let result = await client.fetch(port: 7660, bearer: scenario)
            #expect(result == nil)
        }
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
        case "Bearer complete", "Bearer unauthorized": body = #"{"metal":{"active_memory_gb":1.25}}"#
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
