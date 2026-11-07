import Foundation

/// Allocator counters from /v1/status, in decimal GB (the engine divides by 1e9).
/// These overlap process memory and are not a weights/prefix-cache breakdown.
struct MetalMemorySnapshot: Decodable, Sendable, Equatable {
    let active: Double?
    let cache: Double?
    let peak: Double?

    enum CodingKeys: String, CodingKey {
        case active = "active_memory_gb"
        case cache = "cache_memory_gb"
        case peak = "peak_memory_gb"
    }

    static func label(_ value: Double?) -> String {
        guard let value, value.isFinite, value >= 0 else { return "—" }
        return String(format: "%.2f GB", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}

struct MetalMemoryClient {
    private struct Status: Decodable {
        let metal: MetalMemorySnapshot?
    }

    var session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 3
        configuration.timeoutIntervalForResource = 3
        return URLSession(configuration: configuration)
    }()

    static func request(port: Int, bearer: String) -> URLRequest? {
        guard (1...65535).contains(port), !bearer.isEmpty else { return nil }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/status")!)
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.applyRapidClientHeader()
        return request
    }

    func fetch(port: Int, bearer: String) async -> MetalMemorySnapshot? {
        guard let request = Self.request(port: port, bearer: bearer),
              let (data, response) = try? await session.data(for: request),
              let http = response as? HTTPURLResponse,
              (200...299).contains(http.statusCode)
        else { return nil }
        return (try? JSONDecoder().decode(Status.self, from: data))?.metal
    }
}
