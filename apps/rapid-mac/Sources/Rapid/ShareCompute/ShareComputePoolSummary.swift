import Foundation

// MARK: - Wire DTOs

/// `GET https://pay.quicksilverpro.io/v1/pool/summary`, verbatim.
///
/// Deliberately a separate type from the domain model below. The wire shape is
/// QuickSilver's to change and carries ids Rapid may not know; the domain model
/// is what the views are allowed to render. Keeping them apart is what makes
/// "an unknown `model_id` is safe" a compile-time property rather than a
/// convention.
///
/// The endpoint is unauthenticated by design — no provider, share, or inference
/// key may ever be attached (see ``ShareComputePoolSummaryClient``).
struct ShareComputePoolSummaryDTO: Decodable, Equatable, Sendable {
    struct Totals: Decodable, Equatable, Sendable {
        let connectedNodes: Int
        let readyNodes: Int
        let availableSlots: Int

        enum CodingKeys: String, CodingKey {
            case connectedNodes = "connected_nodes"
            case readyNodes = "ready_nodes"
            case availableSlots = "available_slots"
        }
    }

    struct Model: Decodable, Equatable, Sendable {
        let modelID: String
        let enabled: Bool
        let connectedNodes: Int
        let readyNodes: Int
        let busyNodes: Int
        let availableSlots: Int

        enum CodingKeys: String, CodingKey {
            case modelID = "model_id"
            case enabled
            case connectedNodes = "connected_nodes"
            case readyNodes = "ready_nodes"
            case busyNodes = "busy_nodes"
            case availableSlots = "available_slots"
        }
    }

    let updatedAt: Date
    let totals: Totals
    let models: [Model]

    enum CodingKeys: String, CodingKey {
        case updatedAt = "updated_at"
        case totals
        case models
    }
}

// MARK: - Domain model

/// Per-model availability, as published.
///
/// `availableSlots` is the pool's *currently free* request slots for this
/// model. It is NOT a per-model maximum concurrency and must never be labelled
/// as one.
struct ShareComputePoolModelStats: Equatable, Sendable {
    let modelID: String
    let isEnabled: Bool
    let connectedNodes: Int
    let readyNodes: Int
    let busyNodes: Int
    let availableSlots: Int
}

/// The pool summary, in the terms the UI speaks.
struct ShareComputePoolSummary: Equatable, Sendable {
    /// Server-published totals. Used AS PUBLISHED — never re-derived by summing
    /// ``models``. The server may count nodes the models array does not
    /// enumerate (a model Rapid does not support, a model mid-rollout), so a
    /// local sum would silently under-report the pool.
    struct Totals: Equatable, Sendable {
        let connectedNodes: Int
        let readyNodes: Int
        let availableSlots: Int
    }

    let updatedAt: Date
    let totals: Totals
    /// Every row the server sent, including ids Rapid has no catalog entry for.
    /// Unknown ids are carried, not dropped: they still count toward what is
    /// online, and dropping them here would make a future catalog addition look
    /// like a client bug.
    let models: [ShareComputePoolModelStats]

    /// Stats for one catalog id, or `nil` when the server did not report it.
    ///
    /// `nil` is meaningfully different from an all-zero row: zero means "this
    /// model is published and nobody is serving it", `nil` means "the pool said
    /// nothing about this model". Callers must render the two differently —
    /// `—` for `nil`, `0` for zero.
    func stats(for catalogID: String) -> ShareComputePoolModelStats? {
        models.first { $0.modelID == catalogID }
    }

    /// Catalog ids the pool currently has switched on.
    var enabledModelIDs: [String] {
        models.filter(\.isEnabled).map(\.modelID)
    }

    /// Every node offline and every slot taken — a real, healthy state the
    /// production pool reports routinely. It is NOT an error and must not be
    /// rendered as one.
    var isPoolEmpty: Bool {
        totals.connectedNodes == 0 && totals.readyNodes == 0 && totals.availableSlots == 0
    }

    init(updatedAt: Date, totals: Totals, models: [ShareComputePoolModelStats]) {
        self.updatedAt = updatedAt
        self.totals = totals
        self.models = models
    }

    init(dto: ShareComputePoolSummaryDTO) {
        self.updatedAt = dto.updatedAt
        self.totals = Totals(
            connectedNodes: dto.totals.connectedNodes,
            readyNodes: dto.totals.readyNodes,
            availableSlots: dto.totals.availableSlots
        )
        self.models = dto.models.map {
            ShareComputePoolModelStats(
                modelID: $0.modelID,
                isEnabled: $0.enabled,
                connectedNodes: $0.connectedNodes,
                readyNodes: $0.readyNodes,
                busyNodes: $0.busyNodes,
                availableSlots: $0.availableSlots
            )
        }
    }
}

// MARK: - Errors

enum ShareComputePoolSummaryError: Error, Equatable, Sendable {
    /// Transport failure — DNS, no route, TLS, timeout.
    case unreachable(String)
    /// The endpoint answered, but not with a 2xx. 503 while the pool service
    /// restarts is the common one.
    case httpStatus(Int)
    /// A 2xx whose body is not the documented shape.
    case malformedBody(String)

    /// One short, non-technical line for the UI. Never contains a URL, a header
    /// value, or any part of a response body.
    var displayMessage: String {
        switch self {
        case .unreachable:
            return String(localized: "Pool data unavailable")
        case .httpStatus:
            return String(localized: "Pool data unavailable")
        case .malformedBody:
            return String(localized: "Pool data unavailable")
        }
    }
}

// MARK: - Client

/// Reads the public pool summary.
///
/// Rules this type exists to enforce in one place:
///
/// * The endpoint is **fixed and HTTPS**. There is no sandbox, so there is no
///   override — a settable base URL would only ever be a way to point a desktop
///   app at somebody else's host.
/// * **No credential is ever attached.** The summary is public; sending the
///   provider, share, or inference key would leak it to a route that does not
///   need it. There is deliberately no parameter here to pass one.
/// * **Nothing is logged.** No response body, no headers, no URL. Errors carry
///   a status code or a transport description only.
/// * The transport is injected, so unit tests never touch the network.
struct ShareComputePoolSummaryClient: Sendable {
    /// The one and only endpoint. Not configurable — see the type comment.
    static let endpoint = URL(string: "https://pay.quicksilverpro.io/v1/pool/summary")!

    /// Generous enough for a cold Cloudflare edge, short enough that a hung
    /// socket cannot stall the Live Pool tab past one refresh interval.
    static let timeout: TimeInterval = 10

    /// Performs one request. Injected so tests exercise decoding and status
    /// handling with no network.
    var transport: @Sendable (URLRequest) async throws -> (Data, URLResponse) = {
        try await URLSession.shared.data(for: $0)
    }

    /// Decoder configured for the documented wire format: `snake_case` keys are
    /// handled by explicit `CodingKeys` (not a global strategy, which would also
    /// rewrite keys we never asked it to), and `updated_at` is ISO-8601.
    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let raw = try decoder.singleValueContainer().decode(String.self)
            guard let date = parseISO8601(raw) else {
                throw DecodingError.dataCorruptedError(
                    in: try decoder.singleValueContainer(),
                    debugDescription: "updated_at is not an ISO-8601 timestamp"
                )
            }
            return date
        }
        return decoder
    }

    /// `ISO8601DateFormatter` is a mutable class and not `Sendable`, so it
    /// cannot be a shared `static let`. Building one per call is cheap next to
    /// a 30-second network refresh and keeps this free of shared state.
    ///
    /// Both spellings are tried because `.withInternetDateTime` alone REJECTS
    /// fractional seconds, which the pool service emits under load — parsing
    /// only the strict form would turn a perfectly good response into a
    /// `malformedBody` every time the service got busy.
    static func parseISO8601(_ raw: String) -> Date? {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        if let date = formatter.date(from: raw) { return date }
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: raw)
    }

    func summary() async throws -> ShareComputePoolSummary {
        var request = URLRequest(url: Self.endpoint)
        request.httpMethod = "GET"
        request.timeoutInterval = Self.timeout
        // A product User-Agent is required: Cloudflare fronts every
        // quicksilverpro.io host and 403s default library UAs before the
        // request reaches the service.
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        // The service caches ~15s server-side; asking URLSession not to add its
        // own layer on top keeps "Updated … ago" honest.
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await transport(request)
        } catch {
            // `localizedDescription` of a URLError never contains the body; the
            // URL it may contain is a public, credential-free endpoint.
            throw ShareComputePoolSummaryError.unreachable(error.localizedDescription)
        }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw ShareComputePoolSummaryError.httpStatus(http.statusCode)
        }

        do {
            let dto = try Self.makeDecoder().decode(ShareComputePoolSummaryDTO.self, from: data)
            return ShareComputePoolSummary(dto: dto)
        } catch {
            // Deliberately does NOT embed the body — a malformed response is
            // still a response, and echoing it into an error string is how
            // bodies end up in logs.
            throw ShareComputePoolSummaryError.malformedBody("unreadable pool summary")
        }
    }

    private static var userAgent: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        return "Rapid/\(version ?? "unknown") (macOS)"
    }
}

// MARK: - Page state

/// What Live Pool knows right now.
///
/// Five cases, because the four-case version people reach for first cannot
/// express the two that matter most: a refresh in flight over data already on
/// screen, and a refresh that failed while that data is still the best truth
/// available. Collapsing either into `unavailable` is what makes a pool blink
/// empty every 30 seconds.
enum ShareComputePoolSummaryState: Equatable, Sendable {
    /// First load, nothing on screen yet.
    case loading
    /// Fresh data. Includes a genuine all-zero pool, which is success.
    case loaded(ShareComputePoolSummary)
    /// A refresh is in flight and the previous data is still displayed.
    case refreshing(ShareComputePoolSummary)
    /// No data and no previous data — the only case that shows an error.
    case unavailable(ShareComputePoolSummaryError)
    /// A refresh failed but the previous data stands. Values and the previous
    /// `updated_at` keep showing; the failure is a quiet note, not an error
    /// state, and the stale values are NEVER overwritten with zeros.
    case refreshFailed(ShareComputePoolSummary, ShareComputePoolSummaryError)

    /// The summary to render, if any. The single accessor every view uses, so
    /// no view has to re-derive "do I have something to show?".
    var summary: ShareComputePoolSummary? {
        switch self {
        case .loading, .unavailable:
            return nil
        case .loaded(let summary), .refreshing(let summary), .refreshFailed(let summary, _):
            return summary
        }
    }

    var isLoading: Bool {
        if case .loading = self { return true }
        return false
    }

    var isRefreshing: Bool {
        if case .refreshing = self { return true }
        return false
    }

    /// The error to SHOW. Present only when there is nothing else on screen —
    /// a failed refresh over good data is reported as ``staleNote``, not here.
    var blockingError: ShareComputePoolSummaryError? {
        if case .unavailable(let error) = self { return error }
        return nil
    }

    /// A quiet line for a refresh that failed over data still worth showing.
    var staleNote: String? {
        guard case .refreshFailed = self else { return nil }
        return String(localized: "Couldn’t refresh just now — showing the last reading.")
    }

    /// The state to move to when a load begins, preserving anything already on
    /// screen so the page cannot flash empty.
    func beginningLoad() -> Self {
        if let summary { return .refreshing(summary) }
        return .loading
    }

    /// The state to move to when a load fails, preserving anything already on
    /// screen. A failure NEVER replaces real values with zeros.
    func failing(_ error: ShareComputePoolSummaryError) -> Self {
        if let summary { return .refreshFailed(summary, error) }
        return .unavailable(error)
    }
}

// MARK: - Refresh pacing

/// How often Live Pool may re-read the summary.
///
/// The service caches ~15s and rate-limits by IP, so polling faster than the
/// cache buys identical bytes and spends the caller's rate-limit budget. 30s is
/// the recommended interval: comfortably past the cache, well inside the window
/// where "Updated 2m ago" would start to feel wrong.
enum ShareComputePoolRefresh {
    /// Contractual floor. A shorter interval is a bug, not a preference.
    static let minimumInterval: TimeInterval = 15
    static let recommendedInterval: TimeInterval = 30

    /// Clamps a requested interval up to the floor.
    static func interval(_ requested: TimeInterval) -> TimeInterval {
        max(minimumInterval, requested)
    }
}

// MARK: - Relative timestamps

enum ShareComputePoolClock {
    /// `Updated 2m ago`, from the server's own `updated_at`.
    ///
    /// Always formats the PUBLISHED time, never the time Rapid fetched: when a
    /// refresh fails, the label must keep telling the truth about how old the
    /// numbers on screen are.
    static func updatedLabel(_ updatedAt: Date, now: Date = Date()) -> String {
        let seconds = max(0, now.timeIntervalSince(updatedAt))
        let phrase: String
        switch seconds {
        case ..<10:
            phrase = String(localized: "just now")
        case ..<60:
            phrase = String(format: String(localized: "%ds ago"), Int(seconds))
        case ..<3_600:
            phrase = String(format: String(localized: "%dm ago"), Int(seconds / 60))
        case ..<86_400:
            phrase = String(format: String(localized: "%dh ago"), Int(seconds / 3_600))
        default:
            phrase = String(format: String(localized: "%dd ago"), Int(seconds / 86_400))
        }
        return String(format: String(localized: "Updated %@"), phrase)
    }
}
