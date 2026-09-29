import Foundation

// MARK: - Wire DTOs

/// `GET https://pay.quicksilverpro.io/v1/pool/ledger`, verbatim.
///
/// Separate from the domain model below for the same reason
/// ``ShareComputePoolSummaryDTO`` is: the wire shape is QuickSilver's to
/// change, and a status value Rapid has never heard of must not be able to
/// fail a decode.
struct ShareComputeLedgerPageDTO: Decodable, Sendable {
    struct Caps: Decodable, Sendable {
        let nodeMonthlyCapUSD: Decimal
        let poolMonthlyCapUSD: Decimal

        enum CodingKeys: String, CodingKey {
            case nodeMonthlyCapUSD = "node_monthly_cap_usd"
            case poolMonthlyCapUSD = "pool_monthly_cap_usd"
        }
    }

    struct Window: Decodable, Sendable {
        let nodeID: String
        /// Null only when the node/model record behind the window was deleted.
        let modelID: String?
        let periodStart: Date
        let periodEnd: Date
        let requestCount: Int
        let inputTokens: Int
        let outputTokens: Int
        /// Null for historical windows where the pre-cap amount is unknown.
        /// NOT zero — see ``ShareComputeLedgerWindow/accruedCredit``.
        let accruedCredit: Decimal?
        /// Non-null and >= 0 per the contract.
        let finalCredit: Decimal
        let status: String
        /// Null while pending.
        let creditedAt: Date?
        let updatedAt: Date
        let allowanceMonth: String
        let nodeMonthlyCapUSD: Decimal
        let cursor: Int

        enum CodingKeys: String, CodingKey {
            case nodeID = "node_id"
            case modelID = "model_id"
            case periodStart = "period_start"
            case periodEnd = "period_end"
            case requestCount = "request_count"
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
            case accruedCredit = "accrued_credit"
            case finalCredit = "final_credit"
            case status
            case creditedAt = "credited_at"
            case updatedAt = "updated_at"
            case allowanceMonth = "allowance_month"
            case nodeMonthlyCapUSD = "node_monthly_cap_usd"
            case cursor
        }
    }

    let unit: String
    let note: String?
    let caps: Caps
    let ledger: [Window]
    /// Null on the final page.
    let nextCursor: Int?

    enum CodingKeys: String, CodingKey {
        case unit, note, caps, ledger
        case nextCursor = "next_cursor"
    }
}

/// `{"error":{"code","message"}}`.
struct ShareComputeLedgerErrorDTO: Decodable, Sendable {
    struct Body: Decodable, Sendable {
        let code: String
        let message: String?
    }
    let error: Body
}

// MARK: - Domain model

/// Settlement state of one ledger window.
///
/// `unknown` is load-bearing: the contract explicitly says unrecognised values
/// (it names `void`) must decode rather than fail. A closed enum here would
/// turn a backend adding a status into a blank Credits tab for everyone.
enum ShareComputeLedgerStatus: Equatable, Sendable {
    case pending
    case credited
    case zero
    case unknown(String)

    init(rawValue: String) {
        switch rawValue.lowercased() {
        case "pending": self = .pending
        case "credited": self = .credited
        case "zero": self = .zero
        default: self = .unknown(rawValue)
        }
    }

    var rawValue: String {
        switch self {
        case .pending: return "pending"
        case .credited: return "credited"
        case .zero: return "zero"
        case .unknown(let raw): return raw
        }
    }

    /// Short label for the status column. An unknown value is shown verbatim
    /// (uppercased by the view) rather than mapped onto a status Rapid invented.
    var title: String {
        switch self {
        case .pending: return String(localized: "Pending")
        case .credited: return String(localized: "Credited")
        case .zero: return String(localized: "Zero")
        case .unknown(let raw): return raw
        }
    }

    /// Whether QuickSilver has finished accounting for this window.
    var isSettled: Bool {
        switch self {
        case .credited, .zero: return true
        case .pending, .unknown: return false
        }
    }
}

/// One accounting window, as QuickSilver reports it.
///
/// Deliberately NOT called a session, job, or payout. A window is a fixed
/// half-hour slice of metered usage for one (node, model); it has no
/// relationship to a Rapid local connection receipt, and the two must never be
/// correlated — there is no shared identifier that would make such a join true.
struct ShareComputeLedgerWindow: Identifiable, Equatable, Sendable {
    let nodeID: String
    /// `nil` when the node/model record was deleted. Rendered as a neutral
    /// "model unavailable" label, never as a guess at which model it was.
    let modelID: String?
    let periodStart: Date
    let periodEnd: Date
    let requestCount: Int
    let inputTokens: Int
    let outputTokens: Int
    /// `nil` for historical windows whose pre-cap amount QuickSilver no longer
    /// knows. UNKNOWN, not zero: rendering `0` would assert the window earned
    /// nothing, which is a different and possibly false claim.
    let accruedCredit: Decimal?
    /// The authoritative amount. Always the server's value — Rapid never
    /// recomputes it from `accruedCredit` and a cap, because cap application is
    /// the backend's business and its rules are not published.
    let finalCredit: Decimal
    let status: ShareComputeLedgerStatus
    /// `nil` while pending.
    let creditedAt: Date?
    let updatedAt: Date
    /// `2026-09-01` — the monthly allowance bucket this window counts against.
    let allowanceMonth: String
    let nodeMonthlyCapUSD: Decimal
    /// Server-assigned row cursor. Also the stable identity for SwiftUI.
    let cursor: Int

    var id: Int { cursor }

    init(dto: ShareComputeLedgerPageDTO.Window) {
        nodeID = dto.nodeID
        modelID = dto.modelID
        periodStart = dto.periodStart
        periodEnd = dto.periodEnd
        requestCount = dto.requestCount
        inputTokens = dto.inputTokens
        outputTokens = dto.outputTokens
        accruedCredit = dto.accruedCredit
        finalCredit = dto.finalCredit
        status = ShareComputeLedgerStatus(rawValue: dto.status)
        creditedAt = dto.creditedAt
        updatedAt = dto.updatedAt
        allowanceMonth = dto.allowanceMonth
        nodeMonthlyCapUSD = dto.nodeMonthlyCapUSD
        cursor = dto.cursor
    }

    init(
        nodeID: String,
        modelID: String?,
        periodStart: Date,
        periodEnd: Date,
        requestCount: Int,
        inputTokens: Int,
        outputTokens: Int,
        accruedCredit: Decimal?,
        finalCredit: Decimal,
        status: ShareComputeLedgerStatus,
        creditedAt: Date?,
        updatedAt: Date,
        allowanceMonth: String,
        nodeMonthlyCapUSD: Decimal,
        cursor: Int
    ) {
        self.nodeID = nodeID
        self.modelID = modelID
        self.periodStart = periodStart
        self.periodEnd = periodEnd
        self.requestCount = requestCount
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.accruedCredit = accruedCredit
        self.finalCredit = finalCredit
        self.status = status
        self.creditedAt = creditedAt
        self.updatedAt = updatedAt
        self.allowanceMonth = allowanceMonth
        self.nodeMonthlyCapUSD = nodeMonthlyCapUSD
        self.cursor = cursor
    }
}

/// Monthly caps, as published.
struct ShareComputeLedgerCaps: Equatable, Sendable {
    let nodeMonthlyUSD: Decimal
    let poolMonthlyUSD: Decimal
}

/// One page of the ledger.
struct ShareComputeLedgerPage: Equatable, Sendable {
    let unit: String
    let note: String?
    let caps: ShareComputeLedgerCaps
    let windows: [ShareComputeLedgerWindow]
    let nextCursor: Int?

    init(dto: ShareComputeLedgerPageDTO) {
        unit = dto.unit
        note = dto.note
        caps = ShareComputeLedgerCaps(
            nodeMonthlyUSD: dto.caps.nodeMonthlyCapUSD,
            poolMonthlyUSD: dto.caps.poolMonthlyCapUSD
        )
        windows = dto.ledger.map(ShareComputeLedgerWindow.init(dto:))
        nextCursor = dto.nextCursor
    }

    init(
        unit: String,
        note: String?,
        caps: ShareComputeLedgerCaps,
        windows: [ShareComputeLedgerWindow],
        nextCursor: Int?
    ) {
        self.unit = unit
        self.note = note
        self.caps = caps
        self.windows = windows
        self.nextCursor = nextCursor
    }
}

/// Everything the Credits tab knows after following the cursor to the end.
///
/// ACCOUNT-WIDE: the read key is account-scoped, so these windows span every
/// node on the account, not just this Mac. The UI copy has to say so — a user
/// with two contributing machines would otherwise read another machine's
/// credit as this one's.
struct ShareComputeLedgerAccount: Equatable, Sendable {
    let unit: String
    let note: String?
    let caps: ShareComputeLedgerCaps
    /// Newest first, as served.
    let windows: [ShareComputeLedgerWindow]
    /// True when a page guard stopped the walk before `next_cursor` was nil,
    /// so totals below describe what was fetched rather than the whole account.
    let isTruncated: Bool

    var isEmpty: Bool { windows.isEmpty }

    /// Distinct node ids present, in first-seen order. Drives the
    /// "across N nodes" wording.
    var nodeIDs: [String] {
        var seen = Set<String>()
        return windows.compactMap { seen.insert($0.nodeID).inserted ? $0.nodeID : nil }
    }

    /// The newest allowance bucket present, e.g. `2026-09-01`.
    var latestAllowanceMonth: String? {
        windows.map(\.allowanceMonth).max()
    }

    /// Totals for one allowance bucket, summed over EVERY fetched row.
    ///
    /// Not the first page: a busy account produces a window every half hour, so
    /// one 200-row page is under five days and a month total built from it
    /// would be silently short. ``ShareComputeLedgerClient/account`` follows
    /// `next_cursor` to the end precisely so this sum can be honest.
    func totals(forAllowanceMonth month: String) -> ShareComputeLedgerMonthTotals {
        let rows = windows.filter { $0.allowanceMonth == month }
        return ShareComputeLedgerMonthTotals(
            allowanceMonth: month,
            windowCount: rows.count,
            requestCount: rows.reduce(0) { $0 + $1.requestCount },
            inputTokens: rows.reduce(0) { $0 + $1.inputTokens },
            outputTokens: rows.reduce(0) { $0 + $1.outputTokens },
            // Server values, summed. Cap application already happened upstream;
            // re-deriving it here would invent a second source of truth.
            finalCredit: rows.reduce(Decimal(0)) { $0 + $1.finalCredit },
            // `nil` accrued values are UNKNOWN, so a bucket containing one has
            // no honest accrued total. Absence propagates rather than being
            // quietly treated as zero.
            accruedCredit: rows.contains(where: { $0.accruedCredit == nil })
                ? nil
                : rows.reduce(Decimal(0)) { $0 + ($1.accruedCredit ?? 0) },
            nodeIDs: {
                var seen = Set<String>()
                return rows.compactMap { seen.insert($0.nodeID).inserted ? $0.nodeID : nil }
            }(),
            hasPendingWindows: rows.contains { !$0.status.isSettled }
        )
    }
}

/// Aggregated figures for one allowance month.
struct ShareComputeLedgerMonthTotals: Equatable, Sendable {
    let allowanceMonth: String
    let windowCount: Int
    let requestCount: Int
    let inputTokens: Int
    let outputTokens: Int
    let finalCredit: Decimal
    /// `nil` when any contributing window had an unknown accrued amount.
    let accruedCredit: Decimal?
    let nodeIDs: [String]
    let hasPendingWindows: Bool
}

// MARK: - Errors

/// Classified ledger failures.
///
/// No case carries a response body, a credential, or a rendered request. The
/// endpoint is authenticated, so an error string that echoed the request would
/// be one refactor away from putting `Authorization: Bearer qsprk-…` in a log.
enum ShareComputeLedgerError: Error, Equatable, Sendable {
    /// No read key saved. Not a failure — the onboarding state.
    case noReadKey
    /// 401 `missing_key` / `invalid_key`. The saved key is rejected; the UI
    /// offers replacement and never re-displays the rejected value.
    case unauthorized
    /// 400 `bad_cursor` / `bad_limit`. A client bug, surfaced distinctly so it
    /// is not mistaken for an outage.
    case badRequest(code: String)
    /// 429. The endpoint allows 120 requests/hour/IP.
    case rateLimited
    /// 503 `db_unavailable`, or any other 5xx.
    case serviceUnavailable
    case unreachable(String)
    case malformedBody

    var displayMessage: String {
        switch self {
        case .noReadKey:
            return String(localized: "Add a QuickSilver read key to see your credit ledger.")
        case .unauthorized:
            return String(localized: "This read key was rejected. It may have been revoked or replaced.")
        case .badRequest:
            return String(localized: "QuickSilver couldn’t read that request.")
        case .rateLimited:
            return String(localized: "Too many ledger checks. Try again in a few minutes.")
        case .serviceUnavailable:
            return String(localized: "QuickSilver’s ledger is unavailable right now.")
        case .unreachable:
            return String(localized: "Couldn’t reach QuickSilver.")
        case .malformedBody:
            return String(localized: "QuickSilver returned a ledger Rapid couldn’t read.")
        }
    }

    /// Whether the fix is a new key rather than waiting.
    var needsNewKey: Bool { self == .unauthorized }
}

// MARK: - Client

/// Reads the account's contributor ledger.
///
/// The ONLY place in Rapid that attaches a QuickSilver credential to an
/// outbound request other than the share subprocess. Everything about that is
/// deliberately narrow: one endpoint, one header, one call site.
struct ShareComputeLedgerClient: Sendable {
    static let endpoint = URL(string: "https://pay.quicksilverpro.io/v1/pool/ledger")!
    static let timeout: TimeInterval = 10

    /// Contract maximum, and what the paginating walk always asks for — fewer
    /// round trips against a 120/hour budget.
    static let maximumLimit = 200
    static let defaultLimit = 50

    /// Hard ceiling on pages followed in one walk. At 200 rows a page this is
    /// 10 000 windows — years of half-hour buckets. A backend whose
    /// `next_cursor` never goes nil must not spin this client forever.
    static let maximumPages = 50

    /// Performs one request. Injected so tests and fixtures never reach the
    /// network; the default is the hardened session below, never
    /// `URLSession.shared`.
    ///
    /// `URLSession.shared` is disqualified on three counts, and this request
    /// carries a credential so all three matter: it follows redirects (and
    /// REPLAYS the `Authorization` header to wherever it is bounced), it owns
    /// a process-wide cookie store, and it writes responses into a shared
    /// on-disk cache.
    var transport: @Sendable (URLRequest) async throws -> (Data, URLResponse) = {
        try await ShareComputeLedgerClient.session.data(for: $0)
    }

    // MARK: Hardened session

    /// Refuses every 3xx.
    ///
    /// `URLSession`'s default behaviour is to follow a redirect and re-send the
    /// original headers — which would ship `Authorization: Bearer qsprk-…` to
    /// whatever host the response named. Returning `nil` from this delegate
    /// callback means the redirect is never followed and the 3xx surfaces as
    /// the response instead, so the credential cannot leave the pinned origin.
    final class NoRedirectDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(
            _ session: URLSession,
            task: URLSessionTask,
            willPerformHTTPRedirection response: HTTPURLResponse,
            newRequest request: URLRequest,
            completionHandler: @escaping (URLRequest?) -> Void
        ) {
            // `request` is the redirect URLSession WANTS to send, already
            // carrying our Authorization header. It is discarded unsent.
            completionHandler(nil)
        }
    }

    /// Ephemeral, cookie-free, cache-free configuration.
    ///
    /// Exposed so a test can assert the policy without constructing a session.
    static func makeConfiguration() -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        // No cookie jar: the ledger is stateless and a Set-Cookie from a
        // misbehaving edge must not become ambient state on later requests.
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.httpCookieAcceptPolicy = .never
        // No persistent cache: an authenticated, account-scoped response has
        // no business on disk.
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForRequest = timeout
        return configuration
    }

    /// One shared session for the app's ledger reads. A per-request session
    /// would leak a URLSession each time (they are not released until
    /// invalidated).
    static let session: URLSession = URLSession(
        configuration: makeConfiguration(),
        delegate: NoRedirectDelegate(),
        delegateQueue: nil
    )

    static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let raw = try decoder.singleValueContainer().decode(String.self)
            guard let date = ShareComputePoolSummaryClient.parseISO8601(raw) else {
                throw DecodingError.dataCorruptedError(
                    in: try decoder.singleValueContainer(),
                    debugDescription: "not an ISO-8601 timestamp"
                )
            }
            return date
        }
        return decoder
    }

    /// One page.
    ///
    /// `limit` is clamped to the contract's 1...200 rather than sent verbatim:
    /// an out-of-range value earns a 400 `bad_limit`, which spends a request
    /// from the hourly budget to learn something checkable locally.
    func page(
        key: ShareComputeReadKey,
        limit: Int = defaultLimit,
        cursor: Int? = nil,
        nodeID: String? = nil
    ) async throws -> ShareComputeLedgerPage {
        var components = URLComponents(url: Self.endpoint, resolvingAgainstBaseURL: false)!
        var items = [URLQueryItem(name: "limit", value: String(min(max(1, limit), Self.maximumLimit)))]
        if let cursor { items.append(URLQueryItem(name: "cursor", value: String(cursor))) }
        if let nodeID, !nodeID.isEmpty {
            // URLComponents percent-encodes the value, so a node id containing
            // `&` or `#` stays one parameter.
            items.append(URLQueryItem(name: "node_id", value: nodeID))
        }
        components.queryItems = items

        var request = URLRequest(url: components.url!)
        request.httpMethod = "GET"
        request.timeoutInterval = Self.timeout
        // The one and only credential-bearing header in this client, added
        // here and nowhere else.
        request.setValue("Bearer \(key.rawValue)", forHTTPHeaderField: "Authorization")
        request.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await transport(request)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // `localizedDescription` of a URLError never contains the request
            // headers, so the bearer cannot ride out this way.
            throw ShareComputeLedgerError.unreachable(error.localizedDescription)
        }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw Self.classify(status: http.statusCode, body: data)
        }

        do {
            let dto = try Self.makeDecoder().decode(ShareComputeLedgerPageDTO.self, from: data)
            return ShareComputeLedgerPage(dto: dto)
        } catch {
            // Deliberately body-free: a malformed ledger is still an
            // authenticated response and may contain account data.
            throw ShareComputeLedgerError.malformedBody
        }
    }

    /// Follows `next_cursor` to the end and returns every window.
    ///
    /// Three guards, because a paginating loop against a remote cursor is a
    /// spin waiting to happen: a page ceiling, a repeated-cursor check (a
    /// backend that returns the same `next_cursor` twice would otherwise loop
    /// forever), and cooperative cancellation between pages so leaving the tab
    /// stops the walk mid-flight.
    func account(
        key: ShareComputeReadKey,
        nodeID: String? = nil
    ) async throws -> ShareComputeLedgerAccount {
        var windows: [ShareComputeLedgerWindow] = []
        var cursor: Int? = nil
        var seenCursors = Set<Int>()
        var unit = "usd_api_credit"
        var note: String?
        var caps = ShareComputeLedgerCaps(nodeMonthlyUSD: 0, poolMonthlyUSD: 0)

        for pageIndex in 0..<Self.maximumPages {
            try Task.checkCancellation()
            let page = try await self.page(
                key: key,
                limit: Self.maximumLimit,
                cursor: cursor,
                nodeID: nodeID
            )
            if pageIndex == 0 {
                unit = page.unit
                note = page.note
                caps = page.caps
            }
            windows.append(contentsOf: page.windows)

            guard let next = page.nextCursor else {
                return ShareComputeLedgerAccount(
                    unit: unit, note: note, caps: caps,
                    windows: windows, isTruncated: false
                )
            }
            // A cursor we have already followed means the server is not
            // advancing. Stop and report truncation rather than loop.
            guard seenCursors.insert(next).inserted else { break }
            cursor = next
        }

        // Reaching here means either a repeated cursor or the page ceiling —
        // both leave `next_cursor` unfollowed, so the totals describe a prefix
        // of the account and must say so.
        return ShareComputeLedgerAccount(
            unit: unit, note: note, caps: caps,
            windows: windows,
            isTruncated: true
        )
    }

    static func classify(status: Int, body: Data) -> ShareComputeLedgerError {
        let code = (try? JSONDecoder().decode(ShareComputeLedgerErrorDTO.self, from: body))?.error.code
        switch status {
        case 401: return .unauthorized
        case 400: return .badRequest(code: code ?? "bad_request")
        case 429: return .rateLimited
        case 503: return .serviceUnavailable
        default:
            return status >= 500 ? .serviceUnavailable : .badRequest(code: code ?? "http_\(status)")
        }
    }

    private static var userAgent: String {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
        return "Rapid/\(version ?? "unknown") (macOS)"
    }
}

// MARK: - Page state

/// What Credits knows about the ledger right now.
///
/// Distinct from ``ShareComputePoolSummaryState`` because the failure modes are
/// different in kind: a pool summary can only be absent or stale, whereas a
/// ledger can additionally have no credential, a rejected credential, or a
/// spent rate-limit budget — and each of those has a different fix.
enum ShareComputeLedgerState: Equatable, Sendable {
    /// No key saved. The onboarding surface, not an error.
    case noReadKey
    case loading
    case loaded(ShareComputeLedgerAccount)
    /// Authenticated fine, account has no windows yet. A successful 200.
    case loadedEmpty(ShareComputeLedgerAccount)
    case refreshing(ShareComputeLedgerAccount)
    case refreshFailed(ShareComputeLedgerAccount, ShareComputeLedgerError)
    /// The saved key was rejected. Carries nothing of the key itself.
    case unauthorized
    case rateLimited
    case unavailable(ShareComputeLedgerError)

    var account: ShareComputeLedgerAccount? {
        switch self {
        case .loaded(let a), .loadedEmpty(let a), .refreshing(let a), .refreshFailed(let a, _):
            return a
        case .noReadKey, .loading, .unauthorized, .rateLimited, .unavailable:
            return nil
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

    var isBusy: Bool { isLoading || isRefreshing }

    /// True only for the onboarding surface.
    var needsReadKey: Bool {
        self == .noReadKey || self == .unauthorized
    }

    /// The error to SHOW as the whole surface. Absent when previous data is on
    /// screen — a failed refresh over good rows is a note, not a takeover.
    var blockingError: ShareComputeLedgerError? {
        switch self {
        case .unauthorized: return .unauthorized
        case .rateLimited: return .rateLimited
        case .unavailable(let error): return error
        default: return nil
        }
    }

    /// Quiet line for a refresh that failed over data worth keeping.
    var staleNote: String? {
        guard case .refreshFailed(_, let error) = self else { return nil }
        return String(
            format: String(localized: "Couldn’t refresh just now — showing the last reading. (%@)"),
            error.displayMessage
        )
    }

    func beginningLoad() -> Self {
        if let account { return .refreshing(account) }
        return .loading
    }

    /// Never discards rows that are already on screen, EXCEPT when the key
    /// itself was rejected — stale rows under a dead credential would imply
    /// they are still being updated.
    func failing(_ error: ShareComputeLedgerError) -> Self {
        if error == .unauthorized { return .unauthorized }
        if let account { return .refreshFailed(account, error) }
        switch error {
        case .noReadKey: return .noReadKey
        case .rateLimited: return .rateLimited
        default: return .unavailable(error)
        }
    }

    static func loadedState(_ account: ShareComputeLedgerAccount) -> Self {
        account.isEmpty ? .loadedEmpty(account) : .loaded(account)
    }
}

// MARK: - Refresh pacing

/// How often Credits may re-read the ledger.
///
/// The endpoint allows 120 requests/hour/IP and one account walk costs a
/// request PER PAGE, so Live Pool's 30-second cadence would exhaust the budget
/// within minutes and leave the user rate-limited. Five minutes is the floor;
/// the tab also stops entirely when it is no longer visible.
enum ShareComputeLedgerRefresh {
    static let minimumInterval: TimeInterval = 300
    static let recommendedInterval: TimeInterval = 600

    /// How long Refresh stays disabled after a 429.
    ///
    /// Longer than the ordinary interval: the endpoint has just told us we are
    /// asking too often, so backing off further is the only correct response.
    /// A Refresh button that stays clickable after a 429 invites the user to
    /// dig the hole deeper.
    static let rateLimitCooldown: TimeInterval = 600

    static func interval(_ requested: TimeInterval) -> TimeInterval {
        max(minimumInterval, requested)
    }
}

// MARK: - Formatting

enum ShareComputeCreditFormatter {
    /// `$0.0123` — API credit, in the unit the contract defines (1.0 == $1 of
    /// API usage).
    ///
    /// Formats the `Decimal` directly. Going through `Double` would reintroduce
    /// exactly the binary rounding the wire format avoids by sending decimal
    /// literals, and a ledger that displays a cent differently from the
    /// dashboard is worse than useless.
    static func credit(_ value: Decimal) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.minimumFractionDigits = 2
        // Four places: the sample row credits $0.0123, and rounding it to two
        // would display an earned amount as $0.01.
        formatter.maximumFractionDigits = 4
        formatter.usesGroupingSeparator = true
        let number = NSDecimalNumber(decimal: value)
        return "$" + (formatter.string(from: number) ?? number.stringValue)
    }

    /// The placeholder for an unknown accrued amount. Never `$0.00`.
    static let unknown = "—"

    static func credit(_ value: Decimal?) -> String {
        value.map(credit) ?? unknown
    }

    /// `1.28M`, `218k`, `7` — token and request counts.
    static func count(_ value: Int) -> String {
        switch value {
        case ..<1_000: return "\(value)"
        case ..<1_000_000:
            return String(format: "%.1fk", Double(value) / 1_000).replacingOccurrences(of: ".0k", with: "k")
        default:
            return String(format: "%.2fM", Double(value) / 1_000_000)
        }
    }
}
