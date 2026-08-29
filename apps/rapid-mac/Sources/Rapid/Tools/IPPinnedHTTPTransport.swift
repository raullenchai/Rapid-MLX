import Foundation
import Network

/// An HTTP/1.1 transport that opens sockets to an address selected by
/// ``BrowseSSRFGuard``. A hostile DNS response cannot swap the address after
/// validation because this transport never resolves the original hostname.
enum IPPinnedHTTPTransport {
    static func fetch(
        url: URL,
        address: ParsedIP,
        byteLimit: Int
    ) async throws -> (data: Data, response: HTTPURLResponse) {
        guard let scheme = url.scheme?.lowercased() else {
            throw transportError("URL has no scheme")
        }
        guard let host = url.host, !host.isEmpty else {
            throw transportError("URL has no host")
        }
        guard let port = endpointPort(for: url) else {
            throw transportError("URL has an invalid port")
        }

        let connection = try makeConnection(
            address: address,
            port: port,
            secure: scheme == "https",
            serverName: host
        )
        let reader = ResponseReader(
            connection: connection,
            request: request(for: url, address: address, host: host, port: port),
            url: url,
            byteLimit: byteLimit
        )

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                reader.start { result in
                    continuation.resume(with: result)
                }
            }
        } onCancel: {
            reader.cancel()
        }
    }

    private static func makeConnection(
        address: ParsedIP,
        port: UInt16,
        secure: Bool,
        serverName: String
    ) throws -> NWConnection {
        let endpointHost: NWEndpoint.Host
        switch address.family {
        case .v4:
            guard let ipv4 = IPv4Address(address.canonical) else {
                throw transportError("invalid resolved IPv4 address")
            }
            endpointHost = .ipv4(ipv4)
        case .v6:
            guard let ipv6 = IPv6Address(address.canonical) else {
                throw transportError("invalid resolved IPv6 address")
            }
            endpointHost = .ipv6(ipv6)
        }
        guard let endpointPort = NWEndpoint.Port(rawValue: port) else {
            throw transportError("invalid endpoint port")
        }

        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.connectionTimeout = 10
        let parameters: NWParameters
        if secure {
            let tlsOptions = NWProtocolTLS.Options()
            sec_protocol_options_set_tls_server_name(
                tlsOptions.securityProtocolOptions,
                serverName
            )
            parameters = NWParameters(tls: tlsOptions, tcp: tcpOptions)
        } else {
            parameters = NWParameters(tls: nil, tcp: tcpOptions)
        }
        parameters.allowLocalEndpointReuse = false

        return NWConnection(
            host: endpointHost,
            port: endpointPort,
            using: parameters
        )
    }

    private static func endpointPort(for url: URL) -> UInt16? {
        if let port = url.port {
            return UInt16(exactly: port)
        }
        return url.scheme?.lowercased() == "https" ? 443 : 80
    }

    private static func request(
        for url: URL,
        address: ParsedIP,
        host: String,
        port: UInt16
    ) -> Data {
        let path = url.path.isEmpty ? "/" : url.path
        let target = url.query.map { "\(path)?\($0)" } ?? path
        let headers = [
            "GET \(target) HTTP/1.1",
            "Host: \(hostHeader(host: host, address: address, port: port))",
            "User-Agent: \(BrowseTool.userAgent)",
            "Accept: text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.8",
            "Accept-Encoding: identity",
            "Connection: close",
        ].joined(separator: "\r\n") + "\r\n\r\n"
        return Data(headers.utf8)
    }

    private static func hostHeader(
        host: String,
        address: ParsedIP,
        port: UInt16
    ) -> String {
        let bareHost = host.hasPrefix("[") && host.hasSuffix("]")
            ? String(host.dropFirst().dropLast())
            : host
        let hostText = address.family == .v6 ? "[\(bareHost)]" : bareHost
        return (port == 80 || port == 443) ? hostText : "\(hostText):\(port)"
    }

    fileprivate static func transportError(_ message: String) -> NSError {
        NSError(
            domain: "RapidBrowseTransport",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }
}

/// Parses the complete HTTP/1.1 bytes collected by ``ResponseReader``.
enum IPHTTPResponseParser {
    static func parse(data: Data, url: URL) throws -> (Data, HTTPURLResponse) {
        let headerSeparator = Data("\r\n\r\n".utf8)
        guard let headerEnd = data.range(of: headerSeparator) else {
            throw IPPinnedHTTPTransport.transportError("response ended before HTTP headers completed")
        }
        guard headerEnd.lowerBound <= 128 * 1024 else {
            throw IPPinnedHTTPTransport.transportError("HTTP response headers exceeded 128 KB")
        }

        let rawHeaders = String(decoding: data[..<headerEnd.lowerBound], as: UTF8.self)
        let lines = rawHeaders.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard let statusLine = lines.first else {
            throw IPPinnedHTTPTransport.transportError("missing HTTP status line")
        }
        let statusParts = statusLine.split(separator: " ", maxSplits: 2)
        guard statusParts.count >= 2, let statusCode = Int(statusParts[1]) else {
            throw IPPinnedHTTPTransport.transportError("invalid HTTP status line")
        }

        var headerFields: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[..<colon]).trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            headerFields[name] = headerFields[name].map { "\($0), \(value)" } ?? value
        }

        let bodyData = Data(data[headerEnd.upperBound...])
        let responseBody = if headerFields["transfer-encoding"]?
            .lowercased()
            .contains("chunked") == true {
            try decodeChunked(bodyData)
        } else {
            bodyData
        }
        guard let response = HTTPURLResponse(
            url: url,
            statusCode: statusCode,
            httpVersion: "HTTP/1.1",
            headerFields: headerFields
        ) else {
            throw IPPinnedHTTPTransport.transportError("could not construct HTTP response")
        }
        return (responseBody, response)
    }

    private static func decodeChunked(_ input: Data) throws -> Data {
        var output = Data()
        var offset = input.startIndex

        while true {
            guard let lineEnd = input.range(of: Data("\r\n".utf8), in: offset..<input.endIndex) else {
                throw IPPinnedHTTPTransport.transportError("truncated HTTP chunked response")
            }
            let sizeLine = String(decoding: input[offset..<lineEnd.lowerBound], as: UTF8.self)
            let sizeText = sizeLine.split(separator: ";").first.map(String.init) ?? sizeLine
            guard let size = Int(sizeText.trimmingCharacters(in: .whitespaces), radix: 16), size >= 0 else {
                throw IPPinnedHTTPTransport.transportError("invalid HTTP chunk size")
            }
            offset = lineEnd.upperBound
            if size == 0 {
                return output
            }
            let chunkEnd = input.index(offset, offsetBy: size)
            guard chunkEnd <= input.endIndex else {
                throw IPPinnedHTTPTransport.transportError("truncated HTTP chunked response")
            }
            output.append(contentsOf: input[offset..<chunkEnd])
            let terminator = input.index(chunkEnd, offsetBy: 2)
            guard terminator <= input.endIndex,
                  input[chunkEnd] == 0x0D,
                  input[input.index(after: chunkEnd)] == 0x0A else {
                throw IPPinnedHTTPTransport.transportError("invalid HTTP chunk terminator")
            }
            offset = terminator
        }
    }
}

private final class ResponseReader: @unchecked Sendable {
    private let connection: NWConnection
    private let request: Data
    private let url: URL
    private let byteLimit: Int
    private let queue = DispatchQueue(label: "rapid.mlx.browse.ip-pinned")
    private let lock = NSLock()
    private var isFinished = false
    private var activeCompletion: (@Sendable (Result<(Data, HTTPURLResponse), Error>) -> Void)?
    private var bufferedData = Data()

    init(connection: NWConnection, request: Data, url: URL, byteLimit: Int) {
        self.connection = connection
        self.request = request
        self.url = url
        self.byteLimit = byteLimit
    }

    func start(
        completion: @escaping @Sendable (Result<(Data, HTTPURLResponse), Error>) -> Void
    ) {
        lock.lock()
        let wasCancelledBeforeStart = isFinished
        if !wasCancelledBeforeStart {
            activeCompletion = completion
        }
        lock.unlock()
        if wasCancelledBeforeStart {
            completion(.failure(CancellationError()))
            return
        }

        connection.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.connection.send(
                    content: self.request,
                    completion: .contentProcessed { error in
                        if let error {
                            self.finish(.failure(error))
                        }
                    }
                )
                self.receive()
            case .failed(let error):
                self.finish(.failure(error))
            case .cancelled:
                self.finish(.failure(CancellationError()))
            default:
                break
            }
        }
        connection.start(queue: queue)
    }

    func cancel() {
        finish(.failure(CancellationError()))
    }

    private func receive(
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            if let error {
                self.finish(.failure(error))
                return
            }
            if let data {
                do {
                    try self.append(data)
                } catch {
                    self.finish(.failure(error))
                    return
                }
            }
            if isComplete {
                self.finish(self.parse())
                return
            }
            self.receive()
        }
    }

    private func append(_ data: Data) throws {
        bufferedData.append(data)
        guard bufferedData.count <= byteLimit else {
            throw IPPinnedHTTPTransport.transportError(
                "page exceeded \(byteLimit / (1024 * 1024)) MB cap"
            )
        }
    }

    private func parse() -> Result<(Data, HTTPURLResponse), Error> {
        Result {
            try IPHTTPResponseParser.parse(data: bufferedData, url: url)
        }
    }

    private func finish(
        _ result: Result<(Data, HTTPURLResponse), Error>,
    ) {
        lock.lock()
        guard !isFinished else {
            lock.unlock()
            return
        }
        isFinished = true
        let pendingCompletion = activeCompletion
        activeCompletion = nil
        lock.unlock()
        connection.cancel()
        pendingCompletion?(result)
    }
}
