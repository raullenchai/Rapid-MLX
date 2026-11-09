import Darwin
import Foundation
import Network
import Observation
import SystemConfiguration

/// LAN mode must never fall back to the resolver's one-time credential.
enum LANAccess {
    static func credential(
        store: any EmbeddedBearerCredentialStoring,
        generateSecret: () -> String? = { BearerSecret.generate() }
    ) -> EmbeddedBearerMaterial? {
        let material = EmbeddedBearerMaterialResolver.resolve(
            lifetime: .explicit, store: store, now: Date(), generateSecret: generateSecret
        )
        guard !material.secret.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              material.isPersisted, material.issue == nil,
              case .found(let saved) = store.loadCredential(),
              saved.secret == material.secret, saved.lifetime == .explicit else { return nil }
        return material
    }

    static func baseURL(address: String, port: Int) -> String? {
        var parsed = in_addr()
        guard (1...65535).contains(port),
              inet_pton(AF_INET, address, &parsed) == 1,
              !address.hasPrefix("127."), !address.hasPrefix("169.254."),
              (1...223).contains(Int(address.split(separator: ".")[0]) ?? 0) else { return nil }
        return "http://\(address):\(port)/v1"
    }

    static func addresses() -> [String] {
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return [] }
        defer { freeifaddrs(first) }
        var addresses = Set<String>()
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let item = cursor {
            let interface = item.pointee
            defer { cursor = interface.ifa_next }
            guard let address = interface.ifa_addr,
                  Int32(address.pointee.sa_family) == AF_INET,
                  interface.ifa_flags & UInt32(IFF_UP) != 0,
                  interface.ifa_flags & UInt32(IFF_LOOPBACK) == 0 else { continue }
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(address, socklen_t(address.pointee.sa_len), &buffer,
                           socklen_t(buffer.count), nil, 0, NI_NUMERICHOST) == 0 {
                let value = String(cString: buffer)
                if baseURL(address: value, port: 1) != nil { addresses.insert(value) }
            }
        }
        let state = SCDynamicStoreCopyValue(nil, "State:/Network/Global/IPv4" as CFString) as? [String: Any]
        let primaryInterface = state?["PrimaryInterface"] as? String
        let primaryState = primaryInterface.flatMap {
            SCDynamicStoreCopyValue(nil, "State:/Network/Interface/\($0)/IPv4" as CFString) as? [String: Any]
        }
        let primary = (primaryState?["Addresses"] as? [String])?.first
        return PhoneConnect.orderedAddresses(Array(addresses), primary: primary)
    }
}

/// A public page URL with the existing LAN bearer in its client-only fragment.
enum PhoneConnect {
    static func url(address: String, port: Int, key: String, lanEnabled: Bool) -> URL? {
        guard lanEnabled, !key.isEmpty,
              let base = LANAccess.baseURL(address: address, port: port),
              var components = URLComponents(string: base) else { return nil }
        components.path = "/connect"
        var fragment = URLComponents()
        fragment.queryItems = [URLQueryItem(name: "k", value: key)]
        components.percentEncodedFragment = fragment.percentEncodedQuery?.replacingOccurrences(of: "+", with: "%2B")
        return components.url
    }

    static func orderedAddresses(_ addresses: [String], primary: String?) -> [String] {
        let valid = Set(addresses.filter { LANAccess.baseURL(address: $0, port: 1) != nil })
        return valid.sorted { left, right in
            if left == primary { return true }
            if right == primary { return false }
            return left < right
        }
    }
}

@MainActor @Observable
final class LANAddressMonitor {
    private(set) var addresses: [String] = []
    @ObservationIgnored private var monitor: NWPathMonitor?

    func start() {
        refresh()
        guard monitor == nil else { return }
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] _ in
            Task { @MainActor [weak self] in self?.refresh() }
        }
        self.monitor = monitor
        monitor.start(queue: DispatchQueue(label: "Rapid.LANAddresses"))
    }

    func refresh() { addresses = LANAccess.addresses() }

    func stop() {
        monitor?.cancel()
        monitor = nil
    }
}
