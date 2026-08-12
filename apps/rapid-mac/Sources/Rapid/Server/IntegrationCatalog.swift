import Foundation

struct IntegrationTarget: Codable, Identifiable, Equatable, Sendable {
    enum Kind: String, Codable, Sendable {
        case configWriter = "config_writer"
        case adapterProfile = "adapter_profile"
    }

    let id: String
    let name: String
    let kind: Kind
    let configPath: String?

    enum CodingKeys: String, CodingKey {
        case id, name, kind
        case configPath = "config_path"
    }
}

enum IntegrationCatalog {
    static func decode(_ data: Data) throws -> [IntegrationTarget] {
        try JSONDecoder().decode([IntegrationTarget].self, from: data)
    }

    static func load(binary: URL? = ServerLocator.find()) async -> [IntegrationTarget] {
        guard let binary else { return [] }
        return await Task.detached(priority: .utility) {
            let process = Process()
            process.executableURL = binary
            process.arguments = ["launch", "list", "--json"]
            var environment = ProcessInfo.processInfo.environment
            environment["RAPID_MLX_TELEMETRY"] = "0"
            process.environment = environment
            let output = Pipe()
            process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            do {
                try process.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else { return [] }
                return (try? decode(data)) ?? []
            } catch {
                return []
            }
        }.value
    }
}
