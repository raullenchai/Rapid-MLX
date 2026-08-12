import Foundation
import Testing
@testable import Rapid

struct IntegrationCatalogTests {
    @Test("engine registry JSON decodes both integration kinds")
    func decodesRegistry() throws {
        let data = Data(#"[{"id":"cline","name":"Cline","kind":"config_writer","config_path":"/tmp/cline.json"},{"id":"codex","name":"Codex CLI","kind":"adapter_profile","config_path":null}]"#.utf8)
        let targets = try IntegrationCatalog.decode(data)
        #expect(targets.count == 2)
        #expect(targets[0].kind == .configWriter)
        #expect(targets[0].configPath == "/tmp/cline.json")
        #expect(targets[1].kind == .adapterProfile)
        #expect(targets[1].configPath == nil)
    }
}
