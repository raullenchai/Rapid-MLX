import Foundation
import Testing
@testable import Rapid

@Suite("Cached repository identity collisions")
struct CachedRepositoryCollisionTests {
    @Test("Both merge paths preserve the catalog and expose an unambiguous cached choice")
    func collision() throws {
        for external in [false, true] {
            let path = "/models/local-chat-1b"
            let managedRow = external ? """
            {"alias":null,"repo":"local-chat-1b","subfolder":null,"size_bytes":1024,
            "external":false,"state":"unmapped","launch_path":null},
            """ : ""
            let payload = """
            {"cached":[\(managedRow){"alias":null,"repo":"local-chat-1b","subfolder":null,
            "size_bytes":1024,"external":\(external),
            "state":"\(external ? "external" : "unmapped")","launch_path":"\(path)"}]}
            """
            let cached = try #require(ModelCatalog.parseCachedJSON(payload))
            let original = ModelEntry(
                alias: "local-chat-1b", hfRepo: "org/Unrelated", sizeOnDisk: nil,
                cached: false, sourceSubfolder: "nested"
            )
            let variants = [
                ModelCatalog.mergeAtomicAndCached(atomic: [original], cached: cached, excluded: []),
                ModelCatalog.mergeAvailableAndCached(
                    available: [(original.alias, original.hfRepo)], cached: cached, excluded: []
                ),
            ]
            for entries in variants {
                #expect(entries.count == 2)
                #expect(Set(entries.map(\.id)).count == 2)
                let catalog = try #require(entries.first { $0.alias == original.alias })
                #expect(catalog.hfRepo == "org/Unrelated")
                #expect(!catalog.cached)
                let local = try #require(entries.first { $0.alias == path })
                #expect(local.cached)
                #expect(local.isExternal == external)
                #expect(local.hfRepo == original.alias)
                #expect(SessionModelRestore.shouldPersistChatAlias(catalogEntry: local))
                let restored = SessionModelRestore.resolve(
                    legacyLastAlias: path, dictationAlias: nil, speechAlias: nil, catalog: entries
                )
                #expect(restored.chatAlias == path)
                let hardware = MacHardware(
                    brandString: "Apple M2", family: .m2, tier: .base,
                    physicalRAMBytes: 16 * 1024 * 1024 * 1024, memoryBandwidthGBs: 100
                )
                #expect(CacheAwareDefault.pick(
                    catalog: entries, hardware: hardware, bucketedDefault: original.alias
                ) == path)
            }
        }
    }

    @Test("Complete non-chat collisions remain inventory-only without a certified path")
    func uncertifiedCollision() throws {
        let payload = """
        {"cached":[{"alias":null,"repo":"local-chat","subfolder":null,
        "size_bytes":1024,"external":false,"state":"unmapped","launch_path":null}]}
        """
        let cached = try #require(ModelCatalog.parseCachedJSON(payload))
        let original = ModelEntry(alias: "local-chat", hfRepo: "org/Unrelated", sizeOnDisk: nil, cached: false)
        let variants = [
            ModelCatalog.mergeAtomicAndCached(atomic: [original], cached: cached, excluded: []),
            ModelCatalog.mergeAvailableAndCached(
                available: [(original.alias, original.hfRepo)], cached: cached, excluded: []
            ),
        ]
        for entries in variants {
            #expect(entries.count == 1)
            #expect(entries.first?.hfRepo == "org/Unrelated")
            #expect(entries.first?.cached == false)
        }
    }

    @Test("Collision paths honor discovery in Quickstart and session persistence")
    @MainActor
    func discoveryOnCollisionPaths() throws {
        for external in [false, true] {
            for ready in [false, true] {
                let path = "/models/certified-checkpoint"
                let payload = """
                {"cached":[{"alias":null,"repo":"local-chat","subfolder":null,
                "size_bytes":1024,"external":\(external),
                "state":"\(external ? "external" : "unmapped")","launch_path":"\(path)",
                "chat_ready":\(ready),"chat_fit":"yes","chat_mlx_native":true}]}
                """
                let cached = try #require(ModelCatalog.parseCachedJSON(payload))
                let original = ModelEntry(
                    alias: "local-chat", hfRepo: "org/Unrelated", sizeOnDisk: nil, cached: false
                )
                for entries in [
                    ModelCatalog.mergeAtomicAndCached(atomic: [original], cached: cached, excluded: []),
                    ModelCatalog.mergeAvailableAndCached(
                        available: [(original.alias, original.hfRepo)], cached: cached, excluded: []
                    ),
                ] {
                    let local = try #require(entries.first { $0.alias == path })
                    #expect(local.isUncataloged)
                    #expect(local.cachedChatDiscovery?.ready == ready)
                    #expect(!entries.first { $0.alias == original.alias }!.cached)
                    #expect(SessionModelRestore.shouldPersistChatAlias(catalogEntry: local) == ready)
                    #expect(SessionModelRestore.resolve(
                        legacyLastAlias: path, dictationAlias: nil, speechAlias: nil, catalog: entries
                    ).chatAlias == (ready ? path : nil))
                    let hardware = MacHardware(
                        brandString: "Apple M2", family: .m2, tier: .base,
                        physicalRAMBytes: 16 * 1024 * 1024 * 1024, memoryBandwidthGBs: 100
                    )
                    let picked = CacheAwareDefault.pick(
                        catalog: entries, hardware: hardware, bucketedDefault: original.alias
                    )
                    #expect((picked == path) == ready)
                    #expect(QuickstartView.quickstartCachedModels(entries).contains { $0.alias == path } == ready)
                }
            }
        }
    }

    @Test("Invalid local launch paths never become launch identities")
    func invalidPath() {
        for path in ["relative", "/models/../other"] {
            let payload = """
            {"cached":[{"alias":null,"repo":"local-chat","subfolder":null,
            "size_bytes":1024,"external":false,"state":"unmapped","launch_path":"\(path)"}]}
            """
            #expect(ModelCatalog.parseCachedJSON(payload) == nil)
        }
    }
    @Test("Legacy five-field inventory retains explicit collision paths")
    func legacyLaunchPath() throws {
        let path = "/models/local-chat-1b"
        let original = ModelEntry(
            alias: "local-chat-1b", hfRepo: "org/Unrelated", sizeOnDisk: nil, cached: false
        )
        let cached: [(String, String?, String?, String?, String?)] = [
            ("(unmapped)", "local-chat-1b", nil, "1 GiB", path)
        ]
        for entries in [
            ModelCatalog.mergeAtomicAndCached(atomic: [original], cached: cached, excluded: []),
            ModelCatalog.mergeAvailableAndCached(
                available: [(original.alias, original.hfRepo)], cached: cached, excluded: []
            )
        ] {
            #expect(entries.count == 2)
            let local = try #require(entries.first { $0.alias == path })
            #expect(local.cached)
            #expect(local.hfRepo == original.alias)
            #expect(local.cachedChatDiscovery == nil)
            #expect(entries.first { $0.alias == original.alias }?.cached == false)
        }
    }

}
