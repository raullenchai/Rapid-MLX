import Foundation
import Testing
@testable import Rapid

@Suite("Weather region resolution")
struct WeatherRegionTests {
    private static let texas = Data(#"""
    {"results":[
      {"name":"Texas","latitude":20.02,"longitude":-99.19,"country":"Mexico","country_code":"MX","admin1":"Hidalgo","population":993},
      {"name":"Texas","latitude":-28.85,"longitude":151.17,"country":"Australia","country_code":"AU","admin1":"New South Wales","population":843},
      {"name":"Texas","latitude":-2.13,"longitude":-79.59,"country":"Ecuador","country_code":"EC","admin1":"Guayas","population":740}
    ]}
    """#.utf8)

    private static let chineseTexas = Data(#"""
    {"results":[
      {"name":"得克萨斯州新星","latitude":32.09597,"longitude":-98.96367,
       "country":"美国","country_code":"US","admin1":"得克萨斯州","population":827}
    ]}
    """#.utf8)

    private static let bavaria = Data(#"""
    {"results":[
      {"name":"Bavaria","latitude":45.81,"longitude":12.18,"country":"Italy","country_code":"IT","admin1":"Veneto"},
      {"name":"Bavaria","latitude":11.23,"longitude":-74.20,"country":"Colombia","country_code":"CO","admin1":"Magdalena Department"},
      {"name":"Bavaria","latitude":38.79,"longitude":-97.75,"country":"United States","country_code":"US","admin1":"Kansas"}
    ]}
    """#.utf8)

    private static let chicago = Data(#"""
    {"results":[
      {"name":"Chicago","latitude":41.85,"longitude":-87.65,"country":"United States","country_code":"US","admin1":"Illinois","population":2695598},
      {"name":"Chicago","latitude":33.5,"longitude":-90.5,"country":"United States","country_code":"US","admin1":"Mississippi","population":100}
    ]}
    """#.utf8)

    private static let springfield = Data(#"""
    {"results":[
      {"name":"Springfield","latitude":39.8,"longitude":-89.64,"country":"United States","country_code":"US","admin1":"Illinois","population":114000},
      {"name":"Springfield","latitude":37.22,"longitude":-93.3,"country":"United States","country_code":"US","admin1":"Missouri","population":169000},
      {"name":"Springfield","latitude":42.1,"longitude":-72.59,"country":"United States","country_code":"US","admin1":"Massachusetts","population":155000}
    ]}
    """#.utf8)

    private static func loader(
        _ fixtures: [String: Data]
    ) -> WeatherTool.GeocodingLoader {
        { request in
            let url = try #require(request.url)
            let components = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
            let name = try #require(
                components.queryItems?.first(where: { $0.name == "name" })?.value
            )
            let response = try #require(HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: nil, headerFields: nil
            ))
            return (fixtures[name] ?? Data(#"{"results":[]}"#.utf8), response)
        }
    }

    private static func result(
        location: String,
        country: String? = nil,
        admin1: String? = nil,
        fixtures: [String: Data]
    ) async throws -> ToolCallResult {
        let resolution = try await WeatherTool.resolvePlace(
            location: location,
            country: country,
            admin1: admin1,
            load: loader(fixtures)
        )
        switch WeatherTool.unresolvedResult(
            for: resolution, location: location, country: country, admin1: admin1
        ) {
        case .success:
            Issue.record("Expected an unresolved weather location")
            return WeatherTool.notFoundPlaceResult(location: location)
        case .failure(let unresolved):
            return unresolved.result
        }
    }

    @Test("Texas is reported as a region")
    func texasRegion() async throws {
        let result = try await Self.result(location: "Texas", fixtures: ["Texas": Self.texas])
        #expect(result.isError)
        #expect(result.content.contains("Texas, United States"))
        #expect(result.content.contains("Do not retry weather"))
        #expect(!result.content.contains("Add a country"))
    }

    @Test("Unique bare subdivision abbreviation is a region")
    func texasAbbreviation() async throws {
        let result = try await Self.result(location: "TX", fixtures: [:])
        #expect(result.content.contains("Texas, United States"))
        #expect(result.content.contains("state or region"))
    }

    @Test("Texas qualifiers remain a region", arguments: [
        ("Texas, US", nil, nil),
        ("Texas, USA", nil, nil),
        ("Texas", "USA", "Texas"),
    ])
    func texasQualifiers(location: String, country: String?, admin1: String?) async throws {
        let result = try await Self.result(
            location: location,
            country: country,
            admin1: admin1,
            fixtures: ["Texas": Self.texas]
        )
        #expect(result.content.contains("Texas, United States"))
        #expect(result.content.contains("state or region"))
    }

    @Test("Localized provider admin1 identifies a region")
    func localizedTexasRegion() async throws {
        let result = try await Self.result(
            location: "得克萨斯州", fixtures: ["得克萨斯州": Self.chineseTexas]
        )
        #expect(result.content.contains("得克萨斯州"))
        #expect(result.content.contains("state or region"))
    }

    @Test("Unsupported region name with only unrelated towns is not found")
    func bavariaNotFound() async throws {
        let result = try await Self.result(
            location: "Bavaria", fixtures: ["Bavaria": Self.bavaria]
        )
        #expect(result.content.contains("no well-known city or town named"))
        #expect(!result.content.contains("state or region"))
        #expect(!result.content.contains("Add a country"))
    }

    @Test("Overlapping bare abbreviation needs a country hint")
    func westernAustraliaAbbreviation() async throws {
        let bare = try await Self.result(location: "WA", fixtures: [:])
        #expect(!bare.content.contains("state or region"))

        let qualified = try await Self.result(location: "WA, Australia", fixtures: [:])
        #expect(qualified.content.contains("Western Australia, Australia"))
        #expect(qualified.content.contains("state or region"))
    }

    @Test("Existing city resolution is unchanged")
    func chicagoStillResolves() async throws {
        let resolution = try await WeatherTool.resolvePlace(
            location: "Chicago", country: nil, admin1: nil,
            load: Self.loader(["Chicago": Self.chicago])
        )
        guard case .place(let hit) = resolution else {
            Issue.record("Expected Chicago to resolve")
            return
        }
        #expect(hit.name == "Chicago")
        #expect(hit.admin1 == "Illinois")
    }

    @Test("Same-name cities remain ambiguous")
    func springfieldStillAmbiguous() async throws {
        let result = try await Self.result(
            location: "Springfield", fixtures: ["Springfield": Self.springfield]
        )
        #expect(result.content.contains("could not uniquely identify"))
    }

    @Test("A city with an unmatched state qualifier is not found")
    func chicagoWrongState() async throws {
        let result = try await Self.result(
            location: "Chicago, TX", fixtures: ["Chicago": Self.chicago]
        )
        #expect(result.content.contains("no well-known city or town named"))
        #expect(result.content.contains("TX"))
        #expect(!result.content.contains("Add a country"))
    }
}
