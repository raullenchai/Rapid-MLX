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
        #expect(result.failureKind == .needsClarification)
        #expect(result.content.contains("Texas, United States"))
        #expect(result.content.contains("Do not retry weather"))
        #expect(!result.content.contains("Add a country"))
    }

    @Test("Region clarification preserves the exact model-facing instructions")
    func regionClarificationPreservesContent() {
        for label in ["Texas, United States", ""] {
            let result = WeatherTool.regionResult(location: "Texas", label: label)
            let parenthetical = label.isEmpty ? "" : " (\(label))"
            #expect(result.isError)
            #expect(result.failureKind == .needsClarification)
            #expect(result.content == "weather: \"Texas\" is a state or region\(parenthetical), not a single place, so it has no single forecast. Do not retry weather with this region name or with more qualifiers, and do not search the web for it. Ask the user which city they mean, or call weather once with one major city in that region and say in your answer which city the forecast is for.")
        }
    }

    @Test("Missing and ambiguous weather locations stay real failures")
    func realWeatherErrorsStayFailures() async {
        let missing = await WeatherTool.run(arguments: "{}")
        let ambiguous = WeatherTool.ambiguousPlaceResult(location: "Springfield")
        for result in [missing, ambiguous] {
            #expect(result.isError)
            let kind = result.failureKind ?? FailureDiagnoser.toolFailureKind(
                toolName: "weather", content: result.content, isError: result.isError
            )
            #expect(kind == .toolFailed)
            if let kind {
                #expect(FailureDiagnoser.diagnosis(for: kind).severity == .error)
            }
        }
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
        ("Texas", "US", "TX"),
        ("Texas, USA", nil, "USA"),
        ("Texas, TX", nil, nil),
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
        let restated = try await Self.result(location: "WA, WA", fixtures: [:])
        #expect(!restated.content.contains("state or region"))

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

    @Test("A region name shared by major cities stays ambiguous")
    func victoriaStaysAmbiguous() async throws {
        let victoria = Data(#"""
        {"results":[
          {"name":"Victoria","latitude":48.43,"longitude":-123.37,"country":"Canada","country_code":"CA","admin1":"British Columbia","population":289625},
          {"name":"Victoria","latitude":22.28,"longitude":114.16,"country":"Hong Kong","country_code":"HK","admin1":"Central and Western","population":956800},
          {"name":"Victoria","latitude":28.8,"longitude":-97.0,"country":"United States","country_code":"US","admin1":"Texas","population":67574}
        ]}
        """#.utf8)
        let bare = try await Self.result(location: "Victoria", fixtures: ["Victoria": victoria])
        #expect(bare.content.contains("could not uniquely identify"))
        let australia = try await Self.result(
            location: "Victoria, Australia", fixtures: ["Victoria": victoria]
        )
        #expect(australia.content.contains("Victoria, Australia"))
        #expect(australia.content.contains("state or region"))
    }

    @Test("A state restated by an abbreviation that is also an ISO code stays a region", arguments: [
        ("California, CA", "California, United States"),
        ("Georgia, GA", "Georgia, United States"),
        ("Indiana, IN", "Indiana, United States"),
    ])
    func isoLookingStateAbbreviations(location: String, label: String) async throws {
        let result = try await Self.result(location: location, fixtures: [:])
        #expect(result.content.contains("state or region (\(label))"))
    }

    @Test("A subdivision named like a country needs a country hint")
    func georgiaNeedsCountryHint() async throws {
        let bare = try await Self.result(location: "Georgia", fixtures: [:])
        #expect(!bare.content.contains("state or region"))
        let us = try await Self.result(location: "Georgia, US", fixtures: [:])
        #expect(us.content.contains("state or region (Georgia, United States)"))
        let admin1Country = try await Self.result(location: "Georgia", admin1: "USA", fixtures: [:])
        #expect(admin1Country.content.contains("state or region (Georgia, United States)"))
        let admin1Code = try await Self.result(location: "Georgia", admin1: "US", fixtures: [:])
        #expect(admin1Code.content.contains("state or region (Georgia, United States)"))
    }

    @Test("A conflicting comma subdivision is a qualifier mismatch, not a region", arguments: [
        ("Texas, California", "Texas", "California"),
        ("California, NY", "California", "NY"),
    ])
    func conflictingCommaSubdivision(location: String, name: String, qualifier: String) async throws {
        let california = Data(#"""
        {"results":[
          {"name":"California","latitude":38.4,"longitude":-76.5,"country":"United States","country_code":"US","admin1":"Maryland","population":11857},
          {"name":"California","latitude":40.07,"longitude":-79.89,"country":"United States","country_code":"US","admin1":"Pennsylvania","population":6608}
        ]}
        """#.utf8)
        let result = try await Self.result(
            location: location, fixtures: ["Texas": Self.texas, "California": california]
        )
        #expect(!result.content.contains("state or region"))
        #expect(result.content.contains("found places named \"\(name)\", but none matching \"\(qualifier)\""))
    }

    @Test("Contradictory country qualifiers are not a region")
    func contradictoryCountries() async throws {
        let result = try await Self.result(
            location: "Texas, Canada", country: "US", fixtures: ["Texas": Self.texas]
        )
        #expect(!result.content.contains("state or region"))
    }

    @Test("A bare abbreviation that is also a country code needs a country hint", arguments: [
        "CA", "IN", "GA",
    ])
    func isoCodeAbbreviationsNeedHint(location: String) async throws {
        let bare = try await Self.result(location: location, fixtures: [:])
        #expect(!bare.content.contains("state or region"))
        let us = try await Self.result(location: "\(location), USA", fixtures: [:])
        #expect(us.content.contains("state or region"))
    }

    @Test("A contradicting explicit admin1 is a qualifier mismatch, not a region")
    func texasWithCaliforniaAdmin1() async throws {
        let result = try await Self.result(
            location: "Texas", admin1: "California", fixtures: ["Texas": Self.texas]
        )
        #expect(!result.content.contains("state or region"))
        #expect(result.content.contains("found places named \"Texas\", but none matching \"California\""))
    }

    @Test("A known city excluded by its qualifiers asks for one corrected retry")
    func chicagoWrongState() async throws {
        let result = try await Self.result(
            location: "Chicago, TX", fixtures: ["Chicago": Self.chicago]
        )
        #expect(result.content.contains("found places named \"Chicago\", but none matching \"TX\""))
        #expect(result.content.contains("only the place name"))
        #expect(!result.content.contains("state or region"))
    }

    @Test("A city whose admin1 shares its name is never a region")
    func tokyoWithCountryAsAdmin1() async throws {
        let tokyo = Data(#"""
        {"results":[
          {"name":"Tokyo","latitude":35.69,"longitude":139.69,"country":"Japan","country_code":"JP","admin1":"Tokyo","population":9733276},
          {"name":"Tokyo","latitude":-9.0,"longitude":147.0,"country":"Papua New Guinea","country_code":"PG","admin1":"Central Province"}
        ]}
        """#.utf8)
        let result = try await Self.result(
            location: "Tokyo, Japan", admin1: "Japan", fixtures: ["Tokyo": tokyo]
        )
        #expect(!result.content.contains("state or region"))
        #expect(result.content.contains("found places named \"Tokyo\", but none matching \"Japan\"."))
    }

    @Test("A lone accent-folded small namesake is not found, not ambiguous")
    func foldOnlyNamesakeNotFound() async throws {
        let xian = Data(#"""
        {"results":[{"name":"Xián","latitude":42.0,"longitude":-8.0,"country":"Spain","country_code":"ES","admin1":"Galicia","population":40}]}
        """#.utf8)
        let result = try await Self.result(location: "Xian", fixtures: ["Xian": xian])
        #expect(result.content.contains("no well-known city or town named"))
        #expect(!result.content.contains("Add a country"))
    }

    @Test("Repeating a city name as admin1 does not make it a region")
    func cityNameAsAdmin1IsNotRegion() async throws {
        let paris = Data(#"""
        {"results":[{"name":"Paris","latitude":48.85,"longitude":2.35,"country":"France","country_code":"FR","admin1":"Île-de-France","population":2138551}]}
        """#.utf8)
        let result = try await Self.result(
            location: "Paris", country: "France", admin1: "Paris", fixtures: ["Paris": paris]
        )
        #expect(!result.content.contains("state or region"))
    }
}
