import Foundation
import Testing
@testable import Rapid

@Suite("Weather geocoding qualifiers")
struct WeatherGeocodingQualifierTests {
    private static let chicago = Data(#"""
    {"results":[
      {"name":"Chicago","latitude":41.85,"longitude":-87.65,"country":"United States","country_code":"US","admin1":"Illinois","population":2695598,"timezone":"America/Chicago"},
      {"name":"Chicago","latitude":33.5,"longitude":-90.5,"country":"United States","country_code":"US","admin1":"Mississippi","population":100,"timezone":"America/Chicago"},
      {"name":"Chicago Heights","latitude":41.5,"longitude":-87.6,"country":"United States","country_code":"US","admin1":"Illinois","population":30000,"timezone":"America/Chicago"}
    ]}
    """#.utf8)

    private static let portland = Data(#"""
    {"results":[
      {"name":"Portland","latitude":45.52,"longitude":-122.68,"country":"United States","country_code":"US","admin1":"Oregon","population":650000,"timezone":"America/Los_Angeles"},
      {"name":"Portland","latitude":43.66,"longitude":-70.26,"country":"United States","country_code":"US","admin1":"Maine","population":68000,"timezone":"America/New_York"}
    ]}
    """#.utf8)

    private static let springfield = Data(#"""
    {"results":[
      {"name":"Springfield","latitude":39.8,"longitude":-89.64,"country":"United States","country_code":"US","admin1":"Illinois","population":114000,"timezone":"America/Chicago"},
      {"name":"Springfield","latitude":37.22,"longitude":-93.3,"country":"United States","country_code":"US","admin1":"Missouri","population":169000,"timezone":"America/Chicago"},
      {"name":"Springfield","latitude":42.1,"longitude":-72.59,"country":"United States","country_code":"US","admin1":"Massachusetts","population":155000,"timezone":"America/New_York"}
    ]}
    """#.utf8)

    private static let london = Data(#"""
    {"results":[
      {"name":"London","latitude":51.51,"longitude":-0.13,"country":"United Kingdom","country_code":"GB","admin1":"England","population":8900000,"timezone":"Europe/London"},
      {"name":"London","latitude":42.98,"longitude":-81.23,"country":"Canada","country_code":"CA","admin1":"Ontario","population":346000,"timezone":"America/Toronto"}
    ]}
    """#.utf8)

    private static let perth = Data(#"""
    {"results":[
      {"name":"Perth","latitude":-31.95,"longitude":115.86,"country":"Australia","country_code":"AU","admin1":"Western Australia","population":1900000,"timezone":"Australia/Perth"},
      {"name":"Perth","latitude":56.4,"longitude":-3.43,"country":"United Kingdom","country_code":"GB","admin1":"Scotland","population":47000,"timezone":"Europe/London"}
    ]}
    """#.utf8)

    @Test("Comma-separated qualifiers resolve through the geocoding loader", arguments: [
        "Chicago, IL", "Chicago, Illinois, USA", "Chicago, Illinois, United States", "Chicago, U.S.",
    ])
    func chicagoCommaQualifiers(location: String) async throws {
        let hit = try await WeatherTool.geocode(location: location, country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            let query = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
            #expect(query.queryItems?.first(where: { $0.name == "name" })?.value == "Chicago")
            let response = try #require(HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: nil, headerFields: nil
            ))
            return (Self.chicago, response)
        }
        #expect(hit?.name == "Chicago")
        #expect(hit?.admin1 == "Illinois")
        #expect(hit?.countryCode == "US")
    }

    @Test("State abbreviations also work when they are other countries' ISO codes", arguments: [
        ("OR", "Oregon"), ("ME", "Maine"),
    ])
    func portlandStates(abbreviation: String, expected: String) {
        let hit = WeatherTool.parseGeocodingResponse(
            Self.portland, location: "Portland", qualifiers: [abbreviation]
        )
        #expect(hit?.admin1 == expected)
        #expect(hit?.countryCode == "US")
    }

    @Test("Springfield requires a qualifier when populations are similar")
    func springfieldAmbiguity() {
        #expect(WeatherTool.parseGeocodingResponse(Self.springfield, location: "Springfield") == nil)
        for (abbreviation, expected) in [("IL", "Illinois"), ("MO", "Missouri")] {
            let hit = WeatherTool.parseGeocodingResponse(
                Self.springfield, location: "Springfield", qualifiers: [abbreviation]
            )
            #expect(hit?.admin1 == expected)
        }
    }

    @Test("British country aliases resolve London", arguments: [
        "UK", "U.K.", "Great Britain", "Britain", "United Kingdom",
    ])
    func londonCountryAliases(alias: String) {
        let hit = WeatherTool.parseGeocodingResponse(
            Self.london, location: "London", qualifiers: [alias], country: alias
        )
        #expect(hit?.countryCode == "GB")
    }

    @Test("UK nations stay admin1 qualifiers, not country aliases")
    func ukNationsAreAdmin1() {
        let hit = WeatherTool.parseGeocodingResponse(Self.london, location: "London", qualifiers: ["England"])
        #expect(hit?.admin1 == "England")
        #expect(WeatherTool.parseGeocodingResponse(Self.london, location: "London", qualifiers: ["Wales"]) == nil)
    }

    @Test("Canadian and Australian abbreviations use the candidate country")
    func candidateCountry() {
        let london = WeatherTool.parseGeocodingResponse(Self.london, location: "London", qualifiers: ["ON"])
        #expect(london?.admin1 == "Ontario")
        #expect(london?.countryCode == "CA")
        let perth = WeatherTool.parseGeocodingResponse(Self.perth, location: "Perth", qualifiers: ["WA"])
        #expect(perth?.admin1 == "Western Australia")
        #expect(perth?.countryCode == "AU")
        #expect(WeatherTool.parseGeocodingResponse(Self.perth, location: "Perth", qualifiers: ["WA"], country: "GB") == nil)
    }

    @Test("Explicit admin1 accepts abbreviations while explicit country does not")
    func explicitArgumentsAndUnmatchedQualifiers() {
        let hit = WeatherTool.parseGeocodingResponse(Self.chicago, location: "Chicago", admin1: "IL")
        #expect(hit?.admin1 == "Illinois")
        #expect(WeatherTool.parseGeocodingResponse(Self.chicago, location: "Chicago", qualifiers: ["TX"]) == nil)
        #expect(WeatherTool.parseGeocodingResponse(Self.chicago, location: "Chicago", country: "IL") == nil)
    }

    @Test("US aliases are normalized before matching", arguments: [
        "USA", "US", "U.S.", "United States", "United States of America",
    ])
    func usCountryAliases(alias: String) {
        let hit = WeatherTool.parseGeocodingResponse(
            Self.chicago, location: "Chicago", qualifiers: [alias], country: alias
        )
        #expect(hit?.countryCode == "US")
    }

    @Test("UAE is a country alias")
    func uaeAlias() {
        let data = Data(#"""
        {"results":[{"name":"Dubai","latitude":25.08,"longitude":55.31,"country":"United Arab Emirates","country_code":"AE","admin1":"Dubai","population":3000000,"timezone":"Asia/Dubai"}]}
        """#.utf8)
        #expect(WeatherTool.parseGeocodingResponse(data, location: "Dubai", country: "UAE")?.countryCode == "AE")
    }

    @Test("Subdivision expansion requires a country code")
    func missingCountryCode() {
        let data = Data(#"""
        {"results":[{"name":"Chicago","latitude":41.85,"longitude":-87.65,"country":"United States","admin1":"Illinois","population":2695598,"timezone":"America/Chicago"}]}
        """#.utf8)
        #expect(WeatherTool.parseGeocodingResponse(data, location: "Chicago", qualifiers: ["IL"]) == nil)
        #expect(WeatherTool.parseGeocodingResponse(data, location: "Chicago", admin1: "IL") == nil)
    }
}
