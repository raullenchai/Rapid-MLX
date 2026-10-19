import Foundation
import Testing
@testable import Rapid

@Suite("Weather Chinese fallback and optional location")
struct WeatherChineseFallbackTests {
    private actor Requests {
        var urls: [URL] = []
        func record(_ url: URL) { urls.append(url) }
    }

    private static func payload(
        name: String, admin1: String, code: String = "CN", population: Int = 679535
    ) -> Data {
        Data("""
        {"results":[{"name":"\(name)","latitude":37.45,"longitude":116.3,
        "admin1":"\(admin1)","country":"\(code == "CN" ? "中国" : "日本")",
        "country_code":"\(code)","population":\(population)}]}
        """.utf8)
    }

    private static let villages = Data(#"""
    {"results":[
      {"name":"德州","latitude":30,"longitude":104,"admin1":"四川","country_code":"CN"},
      {"name":"德州","latitude":26,"longitude":118,"admin1":"福建省","country_code":"CN"}
    ]}
    """#.utf8)

    @Test("Dezhou retries with localized Chinese identity", arguments: [
        nil, "山东省", "Shandong Province", "山东", "Shandong", "安徽省", "Anhui",
    ] as [String?])
    func dezhou(admin1: String?) async throws {
        let requests = Requests()
        let hit = try await WeatherTool.geocode(location: "德州", country: nil, admin1: admin1) { request in
            let url = try #require(request.url)
            await requests.record(url)
            let items = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
            let name = items.first { $0.name == "name" }?.value
            let language = items.first { $0.name == "language" }?.value
            let data: Data
            if name == "dezhou" {
                #expect(language == "zh")
                data = Self.payload(name: "德州市", admin1: "山东")
            } else {
                #expect(name == "德州")
                #expect(language == "zh")
                data = Self.villages
            }
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(await requests.urls.count == 2)
        if admin1 == "安徽省" || admin1 == "Anhui" {
            #expect(hit == nil)
        } else {
            #expect(hit?.fullName == "德州市, 山东, 中国")
        }
    }

    @Test("Romanized qualifiers preserve Chinese and English countries", arguments: ["中国", "China", "CN", "中國"])
    func commaQualifiers(country: String) async throws {
        let hit = try await WeatherTool.geocode(location: "德州, 山东省, \(country)", country: country, admin1: nil) { request in
            let url = try #require(request.url)
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
            #expect(items?.first { $0.name == "language" }?.value == "zh")
            let data = items?.first { $0.name == "name" }?.value == "dezhou"
                ? Self.payload(name: "德州市", admin1: "山东") : Self.villages
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(hit?.admin1 == "山东")
    }

    @Test("Romanized fallback distinguishes Han administrative homophones", arguments: [
        ("陕西省", false), ("山西省", true), ("Shanxi", true),
    ])
    func adminHomophones(admin1: String, accepted: Bool) async throws {
        for commaQualifier in [false, true] {
            let requests = Requests()
            let hit = try await WeatherTool.geocode(
                location: commaQualifier ? "临汾, \(admin1)" : "临汾",
                country: nil, admin1: commaQualifier ? nil : admin1
            ) { request in
                let url = try #require(request.url)
                await requests.record(url)
                let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
                #expect(items?.first { $0.name == "language" }?.value == "zh")
                let data = items?.first { $0.name == "name" }?.value == "linfen"
                    ? Self.payload(name: "临汾市", admin1: "山西", population: 959198)
                    : Data(#"{"results":[]}"#.utf8)
                return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
            }
            #expect((hit != nil) == accepted)
            if accepted { #expect(hit?.admin1 == "山西") }
            #expect(await requests.urls.last.map {
                URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "name" }?.value
            } == "linfen")
        }
    }

    @Test("Romanized fallback preserves the official Shaanxi spelling", arguments: [
        ("Shanxi Province", false), ("Shaanxi", true), ("陕西省", true),
    ])
    func shaanxiAdmin(admin1: String, accepted: Bool) async throws {
        for actual in ["陕西", "陝西"] {
            for commaQualifier in [false, true] {
                let requests = Requests()
                let hit = try await WeatherTool.geocode(
                    location: commaQualifier ? "渭南, \(admin1)" : "渭南",
                    country: nil, admin1: commaQualifier ? nil : admin1
                ) { request in
                    let url = try #require(request.url)
                    await requests.record(url)
                    let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
                    #expect(items?.first { $0.name == "language" }?.value == "zh")
                    let name = items?.first { $0.name == "name" }?.value
                    let data = name == "weinan"
                        ? Self.payload(name: "渭南市", admin1: actual, population: 468941)
                        : Data(#"{"results":[]}"#.utf8)
                    return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
                }
                #expect((hit != nil) == accepted)
                if accepted { #expect(hit?.admin1 == actual) }
                let names = await requests.urls.map {
                    URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "name" }?.value
                }
                #expect(names == ["渭南", "weinan"])
            }
        }
    }

    @Test("Romanized fallback rejects small and foreign namesakes", arguments: [(99999, "CN"), (900000, "JP")])
    func unsafeNamesakes(population: Int, code: String) async throws {
        let hit = try await WeatherTool.geocode(location: "德州", country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
            #expect(items?.first { $0.name == "language" }?.value == "zh")
            let data = items?.first { $0.name == "name" }?.value == "dezhou"
                ? Self.payload(name: "德州市", admin1: "山东", code: code, population: population) : Self.villages
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(hit == nil)
    }

    @Test("Romanized fallback rejects Han homophones", arguments: [
        ("彬州", "binzhou", "滨州市", "山东", 682717),
        ("光州", "guangzhou", "广州", "广东", 18676605),
        ("济州", "jizhou", "冀州", "河北省", 362013),
    ])
    func homophones(original: String, latin: String, localized: String, admin1: String, population: Int) async throws {
        let requests = Requests()
        let hit = try await WeatherTool.geocode(location: original, country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            await requests.record(url)
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
            #expect(items?.first { $0.name == "language" }?.value == "zh")
            let data = items?.first { $0.name == "name" }?.value == latin
                ? Self.payload(name: localized, admin1: admin1, population: population)
                : Data(#"{"results":[]}"#.utf8)
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(hit == nil)
        #expect(await requests.urls.last.map {
            URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "name" }?.value
        } == latin)
    }

    @Test("Localized retry identity normalizes scripts and only city or county suffixes", arguments: [
        ("苏州", "蘇州市", true), ("蘇州", "苏州市", true),
        ("新竹", "新竹縣", true), ("德州市", "德州", true),
        ("德州", "德州市", true), ("德州", "德州省", false),
        ("上市", "上", false), ("德州", "Dezhou", false),
    ])
    func localizedIdentity(original: String, localized: String, accepted: Bool) {
        let hit = WeatherTool.parseGeocodingResponse(
            Self.payload(name: localized, admin1: "地区"),
            location: original, romanized: true
        )
        #expect((hit != nil) == accepted)
    }

    @Test("Successful Han queries make no retry", arguments: ["东京", "北京"])
    func firstQuery(name: String) async throws {
        let requests = Requests()
        let hit = try await WeatherTool.geocode(location: name, country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            await requests.record(url)
            return (Self.payload(name: name, admin1: "地区"),
                    try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(hit?.name == name)
        #expect(await requests.urls.count == 1)
    }

    @Test("Traditional retry precedes Latin retry and apostrophes are removed", arguments: [
        ("苏州", "蘇州", "suzhou"), ("西安", "西安", "xian"),
    ])
    func retryOrder(original: String, traditional: String, latin: String) async throws {
        let requests = Requests()
        let hit = try await WeatherTool.geocode(location: original, country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            await requests.record(url)
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
            #expect(items?.first { $0.name == "language" }?.value == "zh")
            let name = items?.first { $0.name == "name" }?.value
            let data = name == latin
                ? Self.payload(name: traditional + "市", admin1: "陕西") : Data(#"{"results":[]}"#.utf8)
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        let names = await requests.urls.map {
            URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems?.first { $0.name == "name" }?.value
        }
        #expect(names == (original == traditional ? [original, latin] : [original, traditional, latin]))
        #expect(hit?.name == traditional + "市")
    }

    @Test("Latin retry preserves ambiguity rejection")
    func romanizedAmbiguity() async throws {
        let hit = try await WeatherTool.geocode(location: "德州", country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems
            #expect(items?.first { $0.name == "language" }?.value == "zh")
            let data = items?.first { $0.name == "name" }?.value == "dezhou" ? Data(#"""
            {"results":[
              {"name":"德州市","latitude":37,"longitude":116,"admin1":"山东","country_code":"CN","population":679535},
              {"name":"德州市","latitude":30,"longitude":104,"admin1":"四川","country_code":"CN","population":200000}
            ]}
            """#.utf8) : Self.villages
            return (data, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        #expect(hit == nil)
    }

    @Test("Administrative suffixes match in either direction", arguments: [
        ("福州", "福建省", "福建"), ("福州", "福建", "福建省"),
        ("Austin", "Texas", "Texas State"), ("Austin", "Texas State", "Texas"),
        ("London", "Ontario", "Ontario Province"), ("London", "Ontario Province", "Ontario"),
        ("福州", "福建", "福建州"), ("福州", "福建", "福建市"), ("福州", "福建", "福建县"),
        ("福州", "福建", "福建自治区"), ("福州", "福建", "福建特别行政区"),
        ("Fuzhou", "Fujian", "Fujian Prefecture"), ("Fuzhou", "Fujian", "Fujian Sheng"),
        ("Fuzhou", "Fujian", "Fujian Shi"),
    ])
    func suffixes(name: String, actual: String, requested: String) {
        let data = Self.payload(name: name, admin1: actual)
        #expect(WeatherTool.parseGeocodingResponse(data, location: name, qualifiers: [requested]) != nil)
        #expect(WeatherTool.parseGeocodingResponse(data, location: name, admin1: requested) != nil)
    }

    @Test("Administrative normalization never shortens a two-character name or matches city names")
    func preserveNames() {
        #expect(WeatherTool.parseGeocodingResponse(Self.payload(name: "德州", admin1: "德州"), location: "德州", admin1: "德") == nil)
        #expect(WeatherTool.parseGeocodingResponse(Self.payload(name: "福州市", admin1: "福建"), location: "福州") == nil)
        #expect(WeatherTool.parseGeocodingResponse(Self.payload(name: "Austin", admin1: "Texas"), location: "Austin", admin1: "Texasstate") == nil)
    }

    @Test("Optional location uses the first nonblank argument", arguments: [
        (#"{"admin1":"Texas","units":"metric"}"#, "Texas"),
        (#"{"location":"  ","admin1":" Texas ","country":"US"}"#, "Texas"),
        (#"{"location":"Austin","admin1":"Texas"}"#, "Austin"),
        (#"{"admin1":" ","country":" China "}"#, "China"),
    ])
    func effectiveLocation(json: String, expected: String) throws {
        let args = try JSONDecoder().decode(WeatherTool.Args.self, from: Data(json.utf8))
        #expect(WeatherTool.effectiveLocation(args) == expected)
        if expected == "Texas" { #expect(args.admin1?.contains("Texas") == true) }
    }

    @Test("Admin1-only Texas follows upstream region guidance")
    func admin1OnlyRegion() async throws {
        let args = try JSONDecoder().decode(
            WeatherTool.Args.self, from: Data(#"{"admin1":"Texas"}"#.utf8)
        )
        let location = try #require(WeatherTool.effectiveLocation(args))
        let resolution = try await WeatherTool.resolvePlace(
            location: location, country: args.country, admin1: args.admin1
        ) { request in
            let url = try #require(request.url)
            let name = URLComponents(url: url, resolvingAgainstBaseURL: false)?
                .queryItems?.first { $0.name == "name" }?.value
            #expect(name == "Texas")
            return (Self.payload(name: "Texas", admin1: "Hidalgo", code: "MX", population: 993),
                    try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        guard case .region = resolution else {
            Issue.record("Expected Texas to resolve as a region")
            return
        }
        switch WeatherTool.unresolvedResult(
            for: resolution, location: location, country: args.country, admin1: args.admin1
        ) {
        case .success:
            Issue.record("Expected region guidance instead of a place")
        case .failure(let unresolved):
            #expect(unresolved.result.isError)
            #expect(unresolved.result.content.contains("Texas, United States"))
            #expect(unresolved.result.content.contains("state or region"))
            #expect(unresolved.result.content.contains("Do not retry weather"))
        }
    }

    @Test("Missing and malformed arguments retain distinct errors")
    func argumentErrors() async {
        for json in [#"{"units":"metric"}"#, #"{"location":" ","admin1":"","country":" "}"#] {
            let result = await WeatherTool.run(arguments: json)
            #expect(result.isError)
            #expect(result.content == #"weather error: missing required "location" argument. Pass a city name, for example {"location": "Austin, Texas"}."#)
        }
        let result = await WeatherTool.run(arguments: "{")
        #expect(result.isError)
        #expect(result.content == "weather error: could not parse arguments JSON")
    }

    @Test("A failed romanized retry falls back to unresolved-place handling")
    func romanizedRetryFailureIsBestEffort() async throws {
        let resolution = try await WeatherTool.resolvePlace(location: "德州", country: nil, admin1: nil) { request in
            let url = try #require(request.url)
            let name = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?
                .first { $0.name == "name" }?.value
            if name == "dezhou" { throw URLError(.timedOut) }
            return (Self.villages, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
        }
        guard case .place = resolution else { return }
        Issue.record("expected an unresolved resolution")
    }

    @Test("Cancellation during the romanized retry still propagates")
    func romanizedRetryCancellationPropagates() async throws {
        await #expect(throws: CancellationError.self) {
            _ = try await WeatherTool.resolvePlace(location: "德州", country: nil, admin1: nil) { request in
                let url = try #require(request.url)
                let name = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?
                    .first { $0.name == "name" }?.value
                if name == "dezhou" { throw CancellationError() }
                return (Self.villages, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
            }
        }
    }

    @Test("URL-loading cancellation during the romanized retry still propagates")
    func romanizedRetryURLCancellationPropagates() async throws {
        await #expect(throws: URLError.self) {
            _ = try await WeatherTool.resolvePlace(location: "德州", country: nil, admin1: nil) { request in
                let url = try #require(request.url)
                let name = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?
                    .first { $0.name == "name" }?.value
                if name == "dezhou" { throw URLError(.cancelled) }
                return (Self.villages, try #require(HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)))
            }
        }
    }
}
