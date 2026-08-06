import Foundation

/// Current weather lookup via two Open-Meteo endpoints:
///
///   1. ``geocoding-api.open-meteo.com/v1/search`` resolves the place.
///   2. ``api.open-meteo.com/v1/forecast`` returns current temperature,
///      relative humidity, wind speed, and the WMO weather code.
///
/// No API key is required. Falls back to a clean error result
/// rather than throwing so the chat loop continues.
enum WeatherTool {
    static let definition = ToolDefinition(
        name: "weather",
        description: "Get the current weather for a city or place. Preserve the user's place name, and include country or state/province when known. Ambiguous places are not guessed.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "location": .object([
                    "type": .string("string"),
                    "description": .string("City or place name in the user's language, optionally followed by region/country, e.g. '西安', 'Springfield, Illinois', or 'Paris, France'.")
                ]),
                "country": .object([
                    "type": .string("string"),
                    "description": .string("Optional country name or two-letter country code used to disambiguate the place.")
                ]),
                "admin1": .object([
                    "type": .string("string"),
                    "description": .string("Optional state, province, or first-level administrative region used to disambiguate the place.")
                ]),
                "units": .object([
                    "type": .string("string"),
                    "description": .string("Either 'metric' (Celsius, km/h) or 'imperial' (Fahrenheit, mph). Defaults to metric."),
                    "enum": .array([.string("metric"), .string("imperial")])
                ])
            ]),
            "required": .array([.string("location")])
        ])
    )

    struct Args: Decodable {
        let location: String
        let country: String?
        let admin1: String?
        let units: String?
    }

    static func run(arguments: String) async -> ToolCallResult {
        let toolName = "weather"
        guard let data = arguments.data(using: .utf8),
              let args = try? JSONDecoder().decode(Args.self, from: data) else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: could not parse arguments JSON", isError: true)
        }
        let location = args.location.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !location.isEmpty else {
            return ToolCallResult(toolCallID: "", content: "\(toolName) error: empty location", isError: true)
        }
        let imperial = (args.units?.lowercased() == "imperial")
        do {
            let geo = try await geocode(
                location: location,
                country: cleaned(args.country),
                admin1: cleaned(args.admin1)
            )
            guard let hit = geo else {
                return ToolCallResult(
                    toolCallID: "",
                    content: "\(toolName): could not uniquely identify \"\(location)\". Add a country or state/province and try again.",
                    isError: false
                )
            }
            let weather = try await fetchCurrentWeather(lat: hit.latitude, lon: hit.longitude, imperial: imperial)
            let tempUnit = imperial ? "°F" : "°C"
            let windUnit = imperial ? "mph" : "km/h"
            var lines: [String] = []
            lines.append("Current weather for \(hit.fullName):")
            lines.append("  Conditions: \(weatherCodeLabel(weather.weatherCode))")
            lines.append(String(format: "  Temperature: %.1f%@", weather.temperature, tempUnit))
            lines.append("  Humidity: \(weather.humidity)%")
            lines.append(String(format: "  Wind: %.1f %@", weather.windSpeed, windUnit))
            lines.append("Data: Open-Meteo (timezone: \(hit.timezone ?? "n/a"))")
            return ToolCallResult(toolCallID: "", content: lines.joined(separator: "\n"), isError: false)
        } catch {
            return ToolCallResult(
                toolCallID: "",
                content: "\(toolName) error: \(error.localizedDescription)",
                isError: true
            )
        }
    }

    // MARK: - Geocoding

    struct GeoHit: Decodable {
        let name: String
        let latitude: Double
        let longitude: Double
        let country: String?
        let admin1: String?
        let timezone: String?
        let countryCode: String?
        let population: Int?

        enum CodingKeys: String, CodingKey {
            case name, latitude, longitude, country, admin1, timezone, population
            case countryCode = "country_code"
        }

        var fullName: String {
            placeLabel(name, admin1, country, fallback: name)
        }
    }

    static func geocode(location: String, country: String?, admin1: String?) async throws -> GeoHit? {
        let query = geocodingQuery(location: location, country: country, admin1: admin1)
        guard !query.name.isEmpty, let url = geocodingURL(location: query.name) else {
            throw NSError(domain: "WeatherTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not build geocoding URL"])
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        let (data, response) = try await cappedData(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "WeatherTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "geocoding HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"])
        }
        return parseGeocodingResponse(
            data,
            location: query.name,
            qualifiers: query.qualifiers
        )
    }

    static func geocodingURL(location: String) -> URL? {
        var components = URLComponents(string: "https://geocoding-api.open-meteo.com/v1/search")!
        components.queryItems = [
            URLQueryItem(name: "name", value: location),
            URLQueryItem(name: "count", value: "5"),
            URLQueryItem(name: "format", value: "json"),
        ]
        if let language = geocodingLanguage(for: location) {
            components.queryItems?.append(URLQueryItem(name: "language", value: language))
        }
        return components.url
    }

    static func parseGeocodingResponse(
        _ data: Data,
        location: String,
        qualifiers: [String] = []
    ) -> GeoHit? {
        struct Response: Decodable {
            let results: [GeoHit]?
        }
        let candidates = (try? JSONDecoder().decode(Response.self, from: data))?.results ?? []
        return selectGeocodingHit(location: location, qualifiers: qualifiers, candidates: candidates)
    }

    static func selectGeocodingHit(
        location: String,
        qualifiers: [String],
        candidates: [GeoHit]
    ) -> GeoHit? {
        let requestedName = matchKey(location)
        guard !requestedName.isEmpty else { return nil }

        let requestedQualifiers = qualifiers.compactMap { qualifier -> Set<String>? in
            let key = matchKey(qualifier)
            guard !key.isEmpty else { return nil }
            let countryCodes = countryCodes(for: qualifier).map(matchKey)
            return Set([key] + countryCodes)
        }
        let matches = candidates.filter { candidate in
            guard matchKey(candidate.name) == requestedName else { return false }
            let fields = Set([candidate.admin1, candidate.country, candidate.countryCode]
                .compactMap { $0 }
                .map(matchKey))
            return requestedQualifiers.allSatisfy { !$0.isDisjoint(with: fields) }
        }
        guard !matches.isEmpty else { return nil }

        // Rank same-named candidates by population so a well-known city
        // dominates its tiny homonyms (Tokyo over the villages).
        let ranked = matches.sorted { ($0.population ?? 0) > ($1.population ?? 0) }
        guard let first = ranked.first else { return nil }
        let firstPopulation = first.population ?? 0

        // With more than one same-named place, require a decisive population
        // margin, otherwise the name is genuinely ambiguous (e.g. the US
        // Springfields) and we return not-found rather than defaulting to API
        // order.
        if ranked.count > 1 {
            let runnerUpPopulation = ranked[1].population ?? 0
            // ``population`` is untrusted JSON; ``max(1, runnerUp) * 5`` traps on
            // a value near ``Int.max``. Report overflow instead of crashing —
            // an overflowing threshold is unreachable, so the name is ambiguous.
            let (threshold, overflow) = max(1, runnerUpPopulation).multipliedReportingOverflow(by: 5)
            guard !overflow, firstPopulation >= threshold else { return nil }
        }

        // Accept the top candidate when the user gave an explicit qualifier, or
        // spelled the name exactly (accents and all). When the spelling only
        // matches after folding accents/punctuation, require a major-city
        // population floor: a bare "Xian" (only fold-match: Xián, Spain, pop ~0)
        // stays not-found, while "Medellin" → Medellín, Colombia (pop ~2M) and
        // "Sao Paulo" → São Paulo, Brazil still resolve. This replaces an
        // earlier accent-sensitive hard-filter that dropped the real accented
        // city and kept only the exact-ASCII homonym (see #584).
        if !requestedQualifiers.isEmpty || literalKey(first.name) == literalKey(location) {
            return first
        }
        return firstPopulation >= 100_000 ? first : nil
    }

    private struct GeocodingQuery {
        let name: String
        let qualifiers: [String]
    }

    private static func geocodingQuery(
        location: String,
        country: String?,
        admin1: String?
    ) -> GeocodingQuery {
        let parts = location.split(separator: ",", omittingEmptySubsequences: true)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let name = parts.first ?? location.trimmingCharacters(in: .whitespacesAndNewlines)
        let qualifiers = Array(parts.dropFirst()) + [admin1, country].compactMap(cleaned)
        return GeocodingQuery(name: name, qualifiers: qualifiers)
    }

    private static func geocodingLanguage(for location: String) -> String? {
        for scalar in location.unicodeScalars {
            switch scalar.value {
            case 0xAC00...0xD7AF: return "ko"
            case 0x3040...0x30FF: return "ja"
            case 0x3400...0x4DBF, 0x4E00...0x9FFF: return "zh"
            default: continue
            }
        }
        return nil
    }

    private static func matchKey(_ value: String) -> String {
        let folded = value.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: Locale(identifier: "en_US_POSIX")
        )
        return String(folded.unicodeScalars.filter(CharacterSet.alphanumerics.contains))
            .lowercased()
    }

    private static func literalKey(_ value: String) -> String {
        value
            .replacingOccurrences(of: "’", with: "'")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    private static func countryCodes(for value: String) -> [String] {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count == 2, trimmed.unicodeScalars.allSatisfy(CharacterSet.letters.contains) {
            return [trimmed.uppercased()]
        }
        let requested = matchKey(trimmed)
        guard !requested.isEmpty else { return [] }
        let english = Locale(identifier: "en_US")
        let codes = Locale.Region.isoRegions.compactMap { region -> String? in
            let code = region.identifier
            guard code.count == 2,
                  let name = english.localizedString(forRegionCode: code) else {
                return nil
            }
            let localized = matchKey(name)
            return localized == requested
                || localized.hasPrefix(requested)
                || requested.hasPrefix(localized) ? code : nil
        }
        // Prefix matching handles CLDR labels such as "China mainland", but
        // only when the result is unique. "Congo" deliberately maps to none.
        return codes.count == 1 ? codes : []
    }

    private static func cleaned(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    static func placeLabel(_ parts: String?..., fallback: String) -> String {
        var unique: [String] = []
        for part in parts {
            guard let part else { continue }
            let trimmed = part.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, !unique.contains(trimmed) {
                unique.append(trimmed)
            }
        }
        return unique.isEmpty ? fallback : unique.joined(separator: ", ")
    }

    // MARK: - Weather fetch

    struct WeatherReading: Decodable {
        let temperature: Double
        let humidity: Int
        let windSpeed: Double
        let weatherCode: Int

        enum CodingKeys: String, CodingKey {
            case temperature = "temperature_2m"
            case humidity = "relative_humidity_2m"
            case windSpeed = "wind_speed_10m"
            case weatherCode = "weather_code"
        }
    }

    static func fetchCurrentWeather(lat: Double, lon: Double, imperial: Bool) async throws -> WeatherReading {
        guard let url = forecastURL(lat: lat, lon: lon, imperial: imperial) else {
            throw NSError(domain: "WeatherTool", code: 3, userInfo: [NSLocalizedDescriptionKey: "could not build forecast URL"])
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 10
        let (data, response) = try await cappedData(for: req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "WeatherTool", code: 4, userInfo: [NSLocalizedDescriptionKey: "weather HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"])
        }
        guard let reading = parseForecastResponse(data) else {
            throw NSError(domain: "WeatherTool", code: 5, userInfo: [NSLocalizedDescriptionKey: "unrecognised forecast payload"])
        }
        return reading
    }

    static func forecastURL(lat: Double, lon: Double, imperial: Bool) -> URL? {
        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(lat)),
            URLQueryItem(name: "longitude", value: String(lon)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"),
            URLQueryItem(name: "temperature_unit", value: imperial ? "fahrenheit" : "celsius"),
            URLQueryItem(name: "wind_speed_unit", value: imperial ? "mph" : "kmh"),
        ]
        return components.url
    }

    static func parseForecastResponse(_ data: Data) -> WeatherReading? {
        struct Response: Decodable {
            let current: WeatherReading
        }
        return (try? JSONDecoder().decode(Response.self, from: data))?.current
    }

    /// WMO weather-code → human label. Open-Meteo documents these
    /// at https://open-meteo.com/en/docs.
    static func weatherCodeLabel(_ code: Int) -> String {
        switch code {
        case 0: return "Clear sky"
        case 1: return "Mainly clear"
        case 2: return "Partly cloudy"
        case 3: return "Overcast"
        case 45, 48: return "Fog"
        case 51: return "Light drizzle"
        case 53: return "Moderate drizzle"
        case 55: return "Dense drizzle"
        case 56, 57: return "Freezing drizzle"
        case 61: return "Light rain"
        case 63: return "Moderate rain"
        case 65: return "Heavy rain"
        case 66, 67: return "Freezing rain"
        case 71: return "Light snow"
        case 73: return "Moderate snow"
        case 75: return "Heavy snow"
        case 77: return "Snow grains"
        case 80: return "Light rain showers"
        case 81: return "Moderate rain showers"
        case 82: return "Violent rain showers"
        case 85, 86: return "Snow showers"
        case 95: return "Thunderstorm"
        case 96, 99: return "Thunderstorm with hail"
        default: return "Code \(code)"
        }
    }
}
