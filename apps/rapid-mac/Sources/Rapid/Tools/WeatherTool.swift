import Foundation

/// Current weather and daily forecast via two Open-Meteo endpoints:
///
///   1. ``geocoding-api.open-meteo.com/v1/search`` resolves the place.
///   2. ``api.open-meteo.com/v1/forecast`` returns current temperature,
///      relative humidity, wind speed, WMO weather code, and daily forecasts.
///
/// No API key is required. Falls back to a clean error result
/// rather than throwing so the chat loop continues.
enum WeatherTool {
    static let definition = ToolDefinition(
        name: "weather",
        description: "Get current conditions AND a daily forecast up to 7 days for a city or place (today, tomorrow, the next few days, this weekend). Use this tool, not web_search, for these questions. Each forecast day has its local date and weekday. Preserve country or state/province qualifiers from the user's location; ambiguous places are not guessed.",
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
                "days": .object([
                    "type": .string("integer"),
                    "minimum": .number(1),
                    "maximum": .number(7),
                    "description": .string("Forecast days starting with the location's today: 2 covers tomorrow; 7 covers this weekend / the rest of the week. Defaults to 3; bounded to 1...7.")
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
        let days: Int

        enum CodingKeys: String, CodingKey {
            case location, country, admin1, units, days
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            location = try values.decode(String.self, forKey: .location)
            country = try values.decodeIfPresent(String.self, forKey: .country)
            admin1 = try values.decodeIfPresent(String.self, forKey: .admin1)
            units = try values.decodeIfPresent(String.self, forKey: .units)
            let number = (try? values.decode(Double.self, forKey: .days))
                ?? (try? values.decode(String.self, forKey: .days)).flatMap(Double.init)
            // Clamp before converting so even enormous finite values cannot overflow Int.
            days = number.flatMap { $0.isFinite ? Int(min(7, max(1, $0))) : nil } ?? 3
        }
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
                return unresolvedPlaceResult(location: location)
            }
            let forecast = try await fetchForecast(
                lat: hit.latitude, lon: hit.longitude, imperial: imperial, days: args.days
            )
            return ToolCallResult(toolCallID: "", content: formatReport(
                place: hit.fullName, timezone: forecast.timezone ?? hit.timezone,
                reading: forecast.current, daily: forecast.daily, imperial: imperial
            ), isError: false)
        } catch {
            return ToolCallResult(
                toolCallID: "",
                content: "\(toolName) error: \(error.localizedDescription)",
                isError: true
            )
        }
    }

    static func unresolvedPlaceResult(location: String) -> ToolCallResult {
        ToolCallResult(
            toolCallID: "",
            content: "weather: could not uniquely identify \"\(location)\". Add a country or state/province and try again.",
            isError: true
        )
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

    typealias GeocodingLoader = @Sendable (URLRequest) async throws -> (Data, URLResponse)

    static func geocode(
        location: String,
        country: String?,
        admin1: String?,
        load: @escaping GeocodingLoader = { try await cappedData(for: $0) }
    ) async throws -> GeoHit? {
        let query = geocodingQuery(location: location, country: country, admin1: admin1)
        guard !query.name.isEmpty else {
            throw NSError(domain: "WeatherTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not build geocoding URL"])
        }
        if let hit = try await geocodingHit(
            name: query.name, qualifiers: query.qualifiers,
            country: country, admin1: admin1, load: load
        ) {
            return hit
        }
        // The provider indexes some Chinese place names only under their
        // traditional spelling. Retry that spelling only after the original
        // name was unresolved, keeping explicit country/state qualifiers.
        guard let alternate = traditionalGeocodingName(query.name) else { return nil }
        return try await geocodingHit(
            name: alternate, qualifiers: query.qualifiers,
            country: country, admin1: admin1, load: load
        )
    }

    private static func geocodingHit(
        name: String,
        qualifiers: [String],
        country: String?,
        admin1: String?,
        load: GeocodingLoader
    ) async throws -> GeoHit? {
        guard let url = geocodingURL(location: name) else {
            throw NSError(domain: "WeatherTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "could not build geocoding URL"])
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        let (data, response) = try await load(request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "WeatherTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "geocoding HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"])
        }
        return parseGeocodingResponse(
            data,
            location: name,
            qualifiers: qualifiers,
            country: country,
            admin1: admin1
        )
    }

    static func traditionalGeocodingName(_ name: String) -> String? {
        guard let traditional = name.applyingTransform(
            StringTransform(rawValue: "Hans-Hant"), reverse: false
        ), traditional != name else { return nil }
        return traditional
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
        qualifiers: [String] = [],
        country: String? = nil,
        admin1: String? = nil
    ) -> GeoHit? {
        struct Response: Decodable {
            let results: [GeoHit]?
        }
        let candidates = (try? JSONDecoder().decode(Response.self, from: data))?.results ?? []
        return selectGeocodingHit(
            location: location, qualifiers: qualifiers,
            candidates: candidates, country: country, admin1: admin1
        )
    }

    static func selectGeocodingHit(
        location: String,
        qualifiers: [String],
        candidates: [GeoHit],
        country: String? = nil,
        admin1: String? = nil
    ) -> GeoHit? {
        let requestedName = matchKey(location)
        guard !requestedName.isEmpty else { return nil }

        let requestedQualifiers = qualifiers.compactMap { qualifier -> Set<String>? in
            let key = matchKey(qualifier)
            guard !key.isEmpty else { return nil }
            let countryCodes = countryCodes(for: qualifier).map(matchKey)
            let alternate = traditionalGeocodingName(qualifier).map(matchKey)
            return Set([key] + countryCodes + [alternate].compactMap { $0 })
        }
        let matches = candidates.filter { candidate in
            guard matchKey(candidate.name) == requestedName else { return false }
            let fields = Set([candidate.admin1, candidate.country, candidate.countryCode]
                .compactMap { $0 }
                .map(matchKey))
            guard requestedQualifiers.allSatisfy({ !$0.isDisjoint(with: fields) }) else {
                return false
            }
            // Explicit country/state arguments must match their own fields.
            // A namesake province cannot satisfy a requested country.
            if let country = cleaned(country) {
                let accepted = Set([matchKey(country)]
                    + countryCodes(for: country).map(matchKey)
                    + [traditionalGeocodingName(country).map(matchKey)].compactMap { $0 })
                let actual = Set([candidate.country, candidate.countryCode]
                    .compactMap { $0 }.map(matchKey))
                guard !accepted.isDisjoint(with: actual) else { return false }
            }
            if let admin1 = cleaned(admin1) {
                let accepted = Set([matchKey(admin1)]
                    + [traditionalGeocodingName(admin1).map(matchKey)].compactMap { $0 })
                guard let actual = candidate.admin1.map(matchKey),
                      accepted.contains(actual) else { return false }
            }
            return true
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

    static func fetchForecast(
        lat: Double, lon: Double, imperial: Bool, days: Int = 3,
        load: GeocodingLoader = { try await cappedData(for: $0) }
    ) async throws -> ForecastResponse {
        guard let url = forecastURL(lat: lat, lon: lon, imperial: imperial, days: days) else {
            throw NSError(domain: "WeatherTool", code: 3, userInfo: [NSLocalizedDescriptionKey: "could not build forecast URL"])
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 10
        let (data, response) = try await load(req)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw NSError(domain: "WeatherTool", code: 4, userInfo: [NSLocalizedDescriptionKey: "weather HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"])
        }
        guard let reading = try? JSONDecoder().decode(ForecastResponse.self, from: data) else {
            throw NSError(domain: "WeatherTool", code: 5, userInfo: [NSLocalizedDescriptionKey: "unrecognised forecast payload"])
        }
        return reading
    }

    static func forecastURL(lat: Double, lon: Double, imperial: Bool, days: Int = 3) -> URL? {
        var components = URLComponents(string: "https://api.open-meteo.com/v1/forecast")!
        components.queryItems = [
            URLQueryItem(name: "latitude", value: String(lat)),
            URLQueryItem(name: "longitude", value: String(lon)),
            URLQueryItem(name: "current", value: "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"),
            URLQueryItem(name: "daily", value: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum"),
            URLQueryItem(name: "timezone", value: "auto"),
            URLQueryItem(name: "forecast_days", value: String(min(7, max(1, days)))),
            URLQueryItem(name: "precipitation_unit", value: imperial ? "inch" : "mm"),
            URLQueryItem(name: "temperature_unit", value: imperial ? "fahrenheit" : "celsius"),
            URLQueryItem(name: "wind_speed_unit", value: imperial ? "mph" : "kmh"),
        ]
        return components.url
    }

    static func parseForecastResponse(_ data: Data) -> WeatherReading? {
        (try? JSONDecoder().decode(ForecastResponse.self, from: data))?.current
    }

    struct ForecastResponse: Decodable {
        let current: WeatherReading
        let timezone: String?
        let daily: DailyForecast?

        enum CodingKeys: String, CodingKey { case current, timezone, daily }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            current = try values.decode(WeatherReading.self, forKey: .current)
            timezone = try? values.decode(String.self, forKey: .timezone)
            // A broken optional forecast must never discard valid current conditions.
            daily = try? values.decode(DailyForecast.self, forKey: .daily)
        }
    }

    struct DailyForecast: Decodable {
        let time: [String?]
        let weatherCode: [Int?]
        let maximum: [Double?]
        let minimum: [Double?]
        let precipitationProbability: [Double?]
        let precipitation: [Double?]

        enum CodingKeys: String, CodingKey {
            case time
            case weatherCode = "weather_code"
            case maximum = "temperature_2m_max"
            case minimum = "temperature_2m_min"
            case precipitationProbability = "precipitation_probability_max"
            case precipitation = "precipitation_sum"
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            time = try values.decode([String?].self, forKey: .time)
            weatherCode = try values.decode([Int?].self, forKey: .weatherCode)
            maximum = try values.decode([Double?].self, forKey: .maximum)
            minimum = try values.decode([Double?].self, forKey: .minimum)
            precipitationProbability = try values.decodeIfPresent([Double?].self, forKey: .precipitationProbability) ?? []
            precipitation = try values.decodeIfPresent([Double?].self, forKey: .precipitation) ?? []
        }

        var count: Int {
            [time.count, weatherCode.count, maximum.count, minimum.count].min() ?? 0
        }
    }

    /// Parse date-only values in a fixed Gregorian/GMT calendar, never the Mac's calendar.
    static func weekday(for date: String) -> String? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.isLenient = false
        guard let parsed = formatter.date(from: date),
              formatter.string(from: parsed) == date else { return nil }
        formatter.dateFormat = "EEE"
        return formatter.string(from: parsed)
    }

    static func formatReport(
        place: String, timezone: String?, reading: WeatherReading,
        daily: DailyForecast?, imperial: Bool
    ) -> String {
        let tempUnit = imperial ? "°F" : "°C"
        let windUnit = imperial ? "mph" : "km/h"
        let precipitationUnit = imperial ? "in" : "mm"
        let zone = timezone ?? "n/a"
        var lines: [String] = []
        lines.append("Current weather for \(place):")
        lines.append("  Conditions: \(weatherCodeLabel(reading.weatherCode))")
        lines.append(String(format: "  Temperature: %.1f%@", reading.temperature, tempUnit))
        lines.append("  Humidity: \(reading.humidity)%")
        lines.append(String(format: "  Wind: %.1f %@", reading.windSpeed, windUnit))
        if let daily {
            var days: [String] = []
            func number(_ value: Double?, decimals: Int = 1) -> String {
                value.map { String(format: "%.*f", locale: Locale(identifier: "en_US_POSIX"), decimals, $0) } ?? "n/a"
            }
            for index in 0..<min(7, daily.count) {
                guard let date = daily.time[index], !date.isEmpty else { continue }
                let weekday = weekday(for: date) ?? "n/a"
                let relative = index == 0 ? " (today)" : index == 1 ? " (tomorrow)" : ""
                let conditions = daily.weatherCode[index].map(weatherCodeLabel) ?? "n/a"
                let probability = daily.precipitationProbability.indices.contains(index) ? daily.precipitationProbability[index] : nil
                let probabilityText = probability.map { "\(number($0, decimals: 0))%" } ?? "n/a"
                let precipitation = daily.precipitation.indices.contains(index) ? daily.precipitation[index] : nil
                days.append("  \(date) \(weekday)\(relative): \(conditions), \(number(daily.minimum[index]))–\(number(daily.maximum[index]))\(tempUnit), precip \(probabilityText), \(number(precipitation)) \(precipitationUnit)")
            }
            if !days.isEmpty {
                lines.append("Daily forecast (local dates, timezone \(zone)):")
                lines.append(contentsOf: days)
            }
        }
        lines.append("Data: Open-Meteo (timezone: \(zone))")
        return lines.joined(separator: "\n")
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
