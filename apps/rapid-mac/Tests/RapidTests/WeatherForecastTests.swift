import Foundation
import Testing
@testable import Rapid

@Suite("Weather forecast")
struct WeatherForecastTests {
    // Fixed Open-Meteo response shape; all tests run offline.
    static let fixture = """
    {"latitude":39.9,"longitude":116.4,"utc_offset_seconds":28800,
     "timezone":"Asia/Shanghai","elevation":49,
     "current":{"time":"2026-10-18T12:00","interval":900,"temperature_2m":19.4,
                "relative_humidity_2m":45,"wind_speed_10m":8.2,"weather_code":2},
     "daily_units":{"time":"iso8601","temperature_2m_max":"°C","temperature_2m_min":"°C",
                    "precipitation_probability_max":"%","precipitation_sum":"mm","weather_code":"wmo code"},
     "daily":{
       "time":["2026-10-18","2026-10-19","2026-10-20","2026-10-21","2026-10-22","2026-10-23","2026-10-24"],
       "weather_code":[2,61,3,0,1,45,80],
       "temperature_2m_max":[19.4,15.2,16,17,18,19,20],
       "temperature_2m_min":[8.1,7,8,9,10,11,12],
       "precipitation_probability_max":[10,80,null,0,0,10,40],
       "precipitation_sum":[0,3.4,0,0,0,0,1.2]}}
    """

    static func payload(_ modify: (inout [String: Any]) -> Void) throws -> Data {
        var json = try #require(JSONSerialization.jsonObject(with: Data(fixture.utf8)) as? [String: Any])
        modify(&json)
        return try JSONSerialization.data(withJSONObject: json)
    }

    static func report(_ data: Data, imperial: Bool = false) throws -> String {
        let response = try JSONDecoder().decode(WeatherTool.ForecastResponse.self, from: data)
        return WeatherTool.formatReport(place: "Beijing, China", timezone: response.timezone,
            reading: response.current, daily: response.daily, imperial: imperial)
    }

    @Test(arguments: [
        ("", 3), (",\"days\":0", 1), (",\"days\":-2", 1),
        (",\"days\":9", 7), (",\"days\":100", 7),
        (",\"days\":\"5\"", 5), (",\"days\":4.0", 4),
        (",\"days\":4.9", 4), (",\"days\":\"abc\"", 3),
        (",\"days\":null", 3), (",\"days\":true", 3),
        (",\"days\":[]", 3), (",\"days\":{}", 3),
        (",\"days\":1e100", 7), (",\"days\":\"NaN\"", 3),
        (",\"days\":\"Infinity\"", 3), (",\"days\":\"3.8\"", 3)
    ])
    func lenientDays(input: (String, Int)) throws {
        let args = try JSONDecoder().decode(WeatherTool.Args.self,
            from: Data("{\"location\":\"Beijing\"\(input.0)}".utf8))
        #expect(args.days == input.1)
    }

    @Test(arguments: [false, true])
    func url(imperial: Bool) throws {
        for (requested, expected) in [(3,3), (1,1), (7,7), (-2,1), (100,7)] {
            let url = try #require(WeatherTool.forecastURL(lat: 39.9, lon: 116.4, imperial: imperial, days: requested))
            let items = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
            let query = Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") })
            #expect(query["latitude"] == "39.9")
            #expect(query["longitude"] == "116.4")
            #expect(query["current"] == "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code")
            #expect(query["daily"] == "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum")
            #expect(query["timezone"] == "auto")
            #expect(query["forecast_days"] == String(expected))
            #expect(query["temperature_unit"] == (imperial ? "fahrenheit" : "celsius"))
            #expect(query["wind_speed_unit"] == (imperial ? "mph" : "kmh"))
            #expect(query["precipitation_unit"] == (imperial ? "inch" : "mm"))
        }
        #expect(WeatherTool.forecastURL(lat: 0, lon: 0, imperial: imperial)?.absoluteString.contains("forecast_days=3") == true)
    }

    @Test func metricReport() throws {
        let text = try Self.report(Data(Self.fixture.utf8))
        #expect(text.hasPrefix("""
        Current weather for Beijing, China:
          Conditions: Partly cloudy
          Temperature: 19.4°C
          Humidity: 45%
          Wind: 8.2 km/h
        Daily forecast (local dates, timezone Asia/Shanghai):
        """))
        #expect(text.contains("2026-10-18 Sun (today): Partly cloudy, 8.1–19.4°C, precip 10%, 0.0 mm"))
        #expect(text.contains("2026-10-19 Mon (tomorrow): Light rain, 7.0–15.2°C, precip 80%, 3.4 mm"))
        #expect(text.contains("2026-10-20 Tue: Overcast, 8.0–16.0°C, precip n/a, 0.0 mm"))
        #expect(text.contains("2026-10-24 Sat:"))
        #expect(text.hasSuffix("Data: Open-Meteo (timezone: Asia/Shanghai)"))
        #expect(WeatherTool.parseForecastResponse(Data(Self.fixture.utf8))?.temperature == 19.4)
    }

    @Test func imperialReport() throws {
        let data = try Self.payload { json in
            var current = json["current"] as! [String: Any]
            current["temperature_2m"] = 66.9
            current["wind_speed_10m"] = 5.1
            json["current"] = current
            var daily = json["daily"] as! [String: Any]
            daily["temperature_2m_min"] = [46.6]
            daily["temperature_2m_max"] = [66.9]
            daily["precipitation_sum"] = [0.1]
            json["daily"] = daily
        }
        let text = try Self.report(data, imperial: true)
        #expect(text.contains("Temperature: 66.9°F"))
        #expect(text.contains("Wind: 5.1 mph"))
        #expect(text.contains("46.6–66.9°F, precip 10%, 0.1 in"))
    }

    @Test(arguments: ["missing", "null", "wrongType", "wrongArray", "empty", "missingField"])
    func unusableDaily(kind: String) throws {
        let data = try Self.payload { json in
            switch kind {
            case "missing": json.removeValue(forKey: "daily")
            case "null": json["daily"] = NSNull()
            case "wrongType": json["daily"] = "broken"
            case "wrongArray":
                var daily = json["daily"] as! [String: Any]
                daily["weather_code"] = ["broken"]
                json["daily"] = daily
            case "empty":
                var daily = json["daily"] as! [String: Any]
                daily["time"] = []
                json["daily"] = daily
            default: json["daily"] = ["time": ["2026-10-18"]]
            }
            json.removeValue(forKey: "timezone")
        }
        let text = try Self.report(data)
        #expect(text.contains("Temperature: 19.4°C"))
        #expect(!text.contains("Daily forecast"))
        #expect(text.hasSuffix("Data: Open-Meteo (timezone: n/a)"))
    }

    @Test(arguments: ["precipitation_probability_max", "precipitation_sum"])
    func missingPrecipitationColumn(key: String) throws {
        let data = try Self.payload { json in
            var daily = json["daily"] as! [String: Any]
            daily.removeValue(forKey: key)
            json["daily"] = daily
        }
        let response = try JSONDecoder().decode(WeatherTool.ForecastResponse.self, from: data)
        #expect(response.daily?.count == 7)
        let text = try Self.report(data)
        #expect(text.contains("Daily forecast"))
        #expect(text.contains("2026-10-24 Sat:"))
        if key == "precipitation_probability_max" {
            #expect(text.contains("2026-10-19 Mon (tomorrow): Light rain, 7.0–15.2°C, precip n/a, 3.4 mm"))
        } else {
            #expect(text.contains("2026-10-19 Mon (tomorrow): Light rain, 7.0–15.2°C, precip 80%, n/a mm"))
        }
        #expect(!text.contains("precip n/a%"))
    }

    @Test func shortPrecipitationColumns() throws {
        let data = try Self.payload { json in
            var daily = json["daily"] as! [String: Any]
            daily["precipitation_probability_max"] = [10]
            daily["precipitation_sum"] = [0]
            json["daily"] = daily
        }
        let text = try Self.report(data)
        #expect(text.contains("2026-10-18 Sun (today): Partly cloudy, 8.1–19.4°C, precip 10%, 0.0 mm"))
        #expect(text.contains("2026-10-19 Mon (tomorrow): Light rain, 7.0–15.2°C, precip n/a, n/a mm"))
        #expect(text.contains("2026-10-24 Sat:"))
    }

    @Test func nullsAndUnequalArrays() throws {
        let data = try Self.payload { json in
            var daily = json["daily"] as! [String: Any]
            daily["time"] = [NSNull(), "2026-10-19", "", "bad-date", "2026-10-22"]
            daily["weather_code"] = [NSNull(), NSNull(), NSNull(), 999]
            daily["temperature_2m_min"] = [NSNull(), NSNull(), NSNull(), NSNull()]
            daily["temperature_2m_max"] = [NSNull(), NSNull(), NSNull(), NSNull()]
            daily["precipitation_sum"] = [NSNull(), NSNull(), NSNull(), NSNull()]
            json["daily"] = daily
        }
        let text = try Self.report(data)
        #expect(!text.contains("(today)"))
        #expect(text.contains("2026-10-19 Mon (tomorrow): n/a, n/a–n/a°C, precip 80%, n/a mm"))
        #expect(text.contains("bad-date n/a: Code 999"))
        #expect(!text.contains("2026-10-22"))
    }

    @Test(arguments: [("2026-10-18","Sun"), ("2026-10-24","Sat"), ("2026-01-01","Thu"),
                      ("2024-02-29","Thu"), ("2026-03-08","Sun"), ("2026-11-01","Sun")])
    func dateOnlyWeekdays(date: (String, String)) {
        #expect(WeatherTool.weekday(for: date.0) == date.1)
    }

    @Test(arguments: ["", "2026-02-30", "2026-10-18T23:59:00", "garbage"])
    func invalidDates(date: String) {
        #expect(WeatherTool.weekday(for: date) == nil)
    }

    @Test func missingDatesAndSevenDayCap() throws {
        let noDates = try Self.payload { json in
            var daily = json["daily"] as! [String: Any]
            daily["time"] = [NSNull(), NSNull()]
            json["daily"] = daily
        }
        #expect(try !Self.report(noDates).contains("Daily forecast"))
        let extra = try Self.payload { json in
            var daily = json["daily"] as! [String: Any]
            for (key, value) in daily {
                var array = value as! [Any]
                array.append(key == "time" ? "2026-10-25" : 0)
                daily[key] = array
            }
            json["daily"] = daily
        }
        #expect(try !Self.report(extra).contains("2026-10-25"))
    }

    @Test func currentRequired() throws {
        let data = try Self.payload { $0.removeValue(forKey: "current") }
        #expect(WeatherTool.parseForecastResponse(data) == nil)
    }

    @Test func descriptionAndSchema() throws {
        let description = WeatherTool.definition.function.description
        #expect(description.contains("forecast"))
        #expect(description.contains("not web_search"))
        #expect(description.contains("tomorrow"))
        #expect(description.contains("this weekend"))
        #expect(description.contains("ambiguous places are not guessed"))
        #expect(!description.contains("does not provide future forecasts"))
        #expect(WebSearchTool.definition.function.description.contains("Do not use it for current weather or weather forecasts up to 7 days"))
        let data = try JSONEncoder().encode(WeatherTool.definition)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let function = try #require(json["function"] as? [String: Any])
        let parameters = try #require(function["parameters"] as? [String: Any])
        let properties = try #require(parameters["properties"] as? [String: Any])
        let days = try #require(properties["days"] as? [String: Any])
        #expect(days["type"] as? String == "integer")
        #expect(days["minimum"] as? Int == 1)
        #expect(days["maximum"] as? Int == 7)
        #expect(parameters["required"] as? [String] == ["location"])
    }

    @Test func fetchFixture() async throws {
        let response = try await WeatherTool.fetchForecast(lat: 39.9, lon: 116.4, imperial: false, days: 7) { request in
            let url = try #require(request.url)
            #expect(url.absoluteString.contains("forecast_days=7"))
            #expect(request.timeoutInterval == 10)
            return (Data(Self.fixture.utf8), HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil, headerFields: nil)!)
        }
        #expect(response.current.temperature == 19.4)
        #expect(response.daily?.count == 7)
        #expect(response.timezone == "Asia/Shanghai")
    }

    @Test(arguments: [false, true])
    func fetchErrors(badPayload: Bool) async {
        await #expect(throws: NSError.self) {
            _ = try await WeatherTool.fetchForecast(lat: 0, lon: 0, imperial: false) { request in
                (Data("{}".utf8), HTTPURLResponse(url: request.url!, statusCode: badPayload ? 200 : 503, httpVersion: nil, headerFields: nil)!)
            }
        }
    }
}
