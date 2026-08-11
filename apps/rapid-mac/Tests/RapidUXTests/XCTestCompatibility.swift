import Testing

/// Small source-compatible assertion surface for the state-machine suite's
/// XCTest-era call sites. Keeping the existing messages avoids a noisy rewrite
/// while Swift Testing owns discovery, execution, and failure reporting.
func XCTAssertTrue(
    _ expression: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = ""
) {
    if !expression() { Issue.record(Comment(rawValue: message())) }
}

func XCTAssertFalse(
    _ expression: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = ""
) {
    if expression() { Issue.record(Comment(rawValue: message())) }
}

func XCTAssertFalse(
    _ expression: @autoclosure () -> Bool,
    _ message: @autoclosure () -> String = "",
    file _: StaticString,
    line _: UInt
) {
    if expression() { Issue.record(Comment(rawValue: message())) }
}

func XCTAssertEqual<T: Equatable>(
    _ lhs: @autoclosure () -> T,
    _ rhs: @autoclosure () -> T,
    _ message: @autoclosure () -> String = ""
) {
    if lhs() != rhs() { Issue.record(Comment(rawValue: message())) }
}

func XCTAssertNil<T>(
    _ expression: @autoclosure () -> T?,
    _ message: @autoclosure () -> String = ""
) {
    if expression() != nil { Issue.record(Comment(rawValue: message())) }
}

func XCTFail(_ message: @autoclosure () -> String = "") {
    Issue.record(Comment(rawValue: message()))
}
