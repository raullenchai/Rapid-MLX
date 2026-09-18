import Foundation
import Testing

@testable import Rapid

/// #3564 — every major server-side error must reach the GUI faithfully.
///
/// The engine deliberately SANITISES error messages (a generation-time
/// out-of-memory abort leaves the process as a bare ``"Internal server
/// error"``), and the chat UI deliberately renders only curated copy, never
/// raw server text. Bridging those two deliberate choices is a stable,
/// machine-readable ``error.code`` in the server's OpenAI-shaped envelope:
/// the engine stamps the category, and the GUI maps the code to a diagnosis
/// ``Kind`` — so an OOM surfaces the memory card instead of the generic
/// "Rapid couldn't finish that request" fallback.
///
/// These pin the GUI half of that contract: the code table, the envelope
/// decoder, and — the property that actually matters — that the CODE, not a
/// keyword scan of the (sanitised) message, decides the category.
@Suite("Faithful engine-error diagnosis")
struct FaithfulEngineErrorDiagnosisTests {

    // MARK: - code -> Kind table

    @Test("Known engine codes map to their diagnosis kind")
    func knownCodesMap() {
        #expect(FailureDiagnoser.kind(forEngineCode: "insufficient_memory") == .modelOutOfMemory)
        #expect(FailureDiagnoser.kind(forEngineCode: "model_out_of_memory") == .modelOutOfMemory)
        #expect(FailureDiagnoser.kind(forEngineCode: "model_load_failed") == .modelLoadFailed)
        // A transient abort's faithful recovery is a retry, so it maps to the
        // retryable failure kind — NOT the memory card (retrying an OOM would
        // just fail again identically).
        #expect(FailureDiagnoser.kind(forEngineCode: "engine_aborted") == .requestFailed)
    }

    @Test("Unknown or absent codes fall through to the heuristics (nil)")
    func unknownCodesFallThrough() {
        #expect(FailureDiagnoser.kind(forEngineCode: "some_future_code") == nil)
        #expect(FailureDiagnoser.kind(forEngineCode: "") == nil)
        #expect(FailureDiagnoser.kind(forEngineCode: nil) == nil)
    }

    // MARK: - envelope decoding

    @Test("The stable code is decoded from an OpenAI-shaped error body")
    func decodesCodeFromEnvelope() {
        let body = #"{"error":{"message":"The model ran out of memory during generation.","type":"server_error","code":"insufficient_memory","param":null}}"#
        #expect(FailureDiagnoser.engineErrorCode(fromBody: body) == "insufficient_memory")
    }

    @Test("Surrounding whitespace on the code is trimmed")
    func trimsCode() {
        let body = #"{"error":{"code":"  engine_aborted  "}}"#
        #expect(FailureDiagnoser.engineErrorCode(fromBody: body) == "engine_aborted")
    }

    @Test("Non-envelope, missing, null, or blank codes decode to nil")
    func nonEnvelopeBodiesDecodeToNil() {
        // The pre-fix sanitised 500 body — plain text, not the envelope.
        #expect(FailureDiagnoser.engineErrorCode(fromBody: "Internal server error") == nil)
        // A FastAPI-default {"detail": ...} wrapper has no error.code.
        #expect(FailureDiagnoser.engineErrorCode(fromBody: #"{"detail":"nope"}"#) == nil)
        // Envelope present but code explicitly null.
        #expect(FailureDiagnoser.engineErrorCode(fromBody: #"{"error":{"message":"x","code":null}}"#) == nil)
        // Envelope present but code blank once trimmed.
        #expect(FailureDiagnoser.engineErrorCode(fromBody: #"{"error":{"code":"   "}}"#) == nil)
        // Not JSON at all.
        #expect(FailureDiagnoser.engineErrorCode(fromBody: "") == nil)
    }

    // MARK: - the #3564 regression: code beats the sanitised message

    @Test("A 503 whose body carries code=insufficient_memory maps to the memory card even when the message has no memory keyword")
    func oomCodeBeatsKeywordScanOnHTTPStatus() {
        // The message deliberately contains NO memory keyword — only the code
        // identifies the category. Pre-fix (keyword scan only) this classified
        // as .requestFailed and the user saw the generic "couldn't finish"
        // card for an OOM.
        let body = #"{"error":{"message":"The request could not be completed.","type":"server_error","code":"insufficient_memory","param":null}}"#
        let kind = FailureDiagnoser.chatFailureKind(
            error: ChatStreamError.httpStatus(503, body)
        )
        #expect(kind == .modelOutOfMemory)
    }

    @Test("The same code classification applies on the transport-error path")
    func oomCodeClassifiesOnTransport() {
        let body = #"{"error":{"message":"The request could not be completed.","code":"insufficient_memory"}}"#
        #expect(
            FailureDiagnoser.chatFailureKind(error: ChatStreamError.transport(body))
                == .modelOutOfMemory
        )
    }

    @Test("A transient engine_aborted 503 maps to a retryable failure, not the memory card")
    func engineAbortedMapsToRequestFailed() {
        let body = #"{"error":{"message":"Inference was interrupted by a transient engine error. Please try again.","type":"server_error","code":"engine_aborted","param":null}}"#
        #expect(
            FailureDiagnoser.chatFailureKind(error: ChatStreamError.httpStatus(503, body))
                == .requestFailed
        )
    }

    @Test("A sanitised, code-less 5xx body still falls back to the keyword heuristics")
    func codelessBodyFallsBackToKeywords() {
        // Old engine / non-envelope body: no code, generic text -> retryable.
        #expect(
            FailureDiagnoser.chatFailureKind(error: ChatStreamError.httpStatus(500, "Internal server error"))
                == .requestFailed
        )
        // And a code-less body that DOES name memory still classifies as OOM
        // via the existing keyword path (backward compatibility).
        #expect(
            FailureDiagnoser.chatFailureKind(
                error: ChatStreamError.httpStatus(500, "the model needs more memory than your Mac has")
            ) == .modelOutOfMemory
        )
    }

    // MARK: - raw-string entry point

    @Test("chatFailureKind(raw:) prefers the structured code over the keyword scan")
    func rawStringPrefersCode() {
        let body = #"{"error":{"message":"The request could not be completed.","code":"insufficient_memory"}}"#
        #expect(FailureDiagnoser.chatFailureKind(raw: body) == .modelOutOfMemory)
    }

    @Test("chatFailureKind(raw:) still honours the keyword heuristics when there is no code")
    func rawStringKeywordFallback() {
        #expect(FailureDiagnoser.chatFailureKind(raw: "the local engine isn't running") == .engineNotRunning)
        #expect(FailureDiagnoser.chatFailureKind(raw: "insufficient memory to continue") == .modelOutOfMemory)
    }

    // MARK: - end-to-end: a mid-stream abort carries its code to the diagnoser

    @Test("A mid-stream OOM frame carries its code through ChatStreamClient and classifies via the code, not a keyword scan")
    @MainActor
    func midStreamOOMFrameClassifiesViaCodeEndToEnd() async throws {
        let client = ChatStreamClient(
            baseURL: URL(string: "fake://rapid-mlx")!,
            session: MidStreamOOMCodeProtocol.session()
        )
        let req = ChatStreamClient.Request(
            alias: "qwen3.5-4b",
            messages: [ChatMessage(role: .user, content: "hi", status: .complete)]
        )
        do {
            try await client.send(req) { _ in }
            Issue.record("expected a thrown ChatStreamError, got clean return")
        } catch let error as ChatStreamError {
            guard case .transport(let body) = error else {
                Issue.record("expected .transport, got \(error)")
                return
            }
            // The client carries the FULL envelope, so the code is present
            // even though the message was sanitised of any memory keyword.
            #expect(body.contains("insufficient_memory"))
            #expect(!body.lowercased().contains("out of memory"))
            // And the diagnoser reaches the memory card VIA the code — a
            // keyword scan of this message alone would miss it.
            #expect(FailureDiagnoser.chatFailureKind(error: error) == .modelOutOfMemory)
        } catch {
            Issue.record("expected ChatStreamError, got \(error)")
        }
    }
}

/// #3564 end-to-end stub: the server emits a mid-stream error frame whose
/// message has been SANITISED to remove any memory keyword, but whose stable
/// ``code`` still identifies the category. ChatStreamClient must carry the
/// full envelope through ``.transport`` so the diagnoser reads the code.
final class MidStreamOOMCodeProtocol: URLProtocol, @unchecked Sendable {
    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MidStreamOOMCodeProtocol.self] + (config.protocolClasses ?? [])
        return URLSession(configuration: config)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: 200,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        // The message carries NO memory keyword; only ``code`` names OOM.
        let body = """
        data: {"choices":[{"delta":{"content":"working"}}]}\n
        data: {"error":{"message":"The request could not be completed.","type":"server_error","code":"insufficient_memory","param":null}}\n
        """.data(using: .utf8)!
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
