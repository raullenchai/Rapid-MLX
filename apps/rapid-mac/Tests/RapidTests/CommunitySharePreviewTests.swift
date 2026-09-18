import Foundation
import Testing
@testable import Rapid

/// The native consent dialog must describe the submission it is consenting to.
///
/// Two defects, one cause. `preview_run --json` reports `withheld` — the facts
/// the CLI removes so the ingestion validator will accept the payload — and
/// `decodeSharePreview` dropped the key on the floor. Meanwhile the sheet's
/// SHARED column asserted a static "Model name and quantisation", which for
/// any run off a warm cache is simply untrue: the projection replaces the
/// measured quantization with `{kind: unknown}` and removes the resolved
/// revision before sending.
///
/// So the dialog promised to publish provenance it was actively withholding.
/// Both halves are now derived from the preview itself.
@Suite("Share preview consent")
struct CommunitySharePreviewTests {
    /// A preview captured from the real packaged CLI:
    /// `rapid-mlx benchmark share <run> --preview --json`, for a benchmark of
    /// `lfm2.5-1b-4bit` measured off a warm Hugging Face cache. Its
    /// `payload_json` is trimmed to the blocks the sheet reads; `withheld` is
    /// verbatim.
    private static func fixtureData() throws -> Data {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/community-share-preview.json")
        return try Data(contentsOf: url)
    }

    private static func preview() throws -> CommunityBenchmarkUploadPreview {
        try CommunityBenchmarkCommand.decodeSharePreview(
            try fixtureData(), runID: "e1390322-5f48-41cf-bd37-b24391953baf"
        )
    }

    // MARK: - Decoding

    @Test("The withheld facts survive decoding, with path, value and reason")
    func withheldIsDecoded() throws {
        let preview = try Self.preview()
        #expect(preview.withheld.count == 2)

        let byPath = Dictionary(uniqueKeysWithValues: preview.withheld.map { ($0.path, $0) })
        let revision = try #require(byPath["model.components[0].source.resolved_revision"])
        #expect(revision.value == "125e006d991147f3b432249d1bdf0821987f12b0")
        #expect(revision.reason.contains("repository id only"))
        #expect(revision.fieldName == "resolved_revision")

        let quantization = try #require(byPath["model.components[0].quantization"])
        // A nested object is rendered as its pairs, so the user can recognise
        // what is being held back rather than seeing "{…}".
        #expect(quantization.value.contains("kind: weights"))
        #expect(quantization.value.contains("weight_bits_x2: 8"))
        #expect(quantization.value.contains("method: affine"))
        #expect(quantization.reason.contains("unknown quantization block"))
    }

    @Test("The identity decoded is the one on the wire, not the archive's")
    func publishedIdentityIsTheProjectedOne() throws {
        let identity = try #require(try Self.preview().publishedIdentity)
        #expect(identity.repoID == "mlx-community/LFM2.5-1.2B-Instruct-4bit")
        // The archive holds a revision and 4-bit affine facts; the submission
        // holds neither, and this is the object the disclosure describes.
        #expect(identity.resolvedRevision == nil)
        #expect(identity.quantization.isUnknown)
        #expect(identity.quantization.displayName == nil)
    }

    @Test("A preview from an older CLI decodes with nothing withheld")
    func olderPreviewWithoutTheKey() throws {
        // That CLI does not project, so there is nothing to disclose — an
        // absent key is not an error.
        let json = #"""
        {"target":"https://rapidmlx.com/api/benchmarks/atomic",
         "install_id":"c76f82c46d52","payload_digest":"sha256:a","body_digest":"sha256:b",
         "payload_json":"{\"model\":{\"repo_id\":\"mlx-community/X\"}}"}
        """#
        let preview = try CommunityBenchmarkCommand.decodeSharePreview(
            Data(json.utf8), runID: "run-1"
        )
        #expect(preview.withheld.isEmpty)
        #expect(preview.publishedIdentity?.repoID == "mlx-community/X")
    }

    @Test("A malformed withheld entry is skipped, not fatal")
    func malformedWithheldEntries() {
        let facts = CommunityBenchmarkCommand.decodeWithheld([
            ["path": "a.b", "reason": "because", "value": "v"],
            ["path": "no.reason", "value": "v"],
            ["reason": "no path", "value": "v"],
            "not an object",
        ] as [Any])
        #expect(facts.map(\.path) == ["a.b"])
    }

    @Test("Withheld values of every JSON shape render readably")
    func valueRendering() {
        #expect(CommunityBenchmarkCommand.describeWithheldValue("abc") == "abc")
        #expect(CommunityBenchmarkCommand.describeWithheldValue(64) == "64")
        #expect(CommunityBenchmarkCommand.describeWithheldValue(NSNull()) == "—")
        #expect(CommunityBenchmarkCommand.describeWithheldValue(nil) == "—")
        #expect(
            CommunityBenchmarkCommand.describeWithheldValue(
                ["kind": "weights", "bits": 4] as [String: Any]
            ) == "bits: 4, kind: weights"
        )
    }

    // MARK: - The disclosure the user reads

    @Test("The confirmation sheet never grows past the screen")
    func sheetHeightFitsTheDisplay() {
        // A laptop display: the sheet shrinks so its footer stays visible.
        #expect(CommunityBenchmarkShareConfirmationSheet.sheetHeight(availableHeight: 780) == 660)
        // A large display: capped at the design height.
        #expect(CommunityBenchmarkShareConfirmationSheet.sheetHeight(availableHeight: 1400) == 720)
        // Keeps a usable minimum while the display allows it...
        #expect(CommunityBenchmarkShareConfirmationSheet.sheetHeight(availableHeight: 500) == 420)
        // ...and never exceeds the display when it does not.
        #expect(CommunityBenchmarkShareConfirmationSheet.sheetHeight(availableHeight: 300) == 300)
        for available in stride(from: CGFloat(200), through: 1600, by: 100) {
            #expect(CommunityBenchmarkShareConfirmationSheet.sheetHeight(availableHeight: available) <= available)
        }
    }

    @Test("SHARED never claims quantisation the submission does not carry")
    func sharedDoesNotOverclaim() throws {
        let items = CommunityBenchmarkShareConfirmationSheet.sharedItems(
            for: try Self.preview()
        )
        let joined = items.joined(separator: "\n").lowercased()
        // The exact false claim this replaces.
        #expect(!joined.contains("model name and quantisation"))
        #expect(!joined.contains("quantisation"))
        #expect(!joined.contains("checkpoint revision"))
        // What it does publish, named concretely.
        #expect(
            items.contains {
                $0.contains("mlx-community/LFM2.5-1.2B-Instruct-4bit")
            }
        )
        #expect(items.contains { $0.contains("Mac model") })
        #expect(items.contains { $0.contains("timings") })
    }

    @Test("SHARED does name quantisation when the payload really carries it")
    func sharedClaimsWhatIsPresent() throws {
        // A hypothetical widened projection: the claim follows the payload,
        // so it becomes true again on its own rather than needing an edit.
        var preview = try Self.preview()
        preview.publishedIdentity = CommunityModelIdentity(
            repoID: "mlx-community/LFM2.5-1.2B-Instruct-4bit",
            resolvedRevision: "125e006d991147f3b432249d1bdf0821987f12b0",
            quantization: .init(
                kind: "weights", baseDType: "bfloat16", method: "affine",
                weightBitsX2: 8, groupSize: 64
            )
        )
        let items = CommunityBenchmarkShareConfirmationSheet.sharedItems(for: preview)
        #expect(items.contains { $0.contains("4-bit affine") })
        #expect(items.contains { $0.contains("checkpoint revision") })
    }

    @Test("A preview with no decodable identity still lists the repository id")
    func fallbackWhenIdentityIsUnreadable() throws {
        var preview = try Self.preview()
        preview.publishedIdentity = nil
        let items = CommunityBenchmarkShareConfirmationSheet.sharedItems(for: preview)
        #expect(items.contains { $0.contains("repository id") })
        #expect(!items.joined().lowercased().contains("quantisation"))
    }

    @Test("Every withheld fact reaches the sheet with its reason intact")
    func disclosureCoversEveryFact() throws {
        let preview = try Self.preview()
        // The section renders one row per fact; nothing is summarised away,
        // because "2 details withheld" would not let anyone judge the trade.
        #expect(preview.withheld.allSatisfy { !$0.value.isEmpty })
        #expect(preview.withheld.allSatisfy { !$0.reason.isEmpty })
        #expect(Set(preview.withheld.map(\.id)).count == preview.withheld.count)
    }

    @Test("The disclosure and the payload agree about the revision")
    func disclosureMatchesThePayload() throws {
        let preview = try Self.preview()
        let withheldRevision = preview.withheld.first {
            $0.path.hasSuffix("resolved_revision")
        }
        // Withheld in the disclosure, absent from the wire: the two halves
        // describe the same submission.
        #expect(withheldRevision != nil)
        #expect(preview.publishedIdentity?.resolvedRevision == nil)
        #expect(!preview.payloadJSON.contains("resolved_revision"))
    }
}

/// A refusal is a decision, not a malfunction.
///
/// The CLI declines to publish a result measured by a build whose tree
/// differed from its commit, because the wire contract can only name a commit.
/// Desktop has to tell that apart from a crash — the raw failure document
/// carries a `refused` flag for exactly that, and the flag has to survive onto
/// the error, because by the time a caller holds the message the document is
/// gone.
@Suite("Publication refusal")
struct CommunityPublicationRefusalTests {
    @Test("The refused flag is read from the document, not the wording")
    func refusalIsDetectedFromTheFlag() {
        let refused = #"""
        {"error":"this result was measured by a build whose working tree differed from its commit (60be71853…), so publishing it would attribute these numbers to code that did not produce them.","refused":true,"saved":false}
        """#
        #expect(CommunityBenchmarkCommand.isRefusal(refused))
        // The sentence alone is not the signal.
        #expect(!CommunityBenchmarkCommand.isRefusal(#"{"error":"disk full","saved":false}"#))
        #expect(!CommunityBenchmarkCommand.isRefusal("Traceback (most recent call last):"))
    }

    @Test("A traceback preceding the document does not hide the flag")
    func refusalSurvivesLeadingNoise() {
        let detail = """
        warning: something
        {"error":"refused","refused":true,"saved":false}
        """
        #expect(CommunityBenchmarkCommand.isRefusal(detail))
    }

    @Test("The failure carries the flag so the message can stand alone")
    func failureCarriesTheFlag() {
        let refusal = CommunityBenchmarkCommand.Failure(
            message: "…differed from its commit…", isRefusal: true
        )
        #expect(refusal.isRefusal)
        #expect(refusal.errorDescription == "…differed from its commit…")
        // An ordinary failure is not a refusal, and defaults accordingly.
        #expect(!CommunityBenchmarkCommand.Failure(message: "boom").isRefusal)
    }

    @Test("The extracted sentence is what the user reads")
    func summaryExtractsTheSentence() {
        let document = #"""
        {"error":"this result was measured by a build whose working tree differed from its commit","refused":true,"saved":false}
        """#
        let summary = CommunityBenchmarkCommand.failureSummary(from: document)
        #expect(summary == "this result was measured by a build whose working tree differed from its commit")
        // No JSON leaks into the sentence the user sees.
        #expect(!summary.contains("refused"))
        #expect(!summary.contains("{"))
    }
}
