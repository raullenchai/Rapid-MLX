import Foundation
import Testing
@testable import Rapid

/// The fake server's repertoire, verbatim from `scripts/fake-rapid-mlx.sh`.
///
/// The golden flows drive the app against that script, and a title request
/// carries the user's original message — marker and all — so the fake picks
/// the same shape for our background call that it picked for the answer.
/// Whatever these parse to is what lands in `Tests/GUIGoldenFlows/__Snapshots__`,
/// so it is pinned here where a failure names the cause instead of showing up
/// as an unexplained baseline diff.
enum FakeServerReplies {
    static let `default` =
        "Hello from the fake rapid-mlx mock. I return deterministic content "
        + "so the smoke test has something to assert on."

    static let code = """
        Here is the function you asked for:

        ```python
        def fib(n):
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            return a
        ```

        It runs in O(n) time and constant space.
        """

    static let table = """
        | model | size | speed |
        | --- | --- | --- |
        | qwen3.5-9b | 5.2 GB | 74 tok/s |
        | llama-3.1-8b | 4.5 GB | 68 tok/s |

        Both fit comfortably in 16 GB.
        """

    static let math = """
        The Gaussian integral is

        $$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx = \\sqrt{\\pi}$$

        and inline it reads $e^{i\\pi} + 1 = 0$.
        """

    static let list = """
        Three things, in order:

        1. First, read the prompt.
        2. Second, plan the answer.
           - a nested point
        3. Third, write it down.
        """

    static let unicode =
        "中文排版测试:这是一段中文回答,用来检查换行和字宽。 Emoji: 🎯🚀。 Right-to-left: مرحبا."

    static let prose =
        "The lighthouse keeper kept two logbooks. One recorded the weather, "
        + "the ships, the hours of the lamp. The other recorded what he "
        + "thought about while he watched them."

    static let all: [(name: String, text: String)] = [
        ("default", `default`), ("code", code), ("table", table),
        ("math", math), ("list", list), ("unicode", unicode), ("prose", prose),
    ]
}

@Suite("Conversation title suggestion")
struct ConversationTitleSuggestionTests {

    // MARK: - Prompt

    @Test("The prompt carries both sides of the first exchange")
    func promptCarriesFirstExchange() {
        let transcript = [
            ChatMessage(role: .user, content: "How does Euler's theorem work?"),
            ChatMessage(role: .assistant, content: "For coprime a and n, a^φ(n) ≡ 1."),
        ]
        let built = ConversationTitleSuggestion.messages(forFirstExchange: transcript)
        #expect(built?.count == 2)
        #expect(built?.first?.role == .system)
        #expect(built?.last?.content.contains("Euler's theorem") == true)
        #expect(built?.last?.content.contains("a^φ(n)") == true)
    }

    @Test("Each side is capped so the call stays cheap")
    func excerptsAreCapped() {
        let transcript = [
            ChatMessage(role: .user, content: String(repeating: "x", count: 5_000)),
            ChatMessage(role: .assistant, content: String(repeating: "y", count: 5_000)),
        ]
        let prompt = ConversationTitleSuggestion.messages(forFirstExchange: transcript)?
            .last?.content ?? ""
        #expect(prompt.count < ConversationTitleSuggestion.excerptLimit * 2 + 200)
    }

    @Test("An exchange without an answer yet builds nothing")
    func incompleteExchangeBuildsNothing() {
        #expect(ConversationTitleSuggestion.messages(forFirstExchange: []) == nil)
        #expect(ConversationTitleSuggestion.messages(forFirstExchange: [
            ChatMessage(role: .user, content: "hi"),
        ]) == nil)
        // A placeholder assistant row with no prose is not an answer.
        #expect(ConversationTitleSuggestion.messages(forFirstExchange: [
            ChatMessage(role: .user, content: "hi"),
            ChatMessage(role: .assistant, content: "   "),
        ]) == nil)
    }

    @Test("A title retry skips failed and stopped attempts")
    func retryUsesFirstCompletedExchange() {
        let transcript = [
            ChatMessage(role: .user, content: "failed question"),
            ChatMessage(
                role: .assistant,
                content: "Couldn't start the model",
                status: .failed
            ),
            ChatMessage(role: .user, content: "stopped question"),
            ChatMessage(
                role: .assistant,
                content: "partial abandoned answer",
                errorMessage: "Stopped."
            ),
            ChatMessage(role: .user, content: "successful question"),
            ChatMessage(role: .assistant, content: "complete useful answer"),
        ]

        let prompt = ConversationTitleSuggestion.messages(forFirstExchange: transcript)?
            .last?.content ?? ""
        #expect(prompt.contains("successful question"))
        #expect(prompt.contains("complete useful answer"))
        #expect(!prompt.contains("failed question"))
        #expect(!prompt.contains("Couldn't start"))
        #expect(!prompt.contains("stopped question"))
        #expect(!prompt.contains("partial abandoned"))
    }

    // MARK: - Parsing what a small model actually returns

    @Test("Wrapped and labelled titles are unwrapped", arguments: [
        ("Euler's theorem and modular arithmetic", "Euler's theorem and modular arithmetic"),
        ("Title: Euler's theorem", "Euler's theorem"),
        ("title：Euler's theorem", "Euler's theorem"),
        ("\"Euler's theorem\"", "Euler's theorem"),
        ("“Euler's theorem”", "Euler's theorem"),
        ("**Euler's theorem**", "Euler's theorem"),
        ("「欧拉定理」", "欧拉定理"),
        ("1. Euler's theorem", "Euler's theorem"),
        ("- Euler's theorem", "Euler's theorem"),
        ("Euler's theorem.", "Euler's theorem"),
        ("欧拉定理。", "欧拉定理"),
        ("```\nEuler's theorem\n```", "Euler's theorem"),
        ("<think>hmm what to call it</think>\nEuler's theorem", "Euler's theorem"),
        ("Euler's theorem\nAlso known as Fermat's little theorem", "Euler's theorem"),
    ])
    func unwrapsTitles(_ raw: String, _ expected: String) {
        #expect(ConversationTitleSuggestion.parse(raw) == expected)
    }

    /// A question makes a fine title, so the trailing-stop strip must not
    /// take the question mark with it.
    @Test("A question mark survives")
    func questionMarkSurvives() {
        #expect(ConversationTitleSuggestion.parse("Why is the sky blue?") == "Why is the sky blue?")
    }

    @Test("Structure and prose are refused as titles", arguments: [
        // A table row is a row whatever else it is.
        "| model | size | speed |",
        // A lead-in to a list names nothing.
        "Three things, in order:",
        // More than one sentence is the answer, not a name for it.
        "中文排版测试:这是一段中文回答,用来检查换行和字宽。 Emoji: 🎯",
        "Modular arithmetic. It comes up in cryptography.",
    ])
    func refusesStructureAndProse(_ raw: String) {
        #expect(ConversationTitleSuggestion.parse(raw) == nil)
    }

    /// The multi-sentence rule is crude and takes abbreviations with it.
    /// Pinned so the loss is a decision on the record rather than a surprise:
    /// the fallback is the derived title, which is what the row shows today.
    @Test("An abbreviation is collateral, and the fallback is benign")
    func abbreviationIsCollateral() {
        #expect(ConversationTitleSuggestion.parse("Dr. Who and the daleks") == nil)
        // No space after the stop, so a version number is still a title.
        #expect(ConversationTitleSuggestion.parse("Migrating to Swift 6.2") == "Migrating to Swift 6.2")
    }

    @Test("Non-titles are refused", arguments: [
        "",
        "   ",
        "\n\n",
        "I can help with that! A good title would be Euler's theorem.",
        "Sure, here's a title for you",
        "Here is a title",
        "New chat",
        "x",
        // Prose. Truncating this to 42 characters reads worse than the
        // opening words of the user's own question, which is what the row
        // already shows.
        "This conversation is about the mathematical foundations of modular exponentiation",
    ])
    func refusesNonTitles(_ raw: String) {
        #expect(ConversationTitleSuggestion.parse(raw) == nil)
    }

    /// A generated title obeys the same one-line budget as a derived one,
    /// through the same function.
    @Test("A long but valid title is capped exactly like a derived one")
    func longTitleIsCapped() {
        let raw = "Euler's theorem, modular arithmetic and Fermat"
        let parsed = ConversationTitleSuggestion.parse(raw)
        #expect(parsed == ConversationStore.capped(raw))
        #expect(parsed?.count == 43)  // 42 + the ellipsis
    }
}

@Suite("Follow-up suggestion")
struct FollowUpSuggestionTests {

    @Test("The prompt carries the last exchange")
    func promptCarriesLastTurn() {
        let transcript = [
            ChatMessage(role: .user, content: "first"),
            ChatMessage(role: .assistant, content: "first answer"),
            ChatMessage(role: .user, content: "second"),
            ChatMessage(role: .assistant, content: "second answer"),
        ]
        let prompt = FollowUpSuggestion.messages(forTurn: transcript)?.last?.content ?? ""
        #expect(prompt.contains("second"))
        #expect(prompt.contains("second answer"))
        #expect(!prompt.contains("first answer"))
    }

    @Test("Clean output becomes three chips, in order")
    func cleanOutput() {
        let raw = """
            Can you give a concrete example?
            What is the proof?
            How does it relate to Fermat?
            """
        #expect(FollowUpSuggestion.parse(raw) == [
            "Can you give a concrete example?",
            "What is the proof?",
            "How does it relate to Fermat?",
        ])
    }

    @Test("Numbering, bullets and quotes are stripped")
    func decorationIsStripped() {
        let raw = """
            1. Can you give an example?
            - What is the proof?
            "How does it relate to Fermat?"
            """
        #expect(FollowUpSuggestion.parse(raw).count == 3)
        #expect(FollowUpSuggestion.parse(raw).first == "Can you give an example?")
    }

    /// The single rule that does most of the work: a preamble does not end in
    /// a question mark, so it disappears without a rule of its own.
    @Test("A preamble line is dropped without a rule for preambles")
    func preambleIsDropped() {
        let raw = """
            Here are three follow-up questions:
            Can you give an example?
            What is the proof?
            How does it relate to Fermat?
            """
        #expect(FollowUpSuggestion.parse(raw).count == 3)
        #expect(FollowUpSuggestion.parse(raw).first == "Can you give an example?")
    }

    @Test("Extra questions are trimmed to three")
    func extrasAreTrimmed() {
        let raw = "A?\nBB?\nCCC?\nDDDD?\nEEEEE?"
        #expect(FollowUpSuggestion.parse(raw) == ["A?", "BB?", "CCC?"])
    }

    @Test("Fewer than three is nothing at all", arguments: [
        "Can you give an example?\nWhat is the proof?",
        "Can you give an example?",
        "",
        "Here are some follow-up questions:",
        "I'm not sure what you would ask next.",
    ])
    func fewerThanThreeIsNothing(_ raw: String) {
        #expect(FollowUpSuggestion.parse(raw).isEmpty)
    }

    @Test("The question just asked is not offered back")
    func lastUserMessageIsExcluded() {
        let raw = "What is the proof?\nCan you give an example?\nHow does it relate?"
        let parsed = FollowUpSuggestion.parse(raw, excluding: "What is the proof?")
        // Two survivors is fewer than three, so nothing renders.
        #expect(parsed.isEmpty)
    }

    @Test("Duplicates are dropped case-insensitively")
    func duplicatesAreDropped() {
        let raw = "What is the proof?\nwhat is the PROOF?\nAn example?\nA third one?"
        #expect(FollowUpSuggestion.parse(raw) == [
            "What is the proof?", "An example?", "A third one?",
        ])
    }

    @Test("CJK questions with a full-width mark are accepted")
    func cjkQuestionsAccepted() {
        let raw = "能举个例子吗？\n证明过程是什么？\n和费马小定理什么关系？"
        #expect(FollowUpSuggestion.parse(raw).count == 3)
    }

    @Test("Structure that is not a question is refused", arguments: [
        "| model | size | speed |\n| --- | --- | --- |\n| a | b | c |",
        "```\ncode?\ncode?\ncode?\n```",
        "---\n***\n___",
    ])
    func structureIsRefused(_ raw: String) {
        #expect(FollowUpSuggestion.parse(raw).isEmpty)
    }

    @Test("A question too long to scan is refused")
    func overlongQuestionRefused() {
        let long = String(repeating: "word ", count: 30) + "?"
        let raw = "Short one?\n\(long)\nAnother short?"
        #expect(FollowUpSuggestion.parse(raw).isEmpty)
    }
}

/// What the golden-flow fake server's answers do to both parsers.
///
/// Every case here is a fact about `scripts/fake-rapid-mlx.sh`, not a wish.
/// A shape that parses to a title changes the sidebar row's `desc=` in
/// `Tests/GUIGoldenFlows/__Snapshots__/chat-*.txt`; a shape that parses to
/// three chips adds `AXButton` rows. Both are legitimate — the app's
/// behaviour did change — but they must be *known*, and regenerated
/// deliberately rather than discovered as a mystery diff.
@Suite("Background assist against the fake server")
struct BackgroundAssistFakeServerTests {

    @Test("Which fake shapes yield a title")
    func fakeShapeTitles() {
        var titled: [String: String] = [:]
        for (name, text) in FakeServerReplies.all {
            if let title = ConversationTitleSuggestion.parse(text) {
                titled[name] = title
            }
        }
        // Pinned so a parser change that alters golden-flow output fails
        // HERE, with a name, rather than as an unexplained baseline diff.
        #expect(titled == ["math": "The Gaussian integral is"])
    }

    @Test("No fake shape yields follow-up chips")
    func fakeShapesYieldNoChips() {
        for (name, text) in FakeServerReplies.all {
            #expect(FollowUpSuggestion.parse(text).isEmpty, "\(name) produced chips")
        }
    }
}

/// Wrong-language chips, and the instruction-echoing that shipped in the
/// first cut. Both were found by driving the bundled 1.2B model rather than
/// by reading the prompt.
@Suite("Background assist — measured against a small model")
struct SmallModelBehaviourTests {

    /// The reported defect: sending "hi" produced the title `3 to 6 words`.
    ///
    /// The old instruction was "Reply with ONLY the title: 3 to 6 words, no
    /// quotes…", and the model handed the constraint back as the answer. Two
    /// defences now: the instruction contains no noun phrase that could pass
    /// for a title, and the parser refuses words that name the container
    /// rather than the contents.
    @Test("Instruction echoes are refused as titles", arguments: [
        // The reported defect, verbatim.
        "3 to 6 words",
        // And the general form, so a reworded instruction cannot leak a new
        // literal past a list of old ones.
        "2 to 5 words", "5 to 8",
        "Chat conversations", "Chat", "Chat topic",
        "Conversation", "New chat", "Untitled",
    ])
    func instructionEchoesRefused(_ raw: String) {
        #expect(ConversationTitleSuggestion.parse(raw) == nil)
    }

    /// The instruction must not contain a phrase a model could mistake for
    /// the answer. This is the property that broke.
    @Test("The title instruction offers nothing to copy")
    func titleInstructionOffersNothingToCopy() {
        let prompt = ConversationTitleSuggestion.systemPrompt
        #expect(!prompt.contains("3 to 6"))
        #expect(!prompt.lowercased().contains("the title"))
    }

    /// Naming a greeting as a greeting is a real answer about a real
    /// exchange — a bigger model returns exactly that. Only words for the
    /// container are refused.
    @Test("A real but slight title is kept")
    func slightTitleIsKept() {
        #expect(ConversationTitleSuggestion.parse("Friendly greeting") == "Friendly greeting")
    }

    /// Measured: the language instruction was obeyed on 10 of 20 turns, and
    /// 12 of 20 with a `CRITICAL:` clause. Three English chips under a
    /// Chinese answer read as the app not having noticed — so a mismatch
    /// throws the set away.
    @Test("Chips in the wrong script are dropped")
    func wrongScriptIsDropped() {
        let english = "What is the proof?\nCan you give an example?\nHow is it used?"
        let chinese = "证明是什么？\n能举个例子吗？\n它怎么用？"

        #expect(FollowUpSuggestion.parse(english, excluding: "explain euler's theorem").count == 3)
        #expect(FollowUpSuggestion.parse(chinese, excluding: "解释一下欧拉定理").count == 3)
        // Crossed.
        #expect(FollowUpSuggestion.parse(english, excluding: "解释一下欧拉定理").isEmpty)
        #expect(FollowUpSuggestion.parse(chinese, excluding: "explain euler's theorem").isEmpty)
    }

    /// With nothing to compare against, the check must not veto.
    @Test("An empty reference cannot veto")
    func emptyReferenceDoesNotVeto() {
        let english = "What is the proof?\nCan you give an example?\nHow is it used?"
        #expect(FollowUpSuggestion.parse(english, excluding: "").count == 3)
        #expect(FollowUpSuggestion.parse(english).count == 3)
    }
}
