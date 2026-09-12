import Foundation
import Testing
@testable import Rapid

/// Issue #308 (2026-06-20) regression pin for the
/// "tool-not-called" caption introduced as fix (c) of the Quickstart
/// happy-path bug.
///
/// Symptom: the original Quickstart "Speed" pick
/// (``gemma3-1b-qat-4bit``) is too small to reliably emit
/// ``tool_calls`` for arithmetic prompts. With the calculator tool
/// enabled and the Quickstart's own suggested calculator prompt
/// ("What is 15% of 2650 plus the square root of 781?"), the model
/// answered ``43.92504669599178`` (wrong by an order of magnitude;
/// real answer ≈ 425.45) with NO ``tool_calls`` chip and no fallback
/// warning — the user reads a confidently-wrong number as the
/// answer.
///
/// Fix: ``ChatViewModel.runOneStream`` sets
/// ``ChatMessage.toolNotCalledFlagged = true`` when every gate in
/// ``ChatMessage.shouldFlagToolNotCalled`` holds (tools advertised
/// + zero tool_calls + raw-answer-looking prose + tool-shaped user
/// prompt). ``MessageRow`` paints a dismissible caption above the
/// bubble so the user is alerted before reading the (possibly
/// hallucinated) answer.
///
/// This file covers (i) the gate truth-table on
/// ``shouldFlagToolNotCalled`` directly; (ii) the two heuristic
/// helpers (``contentLooksLikeRawAnswer`` and
/// ``promptLooksCalculatorish``); (iii) the canary copy +
/// VoiceOver label snapshots so an accidental reword fails the
/// suite; (iv) the back-compat surface — old on-disk sessions
/// saved before this release decode with ``toolNotCalledFlagged``
/// defaulted false; (v) the ``ChatViewModel.lastUserPromptBefore``
/// walker the wiring depends on.
@Suite("Issue #308 tool-not-called caption — gates, heuristics, copy, back-compat")
struct ToolNotCalledCaptionTests {

    // MARK: - Gate truth-table on shouldFlagToolNotCalled

    /// Canonical issue #308 repro: tools advertised, zero
    /// ``tool_calls`` emitted, raw-numeric content, calculator-shaped
    /// user prompt → caption FIRES.
    @Test("Issue #308 repro: calculator prompt + raw numeric answer + tools-but-no-tool_calls → FLAGS")
    func canonicalReproFires() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650 plus the square root of 781?",
            assistantContent: "43.92504669599178",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true
        )
        #expect(flag,
                "The canonical issue #308 repro must trip the caption — otherwise the user reads the hallucinated number unguarded.")
    }

    /// Gate 1: ``toolsRequested == false`` → caption HIDDEN. Without
    /// this gate a short numeric answer to "what is 17*23?" with no
    /// tools enabled would still wear the caption, which would be a
    /// lie — the user never opted into tool-calling for the turn.
    @Test("Gate 1: tools NOT requested → caption HIDDEN")
    func gateOneToolsNotRequested() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650?",
            assistantContent: "397.5",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: false
        )
        #expect(!flag, "No tools advertised → no tool-not-called caption.")
    }

    /// Gate 2: ``toolCalls`` non-empty → caption HIDDEN. A real tool-
    /// call landed; the chip row already speaks for it. A redundant
    /// caption would lie about the model's behaviour.
    @Test("Gate 2: tool_calls were emitted → caption HIDDEN")
    func gateTwoToolCallsEmitted() {
        let call = ToolCall(
            id: "call_1",
            name: "calculator",
            arguments: "{\"expression\": \"0.15*2650 + sqrt(781)\"}"
        )
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650 plus the square root of 781?",
            assistantContent: "",
            toolCalls: [call],
            finishReason: "tool_calls",
            toolsRequested: true
        )
        #expect(!flag, "A real tool-call turn must not wear the warning caption.")
    }

    /// Gate 3: ``finish_reason == "tool_calls"`` with an empty
    /// ``toolCalls`` array → caption HIDDEN. Corner case where the
    /// stream signalled a tool-call terminal but the capture array
    /// was empty (rapid-mlx stream corruption / cancellation race).
    /// Conservative: stay silent rather than over-fire.
    @Test("Gate 3: finish_reason == tool_calls + empty array → caption HIDDEN (conservative)")
    func gateThreeFinishReasonToolCallsEvenEmpty() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650?",
            assistantContent: "397.5",
            toolCalls: [],
            finishReason: "tool_calls",
            toolsRequested: true
        )
        #expect(!flag, "Stream terminal of tool_calls — stay silent regardless of array shape.")
    }

    /// Gate 4: prose body is long and prose-shaped (no numeric
    /// dominance) → caption HIDDEN. A well-prosed answer to a
    /// tool-shaped prompt is not the silent-wrong-answer failure
    /// mode we're guarding against; the model might be intentionally
    /// answering from training data.
    @Test("Gate 4: long, prose-shaped reply (not numeric-dominated) → caption HIDDEN")
    func gateFourLongProseReply() {
        let prose = String(
            repeating: "The result is roughly four hundred and twenty-five point four five. ",
            count: 5
        )
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650 plus the square root of 781?",
            assistantContent: prose,
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true
        )
        #expect(!flag, "A long well-prosed reply is not the issue #308 failure shape — leave it alone.")
    }

    /// Gate 5: user prompt is NOT calculator-/search-/weather-shaped
    /// → caption HIDDEN. Without this gate every short assistant
    /// reply to a casual question ("yes" / "no" / "thanks") would
    /// wear the caption.
    @Test("Gate 5: user prompt is casual / non-tool-shaped → caption HIDDEN")
    func gateFiveCasualPrompt() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "hello there",
            assistantContent: "Hi! How can I help?",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true
        )
        #expect(!flag, "Casual greeting prompt must not wear the caption.")
    }

    /// Variant: length-truncated raw-numeric answer to a calculator
    /// prompt → caption STILL FIRES. A half-finished numeric answer
    /// is just as suspect as a complete one.
    @Test("Variant: finish_reason == length on a raw-numeric calculator reply → FLAGS")
    func lengthTruncationStillFires() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "Calculate 15% of 2650 plus sqrt(781)",
            assistantContent: "425.4536",
            toolCalls: nil,
            finishReason: "length",
            toolsRequested: true
        )
        #expect(flag, "Length-truncated raw numeric answer is still suspect — caption must fire.")
    }

    /// Variant: nil finishReason (mid-stream snapshot) → caption
    /// gate still evaluated on the other axes. Important because
    /// ``runOneStream`` may not have captured a finish reason if
    /// the stream terminated without a clean ``[DONE]`` chunk.
    @Test("Variant: nil finishReason still evaluated on other gates → can FLAG")
    func nilFinishReasonStillEvaluated() {
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is 15% of 2650?",
            assistantContent: "397.5",
            toolCalls: nil,
            finishReason: nil,
            toolsRequested: true
        )
        #expect(flag, "nil finishReason must not silently exempt the row from the gate.")
    }

    // MARK: - contentLooksLikeRawAnswer heuristic

    @Test("contentLooksLikeRawAnswer: bare numeric blob → TRUE")
    func contentRawNumeric() {
        #expect(ChatMessage.contentLooksLikeRawAnswer("43.92504669599178"))
        #expect(ChatMessage.contentLooksLikeRawAnswer("= 425.45"))
        #expect(ChatMessage.contentLooksLikeRawAnswer("397.5 + 27.95 = 425.45"))
    }

    @Test("contentLooksLikeRawAnswer: short prose WITHOUT digits → FALSE (codex r1 MAJOR fix)")
    func contentShortProseWithoutDigits() {
        // Codex r1 MAJOR-1 (#308 PR) tightening: the original draft
        // flagged every short reply (≤ 80 chars) — which would have
        // fired on "Paris." / "Yes." / "It depends." against a
        // tools-enabled session. After the tighten, a short prose
        // reply with no digits is NOT raw-answer-shaped; the gate
        // requires at least one digit.
        #expect(!ChatMessage.contentLooksLikeRawAnswer("About four hundred something."))
        #expect(!ChatMessage.contentLooksLikeRawAnswer("Paris."))
        #expect(!ChatMessage.contentLooksLikeRawAnswer("It depends."))
    }

    @Test("contentLooksLikeRawAnswer: short reply WITH digit → TRUE")
    func contentShortReplyWithDigit() {
        #expect(ChatMessage.contentLooksLikeRawAnswer("397.5 + 27.95 = 425.45"))
        #expect(ChatMessage.contentLooksLikeRawAnswer("The answer is 42."))
    }

    @Test("contentLooksLikeRawAnswer: long well-prosed reply (low digit ratio) → FALSE")
    func contentLongProse() {
        let prose = String(
            repeating: "The model would have called the calculator in a well-tuned setup. ",
            count: 4
        )
        #expect(!ChatMessage.contentLooksLikeRawAnswer(prose))
    }

    @Test("contentLooksLikeRawAnswer: empty / whitespace-only → FALSE")
    func contentEmpty() {
        #expect(!ChatMessage.contentLooksLikeRawAnswer(""))
        #expect(!ChatMessage.contentLooksLikeRawAnswer("    \n\t  "))
    }

    // MARK: - promptLooksCalculatorish heuristic

    @Test("promptLooksCalculatorish: math operator + digit → TRUE")
    func promptMathOperator() {
        #expect(ChatMessage.promptLooksCalculatorish("compute 17 * 23"))
        #expect(ChatMessage.promptLooksCalculatorish("what is 0.15 * 2650 + sqrt(781)"))
        #expect(ChatMessage.promptLooksCalculatorish("solve x^2 = 49"))
    }

    @Test("promptLooksCalculatorish: math keyword → TRUE")
    func promptMathKeyword() {
        #expect(ChatMessage.promptLooksCalculatorish("Calculate the square root of 781"))
        #expect(ChatMessage.promptLooksCalculatorish("What is 15 percent of 2650?"))
        #expect(ChatMessage.promptLooksCalculatorish("Multiply 17 by 23"))
    }

    @Test("promptLooksCalculatorish: web-search keyword → TRUE")
    func promptWebSearch() {
        #expect(ChatMessage.promptLooksCalculatorish("search for the latest news"))
        #expect(ChatMessage.promptLooksCalculatorish("what is the current weather in Paris"))
        #expect(ChatMessage.promptLooksCalculatorish("look up the population of Tokyo"))
    }

    @Test("Codex r1 MAJOR-1 (#308): evergreen factual prompt is NOT calculator-shaped")
    func promptEvergreenFactualNotFlagged() {
        // The whole point of the codex tightening: "what is the
        // capital of France?" → "Paris." is a normal exchange and
        // must NOT trip the caption. Pre-fix the bare "what is the"
        // keyword (combined with the short-prose content gate)
        // would have flagged it.
        #expect(!ChatMessage.promptLooksCalculatorish("What is the capital of France?"))
        #expect(!ChatMessage.promptLooksCalculatorish("Who wrote Hamlet?"))
        #expect(!ChatMessage.promptLooksCalculatorish("What is the meaning of irony?"))
        // End-to-end: tools enabled, evergreen prompt, short prose
        // reply with no digits → caption STAYS HIDDEN.
        let stillSilent = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is the capital of France?",
            assistantContent: "Paris.",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true
        )
        #expect(!stillSilent,
                "Tools-enabled evergreen factual Q with short prose A must not trip the caption — would be a high false-positive UX regression (codex r1 MAJOR-1).")
    }

    @Test("promptLooksCalculatorish: weather keyword → TRUE")
    func promptWeather() {
        #expect(ChatMessage.promptLooksCalculatorish("Will the temperature drop tomorrow?"))
        #expect(ChatMessage.promptLooksCalculatorish("Show me the forecast"))
    }

    @Test("0.14.1 dogfood: keywords match whole words, not substrings")
    func promptKeywordsAreWholeWords() {
        // Every one of these is ordinary prose that the shipped
        // `lowered.contains(kw)` classified as calculator-shaped:
        // "computer"/"computed"/"computing" ⊃ compute, "sometimes" ⊃
        // times, "surplus" ⊃ plus, "minuscule" ⊃ minus, "forecasting" ⊃
        // forecast, "temperatures" is the plural of a keyword. Paired
        // with the short-answer-with-a-digit content gate, any of them
        // put the "didn't call a tool" caution under a perfectly good
        // grounded answer.
        #expect(!ChatMessage.promptLooksCalculatorish(
            "Summarize what the computer vision section says"))
        #expect(!ChatMessage.promptLooksCalculatorish(
            "How was this computed in the report"))
        #expect(!ChatMessage.promptLooksCalculatorish(
            "Does it sometimes fail on startup"))
        #expect(!ChatMessage.promptLooksCalculatorish(
            "Explain the trade surplus argument"))
        #expect(!ChatMessage.promptLooksCalculatorish(
            "Is the risk minuscule or material"))
        #expect(!ChatMessage.promptLooksCalculatorish(
            "Who owns the forecasting process"))
        // …while the whole words themselves still match.
        #expect(ChatMessage.promptLooksCalculatorish("compute the total"))
        #expect(ChatMessage.promptLooksCalculatorish("3 times 4 equals what"))
        #expect(ChatMessage.promptLooksCalculatorish("what is the forecast"))
        // Punctuation and line ends are boundaries too.
        #expect(ChatMessage.promptLooksCalculatorish("(compute) the total"))
        #expect(ChatMessage.promptLooksCalculatorish("weather?"))
        #expect(ChatMessage.promptLooksCalculatorish("forecast"))
    }

    @Test("containsKeyword: boundaries, phrases, overlaps")
    func containsKeywordContract() {
        #expect(ChatMessage.containsKeyword("plus", in: "2 plus 2"))
        #expect(!ChatMessage.containsKeyword("plus", in: "surplus"))
        #expect(!ChatMessage.containsKeyword("plus", in: "plusher"))
        // Multi-word phrases only need their outer edges checked.
        #expect(ChatMessage.containsKeyword("look up", in: "please look up tokyo"))
        #expect(!ChatMessage.containsKeyword("look up", in: "overlook upstream"))
        // A later whole-word hit is still found after an embedded miss.
        #expect(ChatMessage.containsKeyword("sum of", in: "consumer sum of costs"))
        #expect(!ChatMessage.containsKeyword("", in: "anything"))
    }

    @Test("0.14.1 dogfood: a document-grounded answer wears no caution")
    func attachmentGroundedAnswerIsNotFlagged() throws {
        // The dogfood shape: a scanned invoice attached, a numeric
        // question, and a short correct answer read straight off the
        // page. Nothing on the tool roster could have done better, so
        // "answered without calling any of the available tools" is not a
        // caution — and firing it here spends the trust that makes the
        // caption worth showing when a model really does invent a total.
        let flaggedWithoutTheGate = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is the total due? Calculate it from the invoice.",
            assistantContent: "$1,204.55",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true,
            promptHadAttachment: false
        )
        #expect(flaggedWithoutTheGate,
                "Sanity check: without the attachment the shape IS the #308 failure mode and must still fire.")

        let flaggedWithTheGate = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is the total due? Calculate it from the invoice.",
            assistantContent: "$1,204.55",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true,
            promptHadAttachment: true
        )
        #expect(!flaggedWithTheGate)
    }

    @Test("An attachment does NOT exempt a live-data question")
    func attachmentDoesNotExemptLiveDataQuestions() {
        // codex, reviewing this PR: a blanket attachment exemption lets the
        // exact failure mode straight through. Attach a portfolio PDF, ask
        // for today's price, and no page on earth holds the answer — so a
        // bare number with no tool call is a guess, and the caption is right.
        let liveDataPrompts = [
            "Here is my portfolio. What is today's stock price for it?",
            "What's the current price of the first item on this invoice?",
            "Given this itinerary, what is the weather in Lisbon?",
            "Summarise this and add the latest news about the company.",
            "What's the exchange rate to convert the total on this invoice?",
        ]
        for prompt in liveDataPrompts {
            #expect(ChatMessage.shouldFlagToolNotCalled(
                userPrompt: prompt,
                assistantContent: "$1,204.55",
                toolCalls: nil,
                finishReason: "stop",
                toolsRequested: true,
                promptHadAttachment: true
            ), "A live-data question is not answerable from an attachment: \(prompt)")
        }
    }

    @Test("An attachment does NOT exempt self-contained arithmetic")
    func attachmentDoesNotExemptSelfContainedArithmetic() {
        // codex, round 2: "Attached is my resume; calculate 17*23" carries a
        // document that has nothing to do with the question. The numbers are
        // in the prompt, the calculator is the tool that should have run, and
        // a bare wrong product is the #308 failure mode.
        for prompt in ["Attached is my resume; calculate 17*23",
                       "Here's the invoice. Also, what is 1200 * 0.15?",
                       "Using this deck, confirm 1200 - 180 for me",
                       "this pdf, plus: 88 / 7 = ?"] {
            #expect(ChatMessage.shouldFlagToolNotCalled(
                userPrompt: prompt,
                assistantContent: "391",
                toolCalls: nil,
                finishReason: "stop",
                toolsRequested: true,
                promptHadAttachment: true
            ), "Self-contained arithmetic is not grounded by an attachment: \(prompt)")
        }
        // The dogfood case is the other side of the line and must stay
        // exempt: the operands live on the page, so reading them off it is
        // the grounded, correct answer.
        #expect(!ChatMessage.shouldFlagToolNotCalled(
            userPrompt: "What is the total due? Calculate it from the invoice.",
            assistantContent: "$1,204.55",
            toolCalls: nil,
            finishReason: "stop",
            toolsRequested: true,
            promptHadAttachment: true
        ))
    }

    @Test("promptContainsSelfContainedArithmetic: brought its own numbers?")
    func selfContainedArithmeticContract() {
        for prompt in ["17*23", "what is 1200 * 0.15", "88 / 7", "1200-180",
                       // Spaced around the minus, which is how most people
                       // actually type it (codex round 3).
                       "1200 - 180", "what is 1200 -  180?",
                       "5 + 5 = ?", "2^10", "give me 40% of 250"] {
            #expect(ChatMessage.promptContainsSelfContainedArithmetic(prompt),
                    "should be self-contained: \(prompt)")
        }
        for prompt in ["calculate the total from the invoice",
                       "sum of the line items", "what is the square root of it",
                       // A hyphen is not a minus sign unless it sits between
                       // two digits: a filename or a dashed aside is not math.
                       "see q3-report.pdf - what does it say?",
                       "the 2026 budget - summarise it",
                       "no numbers here at all"] {
            #expect(!ChatMessage.promptContainsSelfContainedArithmetic(prompt),
                    "should not be self-contained: \(prompt)")
        }
    }

    @Test("An attachment does NOT exempt an external-retrieval prompt")
    func attachmentDoesNotExemptExternalRetrieval() {
        // codex, round 4: "search for Ada Lovelace's biography" with a resume
        // attached asks to leave the document entirely, so the page grounds
        // nothing and a short confident answer is ungrounded.
        for prompt in ["Attached is my resume. Search for Ada Lovelace's biography.",
                       "Here's the invoice — look up the vendor's rating online",
                       "google for the current spec, see attached"] {
            #expect(ChatMessage.shouldFlagToolNotCalled(
                userPrompt: prompt,
                assistantContent: "1815.",
                toolCalls: nil,
                finishReason: "stop",
                toolsRequested: true,
                promptHadAttachment: true
            ), "External retrieval is not grounded by an attachment: \(prompt)")
        }
    }

    @Test("promptIsAttachmentAnswerable: only the math-keyword lane")
    func attachmentAnswerableContract() {
        // The ONE exempt shape: math vocabulary whose operands are on the page.
        for prompt in ["what is the total due? calculate it from the invoice",
                       "sum of the line items", "what percent of the total is tax",
                       "compute the subtotal"] {
            #expect(ChatMessage.promptIsAttachmentAnswerable(prompt),
                    "should be answerable from the page: \(prompt)")
        }
        // The three lanes it excludes, one case each.
        #expect(!ChatMessage.promptIsAttachmentAnswerable("calculate 17*23"))
        #expect(!ChatMessage.promptIsAttachmentAnswerable("calculate today's stock price"))
        #expect(!ChatMessage.promptIsAttachmentAnswerable("calculate it, then search for the source"))
        // And a prompt with no math vocabulary at all is not exempted either —
        // Gate 5 would reject it anyway, but the predicate must not claim it.
        #expect(!ChatMessage.promptIsAttachmentAnswerable("what does this say?"))
    }

    @Test("promptAsksForLiveData draws the line at a moving target")
    func liveDataContract() {
        // Names a moving target: no attached document can hold the answer.
        for prompt in ["what is today's close?", "latest news about the merger",
                       "the forecast for tomorrow", "current weather in Oslo",
                       "google for the spec", "what's the temperature outside",
                       "stock price now", "right now, how many users?"] {
            #expect(ChatMessage.promptAsksForLiveData(prompt), "should be live: \(prompt)")
        }
        // Not live-data (they name no moving target) — but note these no
        // longer keep Gate 1c's exemption either: as of round 4 they are
        // external-retrieval language, which withholds it. See
        // ``promptAsksForExternalRetrieval``.
        for prompt in ["look up the invoice number", "search for the clause about refunds",
                       "what is the total due?", "calculate 15 percent of the subtotal",
                       "sum of the line items"] {
            #expect(!ChatMessage.promptAsksForLiveData(prompt), "should not be live: \(prompt)")
        }
        // codex round 2: moving to whole-word matching must not drop ordinary
        // English plurals, which the old substring match caught.
        for prompt in ["what are the temperatures today", "the forecasts for the week",
                       "current prices on these", "latest versions of the spec",
                       "exchange rates right now", "stock prices"] {
            #expect(ChatMessage.promptAsksForLiveData(prompt), "plural should be live: \(prompt)")
        }
        // Whole-word matching still holds here: "concurrent" is not "current",
        // a "forecasting model" question is not a weather question, and an
        // "-ed" inflection is not an "-s" one.
        #expect(!ChatMessage.promptAsksForLiveData("explain concurrent map access"))
        #expect(!ChatMessage.promptAsksForLiveData("what is a temperate climate"))
        #expect(!ChatMessage.promptAsksForLiveData("explain forecasting models"))
        #expect(!ChatMessage.promptAsksForLiveData("the weathered stone wall"))
        // But the live list must still reach promptLooksCalculatorish, so the
        // two cannot drift apart.
        #expect(ChatMessage.promptLooksCalculatorish("what is the forecast"))
        #expect(ChatMessage.promptLooksCalculatorish("google for the spec"))
    }

    @Test("The attachment gate reads the wire history, and every user row in it")
    func attachmentGateReadsTheWireHistory() throws {
        let attachment = try ChatFileAttachment(
            filename: "invoice.pdf",
            kind: .pdf,
            extractedText: "TOTAL DUE 1,204.55",
            sourceByteCount: 18
        )
        let newestRowCarriesIt: [ChatMessage] = [
            ChatMessage(role: .user, content: "earlier question"),
            ChatMessage(role: .assistant, content: "earlier answer"),
            ChatMessage(role: .user, content: "total?", fileAttachments: [attachment]),
        ]
        #expect(ChatViewModel.historyCarriesAttachmentGrounding(newestRowCarriesIt))

        // An EARLIER turn counts, which reverses what an earlier version of
        // this test asserted. codex was right that the premise was wrong:
        // ``ChatMessage/modelContent`` re-sends each attachment's extracted
        // text on every follow-up, so "and 15 percent of that?" two turns on
        // is still answered from a document in front of the model — and
        // captioning it as a guess is the trust-spending false positive this
        // gate exists to prevent.
        let documentIsTwoTurnsBack: [ChatMessage] = [
            ChatMessage(role: .user, content: "total?", fileAttachments: [attachment]),
            ChatMessage(role: .assistant, content: "$1,204.55"),
            ChatMessage(role: .user, content: "and 15 percent of that?"),
        ]
        #expect(ChatViewModel.historyCarriesAttachmentGrounding(documentIsTwoTurnsBack))

        // But it is the WIRE history, so a document the context trim dropped
        // no longer grounds anything — the caption must come back rather than
        // stay silent at the exact moment the evidence is gone.
        let trimmedAway = Array(documentIsTwoTurnsBack.dropFirst(2))
        #expect(!ChatViewModel.historyCarriesAttachmentGrounding(trimmedAway))

        // An attachment on an ASSISTANT row is not user-supplied grounding,
        // and an empty history exempts nothing.
        #expect(!ChatViewModel.historyCarriesAttachmentGrounding([
            ChatMessage(role: .assistant, content: "here", fileAttachments: [attachment])
        ]))
        #expect(!ChatViewModel.historyCarriesAttachmentGrounding([]))
    }

    @Test("promptLooksCalculatorish: casual prompt → FALSE")
    func promptCasual() {
        #expect(!ChatMessage.promptLooksCalculatorish("hello there"))
        #expect(!ChatMessage.promptLooksCalculatorish("tell me a joke"))
        #expect(!ChatMessage.promptLooksCalculatorish("how are you doing"))
    }

    @Test("promptLooksCalculatorish: empty / whitespace → FALSE")
    func promptEmpty() {
        #expect(!ChatMessage.promptLooksCalculatorish(""))
        #expect(!ChatMessage.promptLooksCalculatorish("    \n  "))
    }

    // MARK: - Pinned copy + accessibility label snapshots

    /// Snapshot the visible caption copy. Any future reword has to
    /// land here as an intentional update — otherwise the suite
    /// fails and the author has to acknowledge the UX change.
    @Test("Visible caption copy snapshot")
    func captionCopySnapshot() {
        #expect(
            ChatMessage.toolNotCalledCaptionCopy ==
            "This model didn't call a tool — verify the answer."
        )
    }

    /// Snapshot the VoiceOver accessibility label. Same intent as
    /// the visible copy snapshot: a screen-reader regression is a
    /// real accessibility regression.
    @Test("VoiceOver accessibility label snapshot")
    func captionAccessibilityLabelSnapshot() {
        #expect(
            ChatMessage.toolNotCalledCaptionAccessibilityLabel ==
            "Caution: this model answered without calling any of the available tools. The answer may be a guess. Verify before relying on it."
        )
    }

    /// Repo-hygiene rule (MEMORY.md): all UI copy is English-only.
    /// Cheapest faithful check is "no CJK codepoints"; mirrors the
    /// same guard ToolUseCapability's tooltip test uses.
    @Test("Caption copy + accessibility label are English-only")
    func captionCopyEnglishOnly() {
        let cjkRanges: [ClosedRange<Unicode.Scalar>] = [
            Unicode.Scalar(0x3040)!...Unicode.Scalar(0x309F)!,
            Unicode.Scalar(0x30A0)!...Unicode.Scalar(0x30FF)!,
            Unicode.Scalar(0x4E00)!...Unicode.Scalar(0x9FFF)!,
        ]
        for text in [
            ChatMessage.toolNotCalledCaptionCopy,
            ChatMessage.toolNotCalledCaptionAccessibilityLabel
        ] {
            for scalar in text.unicodeScalars {
                for range in cjkRanges {
                    #expect(!range.contains(scalar),
                            "Tool-not-called copy contains CJK codepoint U+\(String(scalar.value, radix: 16, uppercase: true)); repo rule is English-only UI copy.")
                }
            }
        }
    }

    // MARK: - ChatMessage back-compat

    /// Old on-disk sessions saved before this release have no
    /// ``toolNotCalledFlagged`` key in their JSON envelope. The
    /// custom ``init(from:)`` shim must default the field to
    /// ``false`` so those sessions decode cleanly.
    @Test("Old session envelopes decode with toolNotCalledFlagged = false (back-compat)")
    func backCompatDecode() throws {
        // Synthesised legacy envelope (mirrors the pre-#308 codec
        // output — every field of ChatMessage except the new
        // toolNotCalledFlagged key).
        let legacyJSON = """
        {
          "id": "11111111-1111-1111-1111-111111111111",
          "role": "assistant",
          "content": "Legacy reply",
          "reasoning": "",
          "status": "complete",
          "reasoningTruncated": false,
          "contentTruncated": false,
          "createdAt": 778204800.0
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ChatMessage.self, from: legacyJSON)
        #expect(decoded.toolNotCalledFlagged == false,
                "Pre-#308 session envelope must decode with toolNotCalledFlagged = false; the missing-key fallback is the back-compat contract.")
    }

    /// Round-trip: a flagged message survives encode → decode.
    @Test("toolNotCalledFlagged round-trips through Codable")
    func flagRoundTrips() throws {
        let original = ChatMessage(
            role: .assistant,
            content: "43.92504669599178",
            status: .complete,
            toolNotCalledFlagged: true
        )
        let encoder = JSONEncoder()
        let data = try encoder.encode(original)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        #expect(decoded.toolNotCalledFlagged == true,
                "Flag must survive Codable round-trip — otherwise an exported / re-imported session would silently drop the warning.")
    }

    // MARK: - ChatViewModel.lastUserPromptBefore walker

    @MainActor
    @Test("lastUserPromptBefore: returns the most recent user prompt strictly before the index")
    func lastUserPromptBeforeBasic() {
        let user1 = ChatMessage(role: .user, content: "first user prompt")
        let asst1 = ChatMessage(role: .assistant, content: "first reply", status: .complete)
        let user2 = ChatMessage(role: .user, content: "second user prompt")
        let placeholder = ChatMessage(role: .assistant, content: "", status: .streaming)
        let messages = [user1, asst1, user2, placeholder]
        let prompt = ChatViewModel.lastUserPromptBefore(
            messages: messages,
            placeholderIndex: 3
        )
        #expect(prompt == "second user prompt")
    }

    @MainActor
    @Test("lastUserPromptBefore: tool-call loop with intermediate tool/assistant rows still finds the user prompt")
    func lastUserPromptBeforeWalksPastToolLoop() {
        let user = ChatMessage(role: .user, content: "What is 15% of 2650?")
        let asst1 = ChatMessage(role: .assistant, content: "", status: .complete)
        let tool1 = ChatMessage(role: .tool, content: "397.5", toolCallID: "call_1")
        let placeholder = ChatMessage(role: .assistant, content: "", status: .streaming)
        let messages = [user, asst1, tool1, placeholder]
        let prompt = ChatViewModel.lastUserPromptBefore(
            messages: messages,
            placeholderIndex: 3
        )
        #expect(prompt == "What is 15% of 2650?")
    }

    @MainActor
    @Test("lastUserPromptBefore: no user message before index → empty string")
    func lastUserPromptBeforeNoUser() {
        let sys = ChatMessage(role: .system, content: "system prompt")
        let placeholder = ChatMessage(role: .assistant, content: "", status: .streaming)
        let prompt = ChatViewModel.lastUserPromptBefore(
            messages: [sys, placeholder],
            placeholderIndex: 1
        )
        #expect(prompt == "")
    }

    @MainActor
    @Test("lastUserPromptBefore: index 0 / out-of-bounds → empty string (defensive)")
    func lastUserPromptBeforeBounds() {
        let user = ChatMessage(role: .user, content: "anything")
        #expect(ChatViewModel.lastUserPromptBefore(messages: [user], placeholderIndex: 0) == "")
        // out-of-bounds clamps to messages.count — still finds the user
        #expect(ChatViewModel.lastUserPromptBefore(messages: [user], placeholderIndex: 99) == "anything")
        // empty array
        #expect(ChatViewModel.lastUserPromptBefore(messages: [], placeholderIndex: 5) == "")
    }

    // MARK: - runOneStream wiring shape (codex r1 MINOR-2 closure)

    /// Codex r1 MINOR-2 (#308 PR): exercise the exact composition
    /// that ``ChatViewModel.runOneStream`` runs at terminal —
    /// ``lastUserPromptBefore`` + ``shouldFlagToolNotCalled`` — over
    /// the three shapes the production wiring needs to keep
    /// distinct. The actual ``runOneStream`` body is async + needs
    /// a live HTTP transport so we can't drive it end-to-end in a
    /// pure unit test, but the public composition IS the wiring;
    /// any future refactor that drops one of the helpers will fail
    /// this case.
    @MainActor
    @Test("runOneStream-shape integration: tools-enabled + zero tool_calls + calculator prompt → toolNotCalledFlagged TRUE")
    func runOneStreamShapeFlagsCanonicalRepro() {
        // Simulated session at the moment runOneStream's terminal
        // fires: user asked a calculator question, placeholder
        // assistant turn is about to be flagged.
        let user = ChatMessage(role: .user, content: "What is 15% of 2650 plus the square root of 781?")
        let placeholder = ChatMessage(role: .assistant, content: "43.92504669599178", status: .complete)
        let messages = [user, placeholder]
        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 1)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: placeholder.content,
            toolCalls: placeholder.toolCalls,
            finishReason: "stop",
            toolsRequested: true  // request.tools non-empty
        )
        #expect(flag, "End-to-end repro composition must flag — would silently regress the wiring.")
    }

    @MainActor
    @Test("runOneStream-shape integration: tools NOT enabled → toolNotCalledFlagged FALSE")
    func runOneStreamShapeRespectsToolsRequested() {
        // Same content/prompt, but request body had no tools array
        // — caption MUST NOT fire (user didn't opt in to tools).
        let user = ChatMessage(role: .user, content: "What is 15% of 2650?")
        let placeholder = ChatMessage(role: .assistant, content: "397.5", status: .complete)
        let messages = [user, placeholder]
        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 1)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: placeholder.content,
            toolCalls: placeholder.toolCalls,
            finishReason: "stop",
            toolsRequested: false  // request.tools nil/empty
        )
        #expect(!flag, "Without an advertised tool, the model has no responsibility to call one — caption stays silent.")
    }

    @MainActor
    @Test("runOneStream-shape integration: finish_reason == tool_calls → toolNotCalledFlagged FALSE")
    func runOneStreamShapeRespectsToolCallTerminal() {
        // A real tool-call turn arriving at terminal. Even with
        // tools advertised, if the stream ends in tool_calls the
        // chip row owns the bubble; the caption stays silent.
        let user = ChatMessage(role: .user, content: "Compute sqrt(781)")
        let call = ToolCall(id: "call_1", name: "calculator", arguments: "{}")
        let placeholder = ChatMessage(role: .assistant, content: "", status: .complete, toolCalls: [call])
        let messages = [user, placeholder]
        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 1)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: placeholder.content,
            toolCalls: placeholder.toolCalls,
            finishReason: "tool_calls",
            toolsRequested: true
        )
        #expect(!flag, "A real tool-call terminal must not also wear the warning.")
    }

    @MainActor
    @Test("runOneStream-shape integration: tool-loop intermediate rows still walks back to user prompt")
    func runOneStreamShapeWalksPastToolLoopAtTerminal() {
        // The real wiring has to walk back across an asst (tool-
        // call dispatch) + tool result before it finds the user
        // prompt. ``lastUserPromptBefore`` is the load-bearing
        // helper that makes this work; the integration MUST flag
        // the canonical repro even after a tool-loop round.
        let user = ChatMessage(role: .user, content: "What is 15% of 2650 plus the square root of 781?")
        let toolDispatchAsst = ChatMessage(role: .assistant, content: "", status: .complete)
        let toolResult = ChatMessage(role: .tool, content: "{\"error\": \"calculator timed out\"}", toolCallID: "call_1")
        let finalAsst = ChatMessage(role: .assistant, content: "43.92504669599178", status: .complete)
        let messages = [user, toolDispatchAsst, toolResult, finalAsst]
        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 3)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: finalAsst.content,
            toolCalls: finalAsst.toolCalls,
            finishReason: "stop",
            toolsRequested: true
        )
        #expect(flag, "After a tool-loop round, a raw-numeric reply still must trip the warning — the user is back at the same failure shape.")
    }

    @MainActor
    @Test("Successful multi-step tool loop suppresses the warning (no contradiction with the chip)")
    func successfulToolLoopSuppressesWarning() {
        // Real multi-step turn: the model calls calculator, gets a GOOD
        // result back (a ``.tool`` message with a non-failed status), then
        // writes a short numeric summary. The final assistant message has
        // no ``toolCalls`` of its own, so absent the ``toolSucceededThisTurn``
        // gate this raw-numeric-looking summary would false-positive — but
        // the visible tool-call chip already says the tool ran, so the
        // caption must stay silent.
        let user = ChatMessage(role: .user, content: "What is 15% of 2650 plus the square root of 781?")
        let call = ToolCall(id: "call_1", name: "calculator", arguments: "{}")
        let toolDispatchAsst = ChatMessage(role: .assistant, content: "", status: .complete, toolCalls: [call])
        let toolResult = ChatMessage(role: .tool, content: "425.42504669599178", status: .complete, toolCallID: "call_1")
        let finalAsst = ChatMessage(role: .assistant, content: "425.43", status: .complete)
        let messages = [user, toolDispatchAsst, toolResult, finalAsst]

        let succeeded = ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 3)
        #expect(succeeded, "A non-failed .tool result in this turn is a successful tool use.")

        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 3)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: finalAsst.content,
            toolCalls: finalAsst.toolCalls,
            finishReason: "stop",
            toolsRequested: true,
            toolSucceededThisTurn: succeeded
        )
        #expect(!flag, "A summary after a SUCCESSFUL tool call must not be captioned 'didn't call a tool' — it contradicts the tool-call chip.")
    }

    @MainActor
    @Test("Failed tool loop still warns — hallucinated raw answer after a tool error is the #308 shape")
    func failedToolLoopStillWarns() {
        // Production encodes a tool error as a ``.tool`` message with
        // status ``.failed`` (see ChatViewModel's result mapping). The
        // model then hallucinates a raw number. That is exactly the
        // failure we must keep flagging, so ``turnHadSuccessfulTool``
        // must return false for a failed result.
        let user = ChatMessage(role: .user, content: "What is 15% of 2650 plus the square root of 781?")
        let call = ToolCall(id: "call_1", name: "calculator", arguments: "{}")
        let toolDispatchAsst = ChatMessage(role: .assistant, content: "", status: .complete, toolCalls: [call])
        let toolResult = ChatMessage(
            role: .tool,
            content: "{\"error\": \"calculator timed out\"}",
            status: .failed,
            toolCallID: "call_1"
        )
        let finalAsst = ChatMessage(role: .assistant, content: "43.92504669599178", status: .complete)
        let messages = [user, toolDispatchAsst, toolResult, finalAsst]

        let succeeded = ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 3)
        #expect(!succeeded, "A FAILED .tool result must not count as a successful tool use.")

        let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 3)
        let flag = ChatMessage.shouldFlagToolNotCalled(
            userPrompt: prompt,
            assistantContent: finalAsst.content,
            toolCalls: finalAsst.toolCalls,
            finishReason: "stop",
            toolsRequested: true,
            toolSucceededThisTurn: succeeded
        )
        #expect(flag, "After a FAILED tool round, a raw-numeric reply still must trip the warning.")
    }

    @MainActor
    @Test("turnHadSuccessfulTool stops at the turn boundary — a tool used in an earlier turn never counts")
    func turnHadSuccessfulToolStopsAtTurnBoundary() {
        // Earlier turn used a tool successfully, then a fresh user prompt
        // starts a NEW turn where the model answers raw with no tool. The
        // walker must not leak the earlier turn's tool across the user
        // boundary.
        let user1 = ChatMessage(role: .user, content: "Compute sqrt(781)")
        let call = ToolCall(id: "call_1", name: "calculator", arguments: "{}")
        let asst1 = ChatMessage(role: .assistant, content: "", status: .complete, toolCalls: [call])
        let tool1 = ChatMessage(role: .tool, content: "27.94", status: .complete, toolCallID: "call_1")
        let asst1Summary = ChatMessage(role: .assistant, content: "27.94", status: .complete)
        let user2 = ChatMessage(role: .user, content: "What is 15% of 2650?")
        let asst2 = ChatMessage(role: .assistant, content: "397.5", status: .complete)
        let messages = [user1, asst1, tool1, asst1Summary, user2, asst2]

        #expect(
            !ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 5),
            "A tool from the PRIOR turn must not count for the current turn."
        )
        #expect(
            ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 3),
            "Within the first turn, the successful tool result must count."
        )
    }

    @MainActor
    @Test("turnHadSuccessfulTool requires .complete — a .streaming or .unknown tool row is NOT definitive success")
    func turnHadSuccessfulToolRequiresComplete() {
        // ``runToolLoop`` only ever writes .complete or .failed, but a
        // restored / edited / forward-compat envelope could carry a
        // .streaming (mid-flight) or .unknown (unrecognised) status on a
        // .tool row. Neither proves the tool produced a real result, so
        // the gate must NOT suppress the warning — suppressing it is the
        // unsafe direction (it would hide a raw-numeric hallucination).
        let user = ChatMessage(role: .user, content: "What is 15% of 2650 plus the square root of 781?")
        let call = ToolCall(id: "call_1", name: "calculator", arguments: "{}")
        let dispatch = ChatMessage(role: .assistant, content: "", status: .complete, toolCalls: [call])
        let finalAsst = ChatMessage(role: .assistant, content: "43.92504669599178", status: .complete)

        for badStatus in [ChatMessage.Status.streaming, .unknown] {
            let toolResult = ChatMessage(role: .tool, content: "425.42", status: badStatus, toolCallID: "call_1")
            let messages = [user, dispatch, toolResult, finalAsst]

            #expect(
                !ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 3),
                "A .\(badStatus) tool row must not count as definitive success."
            )
            let prompt = ChatViewModel.lastUserPromptBefore(messages: messages, placeholderIndex: 3)
            let flag = ChatMessage.shouldFlagToolNotCalled(
                userPrompt: prompt,
                assistantContent: finalAsst.content,
                toolCalls: finalAsst.toolCalls,
                finishReason: "stop",
                toolsRequested: true,
                toolSucceededThisTurn: ChatViewModel.turnHadSuccessfulTool(messages: messages, placeholderIndex: 3)
            )
            #expect(flag, "With a non-.complete tool row, the raw-numeric reply must still trip the warning.")
        }
    }
}
