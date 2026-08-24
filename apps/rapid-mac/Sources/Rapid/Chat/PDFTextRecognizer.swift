import CoreGraphics
import Foundation
import PDFKit
import Vision

/// Recognizes text on PDF pages that carry no selectable text — scanned books,
/// photographed documents, image-only exports.
///
/// ## Cost, and what it forces
///
/// Measured on a real 529-page scanned textbook: **~0.69 s per page** at
/// ``renderScale``, so the whole book is about six minutes. That single number
/// dictates the design everywhere this is used — OCR can never run while the
/// user waits, and a caller must be able to stop it.
///
/// Concurrency does NOT help. Recognizing 12 pages across a 10-core machine
/// took the same wall-clock time as doing them one at a time (1.0x), because
/// Vision already saturates the Neural Engine internally. A worker pool here
/// would add cancellation and ordering complexity for nothing, so recognition
/// is deliberately sequential.
///
/// ``renderScale`` is 1.5 because 2.0 measured identically on both quality and
/// time (0.67 vs 0.69 s/page, same recognized text) while allocating ~1.8x the
/// pixels. Rendering is not free either — 0.25 s/page of the total is
/// rasterization, not recognition.
enum PDFTextRecognizer {
    /// Points-to-pixels factor when rasterizing a page for recognition.
    static let renderScale: CGFloat = 1.5

    /// Recognition languages, in priority order. Vision needs to be told:
    /// left to default it recognizes English only and returns near-empty text
    /// for a Chinese scan.
    static let languages = ["zh-Hans", "zh-Hant", "en-US"]

    /// True when the page has no selectable text and is therefore an OCR
    /// candidate. Cheap — reads the existing text layer, never rasterizes.
    static func needsRecognition(_ page: PDFPage) -> Bool {
        (page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").isEmpty
    }

    /// Outcome of a bounded extraction pass.
    ///
    /// `reachedEnd` is what lets a caller distinguish "this is the whole
    /// document" from "this is as much as the budget allowed". Length alone
    /// cannot answer that — a run that stops one character short of the
    /// ceiling looks identical to one that consumed the last page — and
    /// guessing wrong marks a truncated extract complete, which is exactly
    /// the claim ``DocumentContentCache/Entry/isComplete`` exists to make
    /// honestly.
    struct Extraction {
        let text: String
        /// True when every page in the requested range was consumed in full.
        /// False when the character budget, or cancellation, stopped the run.
        let reachedEnd: Bool
    }

    /// Recognize `range`, returning page-tagged text in the same shape the
    /// selectable-text path produces so downstream code cannot tell them apart.
    ///
    /// Checks `Task.isCancelled` between pages: at ~0.69 s each this is the
    /// only place a multi-minute job can be stopped promptly.
    ///
    /// Pages that already carry selectable text are passed through as-is, so a
    /// document with scanned plates among typeset pages pays the OCR cost only
    /// for the plates.
    ///
    /// `characterBudget` bounds PEAK memory, not just the returned string. The
    /// caller's own ceiling is applied to the finished text, which is far too
    /// late: a 100 MB PDF can decompress to gigabytes of text, and collecting
    /// every page before truncating means holding all of it — plus the joined
    /// copy — at once. Stopping the accumulation instead means the process
    /// never holds more than the budget, whatever the source contains.
    ///
    /// The budget also bounds each page BEFORE its text becomes a Swift
    /// string, via ``boundedText(of:limit:)``. Reading `page.string` and then
    /// slicing is far too late for a single highly-compressed sheet that
    /// decompresses to more text than the whole budget.
    ///
    /// `onPageComplete` fires after each page so a caller waiting on this work
    /// can tell "still running" from "stalled" — the wait is far too long to
    /// bound with a fixed timeout.
    static func recognizePages(
        of document: PDFDocument,
        range: Range<Int>,
        characterBudget: Int = .max,
        onPageComplete: (() -> Void)? = nil
    ) -> Extraction {
        var pages: [String] = []
        var remaining = characterBudget
        for index in range {
            if Task.isCancelled { return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false) }
            if remaining <= 0 { return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false) }
            defer { onPageComplete?() }
            guard let page = document.page(at: index) else { continue }

            let bounded = boundedText(of: page, limit: remaining)
            // The budget cut this page short, so whatever follows on it — and
            // every page after — is lost. Recognizing it would also be wasted
            // work: there is nowhere to put the result.
            if bounded.clamped, bounded.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
            let existing = bounded.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let text = existing.isEmpty ? recognize(page: page, characterBudget: remaining) : existing
            guard !text.isEmpty else { continue }
            let tagged = "[Page \(index + 1)]\n\(text)"
            // Charge the separator too, so the joined result cannot exceed the
            // budget by the number of pages.
            let cost = tagged.count + (pages.isEmpty ? 0 : 2)
            guard cost <= remaining else {
                pages.append(String(tagged.prefix(max(0, remaining - 2))))
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
            remaining -= cost
            pages.append(tagged)
            // A page whose own text was clipped means the rest of the document
            // is unreachable even though the accumulator still has room.
            if bounded.clamped {
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
        }
        return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: true)
    }

    /// At most `limit` characters of a page's selectable text, without ever
    /// materializing more than that.
    ///
    /// ``PDFPage/string`` returns the ENTIRE text layer as one Swift string,
    /// so slicing it afterwards is a bound on the result, not on peak memory:
    /// a single highly-compressed sheet that decompresses to hundreds of
    /// megabytes of text is fully resident (and copied again by trimming and
    /// interpolation) before any ceiling applies. ``numberOfCharacters`` reads
    /// the layer's length without copying it, and ``selection(for:)`` copies
    /// only the requested range, so an oversized page costs `limit`
    /// characters instead of its own size.
    ///
    /// An oversized page whose selection cannot be produced yields NOTHING
    /// rather than falling back to the full read. "Malformed text layer" and
    /// "pathological size" are not mutually exclusive — a hostile PDF can
    /// present both at once — so a fallback that reaches for `page.string`
    /// hands that document exactly the unbounded allocation this function
    /// exists to prevent. Losing one page is the correct trade: the caller
    /// reports the extract as truncated, which is true and recoverable.
    ///
    /// - Returns: the bounded text, and whether the page had more to give.
    ///   Empty text with `clamped == true` means the page could not be read
    ///   within the budget at all.
    static func boundedText(of page: PDFPage, limit: Int) -> (text: String, clamped: Bool) {
        guard limit > 0 else { return ("", page.numberOfCharacters > 0) }
        let available = page.numberOfCharacters
        // Small enough to read whole: the only path that touches `page.string`,
        // and it is bounded by the check above.
        guard available > limit else { return (page.string ?? "", false) }
        guard let selection = page.selection(for: NSRange(location: 0, length: limit)),
              let text = selection.string else {
            return ("", true)
        }
        return (String(text.prefix(limit)), true)
    }

    /// Recognize a single page. Returns "" when the page cannot be rendered or
    /// holds no legible text — a blank scan is a normal outcome, not an error.
    ///
    /// `characterBudget` stops collecting observations rather than truncating
    /// the joined result, for the same reason ``recognizePages`` bounds its
    /// accumulation: a dense page can carry more recognized text than the
    /// caller's whole remaining budget.
    static func recognize(page: PDFPage, characterBudget: Int = .max) -> String {
        guard let image = render(page) else { return "" }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = languages
        // Language correction fixes the run-together words and confused
        // homoglyphs that scans produce. It costs little next to recognition.
        request.usesLanguageCorrection = true

        do {
            try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        } catch {
            return ""
        }
        var lines: [String] = []
        var remaining = characterBudget
        for observation in request.results ?? [] {
            guard remaining > 0 else { break }
            guard let line = observation.topCandidates(1).first?.string else { continue }
            let cost = line.count + (lines.isEmpty ? 0 : 1)
            guard cost <= remaining else {
                lines.append(String(line.prefix(max(0, remaining - 1))))
                break
            }
            remaining -= cost
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }

    /// Rasterize a page onto an opaque white background.
    ///
    /// The white fill matters: a PDF page has no background of its own, so
    /// drawing onto the zeroed buffer would put dark text on black and
    /// recognition would return nothing.
    private static func render(_ page: PDFPage) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        let width = Int(bounds.width * renderScale)
        let height = Int(bounds.height * renderScale)
        // A malformed page can report a degenerate or absurd box; CGContext
        // would either fail or try to allocate it.
        guard width > 0, height > 0, width * height <= 64_000_000 else { return nil }

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.scaleBy(x: renderScale, y: renderScale)
        // Pages whose media box does not start at the origin would otherwise
        // render off-canvas.
        context.translateBy(x: -bounds.minX, y: -bounds.minY)
        page.draw(with: .mediaBox, to: context)
        return context.makeImage()
    }
}
