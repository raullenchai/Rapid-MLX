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
    /// `onPageComplete` fires after each page so a caller waiting on this work
    /// can tell "still running" from "stalled" — the wait is far too long to
    /// bound with a fixed timeout.
    static func recognizePages(
        of document: PDFDocument,
        range: Range<Int>,
        characterBudget: Int = .max,
        onPageComplete: (() -> Void)? = nil
    ) -> String {
        var pages: [String] = []
        var remaining = characterBudget
        for index in range {
            if Task.isCancelled { break }
            if remaining <= 0 { break }
            defer { onPageComplete?() }
            guard let page = document.page(at: index) else { continue }

            let existing = page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let text = existing.isEmpty ? recognize(page: page) : existing
            guard !text.isEmpty else { continue }
            let tagged = "[Page \(index + 1)]\n\(text)"
            // Charge the separator too, so the joined result cannot exceed the
            // budget by the number of pages.
            let cost = tagged.count + (pages.isEmpty ? 0 : 2)
            guard cost <= remaining else {
                pages.append(String(tagged.prefix(max(0, remaining - 2))))
                break
            }
            remaining -= cost
            pages.append(tagged)
        }
        return pages.joined(separator: "\n\n")
    }

    /// Recognize a single page. Returns "" when the page cannot be rendered or
    /// holds no legible text — a blank scan is a normal outcome, not an error.
    static func recognize(page: PDFPage) -> String {
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
        return (request.results ?? [])
            .compactMap { $0.topCandidates(1).first?.string }
            .joined(separator: "\n")
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
