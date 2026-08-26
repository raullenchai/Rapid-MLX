import CoreGraphics
import Foundation
import PDFKit
import Vision

/// Sequential, cancellable OCR for PDF pages without selectable text.
enum PDFTextRecognizer {
    static let renderScale: CGFloat = 1.5
    static let languages = ["zh-Hans", "zh-Hant", "en-US"]

    static func needsRecognition(_ page: PDFPage) -> Bool {
        (page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "").isEmpty
    }

    struct Extraction {
        let text: String
        /// False when the character budget or cancellation stopped the pass.
        let reachedEnd: Bool
    }

    /// Extracts page-tagged text without accumulating beyond `characterBudget`.
    /// `recognizeScans` controls whether empty text layers fall back to Vision.
    static func recognizePages(
        of document: PDFDocument,
        range: Range<Int>,
        characterBudget: Int = .max,
        recognizeScans: Bool = true,
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
            if bounded.clamped, bounded.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
            let existing = bounded.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let text = existing.isEmpty && recognizeScans
                ? recognize(page: page, characterBudget: remaining)
                : existing
            guard !text.isEmpty else { continue }
            let tagged = "[Page \(index + 1)]\n\(text)"
            let cost = tagged.count + (pages.isEmpty ? 0 : 2)
            guard cost <= remaining else {
                pages.append(String(tagged.prefix(max(0, remaining - 2))))
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
            remaining -= cost
            pages.append(tagged)
            if bounded.clamped {
                return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: false)
            }
        }
        return Extraction(text: pages.joined(separator: "\n\n"), reachedEnd: true)
    }

    /// Uses PDFKit selection so an oversized page is bounded before allocation.
    /// A failed bounded selection never falls back to the unbounded `page.string`.
    static func boundedText(of page: PDFPage, limit: Int) -> (text: String, clamped: Bool) {
        guard limit > 0 else { return ("", page.numberOfCharacters > 0) }
        let available = page.numberOfCharacters
        guard available > limit else { return (page.string ?? "", false) }
        guard let selection = page.selection(for: NSRange(location: 0, length: limit)),
              let text = selection.string else {
            return ("", true)
        }
        return (String(text.prefix(limit)), true)
    }

    /// Returns an empty string when rendering or recognition yields no text.
    static func recognize(page: PDFPage, characterBudget: Int = .max) -> String {
        guard let image = render(page) else { return "" }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = languages
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

    private static func render(_ page: PDFPage) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        let scaledWidth = bounds.width * renderScale
        let scaledHeight = bounds.height * renderScale
        let maxPixelCount = 64_000_000

        // Validate dimensions before CGFloat-to-Int conversion and allocation.
        guard scaledWidth.isFinite,
              scaledHeight.isFinite,
              scaledWidth >= 1,
              scaledHeight >= 1,
              scaledWidth <= CGFloat(maxPixelCount),
              scaledHeight <= CGFloat(maxPixelCount) else {
            return nil
        }
        let width = Int(scaledWidth)
        let height = Int(scaledHeight)
        let (pixelCount, overflow) = width.multipliedReportingOverflow(by: height)
        guard !overflow, pixelCount <= maxPixelCount else { return nil }

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
        context.translateBy(x: -bounds.minX, y: -bounds.minY)
        page.draw(with: .mediaBox, to: context)
        return context.makeImage()
    }
}
