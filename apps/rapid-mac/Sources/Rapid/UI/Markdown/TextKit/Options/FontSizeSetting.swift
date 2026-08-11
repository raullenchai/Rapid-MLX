import CoreGraphics

/// Mirrors `OAIMarkdown.MarkdownOptions.FontSizeSetting` from ChatGPT Classic:
/// six discrete steps rather than a free-form point size.
///
/// The case names and the count are recovered fact. The point values are not —
/// they compile away — so the ramp below is anchored on the one step we did
/// measure (`size3` body text at 15pt, from CJK ink height ≈14.0) and spaced
/// at roughly 1.09× per step.
///
/// Why a setting object at all, rather than reading Dynamic Type per view:
/// ChatGPT threads `fontSizeSetting` explicitly from `MessagesViewController`
/// into each `MessageRow`. That matters for us too — `@ScaledMetric` re-reads
/// the environment on every body evaluation, which is the wrong cost model
/// inside collection-view cells that re-evaluate on scroll.
enum FontSizeSetting: Int, CaseIterable, Sendable {
    case size1 = 1
    case size2
    case size3
    case size4
    case size5
    case size6

    static let `default` = FontSizeSetting.size3

    /// Body point size for this step. `size3` = 15pt is measured; the rest
    /// are interpolated.
    var bodyPointSize: CGFloat {
        switch self {
        case .size1: 12
        case .size2: 13.5
        case .size3: 15
        case .size4: 16.5
        case .size5: 18
        case .size6: 20
        }
    }

    /// Monospaced size for code blocks. Measured at 13pt when body is 15pt,
    /// so the ratio is carried across the ramp rather than hard-coded.
    var codePointSize: CGFloat {
        bodyPointSize - 2
    }

    /// Line height for code. Measured Δ=16.0 at 13pt → ratio ≈1.23.
    var codeLineHeight: CGFloat {
        (codePointSize * 1.23).rounded()
    }
}
