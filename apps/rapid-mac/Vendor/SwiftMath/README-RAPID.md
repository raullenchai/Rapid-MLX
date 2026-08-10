# SwiftMath vendoring notes

This directory is SwiftMath 1.7.3 at
`fa8244ed032f4a1ade4cb0571bf87d2f1a9fd2d7`.

Rapid carries one bounded integration patch: `SwiftMathResources` resolves
`mathFonts.bundle` from `Bundle.main` before falling back to `Bundle.module`,
and the three upstream resource call sites use that resolver. This lets the
manually assembled, strictly signed macOS app load fonts from
`Contents/Resources`; upstream's generated accessor probes the app wrapper
root instead.

To audit an update, diff `Sources/SwiftMath` against the upstream tag. Expected
differences are the resource resolver/call-site substitutions plus the bounded
correctness fixes pinned by `SwiftMathVendorTests`: correct `\\varsigma`, keep
the CTFont cache inside its synchronization boundary, and honour the macOS
background-color setter.
