import Foundation

/// The offscreen page a diagram is drawn in, and the rules that keep it from
/// reaching anything.
///
/// Kept in one small file on purpose: everything a security review has to
/// read is here, and a source-guard test can assert over it (there is no
/// `http` literal in this file, and a test enforces that).
enum MermaidHostPage {

    /// A scheme of our own. Not `file:` — a file URL would give the document
    /// access to the user's disk under some configurations, and not
    /// `loadHTMLString(_:baseURL: nil)` either, which lands the page on an
    /// `about:blank` origin from which the script tag below cannot load,
    /// forcing 3.4 MB of JavaScript to be inlined into a string on every load.
    static let scheme = "rapid-mermaid"

    /// The only two paths the handler will serve. Anything else fails.
    static let hostPagePath = "/host.html"
    static let libraryPath = "/mermaid.min.js"

    static var hostPageURL: URL { URL(string: "\(scheme)://local\(hostPagePath)")! }

    /// Block everything, then re-allow our own scheme.
    ///
    /// This is the layer that catches what the scheme handler cannot: the
    /// handler is only consulted for its own scheme, so an `https://`
    /// subresource would go straight past it to the network. Rule order
    /// matters — `ignore-previous-rules` has to come second.
    static let contentRuleListJSON = """
        [
          { "trigger": { "url-filter": ".*" },
            "action": { "type": "block" } },
          { "trigger": { "url-filter": "^\(scheme)://" },
            "action": { "type": "ignore-previous-rules" } }
        ]
        """

    /// The document. Inlined as a Swift string rather than bundled, because
    /// every bundled file costs a `cp` in `build.sh`, a stanza in
    /// `verify-app-resources.swift`, and a new way to be missing at runtime —
    /// for markup nobody edits without also editing the Swift beside it.
    ///
    /// The Content-Security-Policy is the earliest and cheapest of the four
    /// layers, and the only one that covers `connect-src` — fetch, XHR,
    /// WebSocket, EventSource and `sendBeacon`, which is where a payload
    /// smuggled into a diagram label would actually try to go.
    static var html: String {
        """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <meta http-equiv="Content-Security-Policy" content="
          default-src 'none';
          script-src 'unsafe-inline' 'unsafe-eval' \(scheme):;
          style-src 'unsafe-inline';
          img-src data:;
          connect-src 'none';
          font-src 'none';
          frame-src 'none';
          object-src 'none';
          base-uri 'none';
          form-action 'none'">
        <script src="\(scheme)://local\(libraryPath)"></script>
        <style>
          html, body { margin: 0; padding: 0; background: transparent; }
          #stage { display: inline-block; }
        </style>
        </head>
        <body>
        <div id="stage"></div>
        <script>
        "use strict";

        // `unsafe-eval` is Mermaid's requirement — its expression parser
        // builds functions at run time. It is confined to this document,
        // which has no network reach and no access to anything of the
        // reader's.

        window.__rapidReady =
            typeof mermaid === "object" && typeof mermaid.render === "function";

        /**
         * Draw one diagram into the page and report its size.
         *
         * The caller then snapshots that rectangle. Returning the SVG source
         * instead was tried first and abandoned: AppKit's SVG renderer drops
         * `<foreignObject>` labels entirely, and with `htmlLabels: false` it
         * mispositions the native `<text>` and never draws `marker` arrowheads.
         * WebKit is the only renderer that agrees with what Mermaid emits.
         */
        window.__rapidRender = async function (source, theme) {
          const stage = document.getElementById("stage");
          stage.innerHTML = "";
          try {
            mermaid.initialize({
              startOnLoad: false,
              theme: theme === "dark" ? "dark" : "default",
              // Mermaid's own sanitiser, above the CSP. `strict` strips HTML
              // from labels rather than trusting it.
              securityLevel: "strict",
              // A `%%{init: …}%%` directive inside the diagram can override
              // configuration. This is the list of keys it may not touch —
              // without it a model-authored diagram can turn `htmlLabels`
              // back on and inject raw markup into this page.
              secure: [
                "secure", "securityLevel", "startOnLoad",
                "maxTextSize", "suppressErrorRendering", "htmlLabels",
                "theme", "themeVariables",
              ],
              // Mermaid's own size guard, so its error fires before ours.
              maxTextSize: 50000,
              // Stops Mermaid injecting its own "Syntax error" bomb into the
              // DOM, which would otherwise be what got snapshotted.
              suppressErrorRendering: true,
              fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
            });
            const id = "d" + (window.__n = (window.__n || 0) + 1);
            const { svg } = await mermaid.render(id, source);
            stage.innerHTML = svg;
            const el = stage.querySelector("svg");
            if (!el) { return { ok: false, error: "no svg produced" }; }
            // Mermaid defaults to `width: 100%`, which would measure as the
            // window rather than the drawing.
            el.removeAttribute("width");
            el.removeAttribute("height");
            el.style.maxWidth = "none";
            const box = el.getBBox();
            if (box.width <= 0 || box.height <= 0) {
              return { ok: false, error: "empty drawing" };
            }
            // getBBox() omits stroke caps and marker arrowheads. Reserve a
            // small perimeter so edge strokes are not cropped by the exact
            // snapshot surface.
            const padding = 8;
            const width = Math.ceil(box.width + padding * 2);
            const height = Math.ceil(box.height + padding * 2);
            // The origin may be negative. Preserve the actual drawing bounds
            // in the viewport rather than folding x/y into its dimensions,
            // which can either clip or add asymmetric empty space.
            el.setAttribute(
              "viewBox",
              `${box.x - padding} ${box.y - padding} ${box.width + padding * 2} ${box.height + padding * 2}`
            );
            el.setAttribute("width", width);
            el.setAttribute("height", height);
            return { ok: true, width: width, height: height };
          } catch (e) {
            return { ok: false, error: String((e && e.message) || e) };
          }
        };
        </script>
        </body>
        </html>
        """
    }
}
