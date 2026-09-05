import { resolve } from 'node:path';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type ProxyOptions } from 'vite';

/**
 * The dev proxy is not a convenience — it is the only configuration that
 * works against a real backend.
 *
 * `auth.origin_is_allowed` (rmlx_web/auth.py:149-185) admits a request
 * only when there is no `Origin` at all, or `Sec-Fetch-Site` is
 * `same-origin`/`none`, or the `Origin` authority equals the `Host`
 * authority. Pointing the dev page straight at 127.0.0.1:7788 satisfies none
 * of those: localhost:5173 -> 127.0.0.1:7788 is `cross-site`, and
 * localhost:5173 -> localhost:7788 is `same-site`. Both are refused, so every
 * request comes back 403 `origin_refused`. The CSP is not involved (Vite
 * serves no CSP in dev), which makes the failure read like a server bug.
 *
 * Proxying makes the page's fetches same-origin, so the browser sends
 * `Sec-Fetch-Site: same-origin` and the first branch admits them.
 */
const devProxy: ProxyOptions = {
  target: 'http://127.0.0.1:7788',
  /**
   * MUST stay false — this is the trap.
   *
   * `changeOrigin: true` rewrites `Host` to 127.0.0.1:7788 while `Origin`
   * stays localhost:5173. Chrome and Safari would still pass on the
   * `Sec-Fetch-Site` branch, but any client that omits that header (an older
   * browser, curl, a Playwright `request` context) falls through to the
   * authority comparison at auth.py:179-185 and is refused. The bug would
   * therefore appear only on some clients, or only after a browser update.
   */
  changeOrigin: false,
  ws: false,
  configure(proxy) {
    // Third safety net: with no `Origin` at all the server takes auth.py's
    // "not a browser, not the confused deputy this guard is about" branch
    // (auth.py:170-171) and admits unconditionally.
    proxy.on('proxyReq', (proxyReq) => {
      proxyReq.removeHeader('origin');
    });
  },
};

export default defineConfig({
  plugins: [tailwindcss(), react()],

  resolve: {
    alias: { '@': resolve(import.meta.dirname, 'src') },
  },

  // app.py serves the shell at GET / and mounts the build at /static.
  base: '/static/',

  build: {
    outDir: resolve(import.meta.dirname, '../rmlx_web/static'),

    // outDir is inside the Python package; Vite's default would delete it.
    // scripts/clean-assets.mjs clears assets/ instead.
    emptyOutDir: false,

    target: 'safari16',
    cssCodeSplit: false,
    minify: 'esbuild',
    sourcemap: false,
  },

  server: {
    port: 5173,
    proxy: {
      '/api': devProxy,
      '/v1': devProxy,
    },
  },
});
