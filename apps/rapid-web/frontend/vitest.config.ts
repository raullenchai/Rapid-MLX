import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: { '@': resolve(import.meta.dirname, 'src') },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}', 'test/**/*.test.{ts,tsx}'],
    // e2e/ is Playwright's; Vitest must not try to run those specs.
    exclude: ['e2e/**', 'node_modules/**'],
    css: { modules: { classNameStrategy: 'non-scoped' } },
  },
});
