import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  // The stub binds an ephemeral port per test, so files can run in parallel.
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'list' : 'html',

  use: {
    trace: 'on-first-retry',
  },

  projects: [
    {
      name: 'mobile-safari',
      // The target audience is overwhelmingly a phone behind a tunnel, so the
      // default project is a phone. WebKit also exercises the MathML path that
      // Chromium would not represent faithfully.
      use: { ...devices['iPhone 14'] },
    },
  ],
});
