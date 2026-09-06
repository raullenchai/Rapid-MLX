import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: ['../rmlx_web/**', 'playwright-report/**', 'test-results/**', 'dist/**'],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  {
    files: ['src/**/*.{ts,tsx}', 'e2e/**/*.ts'],
    languageOptions: {
      globals: globals.browser,
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,

      // The streaming render path deliberately reads and writes a ref during
      // render: the token memo in MessageRow is what keeps per-token commits
      // out of React state. See markdown/lex.ts.
      'react-hooks/refs': 'off',

      // Error, with three annotated exemptions at the call sites: a scroll
      // counter, a live-region re-announce, and a fetch-on-open. The rule
      // caught a real one — `streaming` was derived through an effect.
      'react-hooks/set-state-in-effect': 'error',

      'react-refresh/only-export-components': 'off',

      // A few type-aware rules worth the parse cost. NOT the full
      // recommendedTypeChecked set: `unbound-method` fires on every
      // `onClick(): void` prop declaration in the codebase — 47 of them, none
      // involving `this`.
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/no-unnecessary-type-assertion': 'error',

      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },

  {
    files: ['scripts/**/*.mjs', '*.config.ts', 'vitest.config.ts'],
    languageOptions: { globals: globals.node },
  },
  {
    files: ['scripts/**/*.mjs'],
    ...tseslint.configs.disableTypeChecked,
  },

  // Playwright's fixture argument is named `use`, which the hooks plugin
  // reads as a React `use()` call outside a component.
  {
    files: ['e2e/**/*.ts'],
    rules: { 'react-hooks/rules-of-hooks': 'off' },
  },
);
