// Directory entry point for the documented Node 22 command.
import './core.test.js';

import './races.test.js';

import './shopping.test.js';

import './round5.test.js';

import './round6.test.js';

import './round7.test.js';

import './scenarios.test.js';

import './production.test.js';

import './round-a.test.js';

import './round-b.test.js';

import './ui.test.js';

import './ui-lifecycle.test.js';

import './ui-round2.test.js';

import './round-c.test.js';

import './security-review.test.js';

import './perception.test.js';

import './round-d.test.js';

import './round-e.test.js';

import './local-tuning.test.js';

import './local-round2.test.js';

import './round12.test.js';

import './date-fare.test.js';

import './security-r15.test.js';

import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
test('every test file is imported by the directory entry point', () => {
  const index = readFileSync(new URL('./index.js', import.meta.url), 'utf8');
  for (const f of readdirSync(new URL('.', import.meta.url)).filter(f => f.endsWith('.test.js'))) assert.ok(index.includes(`'./${f}'`), f);
});
