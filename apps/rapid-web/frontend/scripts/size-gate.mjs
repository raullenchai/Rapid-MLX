// Post-build gate over apps/rapid-web/rmlx_web/static/.
//
// 1. INTEGRITY — every asset index.html references must exist on disk.
// 2. SIZE — total bytes against size-budget.json, written back so growth
//    shows up as a reviewable diff.

import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

const FRONTEND_DIR = resolve(import.meta.dirname, '..');
const OUT_DIR = resolve(FRONTEND_DIR, '../rmlx_web/static');
const BUDGET_FILE = join(FRONTEND_DIR, 'size-budget.json');

function fail(message) {
  console.error(`\n  size-gate: ${message}\n`);
  process.exit(1);
}

const html = readFileSync(join(OUT_DIR, 'index.html'), 'utf8');

// ------------------------------------------------------------- integrity

const referenced = [...html.matchAll(/\b(?:src|href)\s*=\s*"([^"]*)"/gi)]
  .map((match) => match[1])
  .filter((url) => url.startsWith('/static/'));

if (referenced.length === 0) {
  fail('index.html references no /static/ assets — did the build inline everything?');
}

for (const url of referenced) {
  const path = join(OUT_DIR, url.replace(/^\/static\//, ''));
  if (!statSync(path, { throwIfNoEntry: false })) {
    fail(`index.html references ${url}, but ${path} does not exist.`);
  }
}

// ------------------------------------------------------------------ size

let total = 0;
const walk = (dir) => {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path);
    else total += statSync(path).size;
  }
};
walk(OUT_DIR);

const kb = (n) => `${(n / 1000).toFixed(1)} KB`;
const budget = JSON.parse(readFileSync(BUDGET_FILE, 'utf8'));

if (total > budget.failBytes) {
  fail(`the build totals ${kb(total)}, over the ${kb(budget.failBytes)} hard limit.`);
}

if (total > budget.warnBytes) {
  console.warn(
    `\n  size-gate: ${kb(total)} total, over the ${kb(budget.warnBytes)} warning line ` +
      `(hard limit ${kb(budget.failBytes)}).\n`,
  );
}

if (budget.measuredBytes !== total) {
  budget.measuredBytes = total;
  writeFileSync(BUDGET_FILE, `${JSON.stringify(budget, null, 2)}\n`, 'utf8');
}

console.log(`  size-gate: OK — ${kb(total)} across ${referenced.length + 1} file(s)`);
