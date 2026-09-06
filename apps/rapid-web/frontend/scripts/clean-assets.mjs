// Clears assets/ before a build. vite.config.ts must set emptyOutDir:false
// (outDir points into the Python package), so nothing else removes the
// previous build's hashed files.

import { existsSync, readdirSync, rmSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const ASSETS_DIR = join(resolve(import.meta.dirname, '../../rmlx_web/static'), 'assets');

if (!existsSync(ASSETS_DIR)) {
  process.exit(0);
}

if (!statSync(ASSETS_DIR).isDirectory()) {
  console.error(`\n  clean-assets: ${ASSETS_DIR} is not a directory.\n`);
  process.exit(1);
}

const removed = readdirSync(ASSETS_DIR).length;
rmSync(ASSETS_DIR, { recursive: true, force: true });
console.log(`  clean-assets: removed ${removed} stale file(s)`);
