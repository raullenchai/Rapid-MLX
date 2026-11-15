import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { deflateRawSync } from 'node:zlib';
const root = fileURLToPath(new URL('../',import.meta.url));
const read = p => readFile(path.join(root,p));
const manifest = JSON.parse(await read('manifest.json'));
function check(condition, message) { if (!condition) throw new Error(message); }
check(manifest.version === JSON.parse(await read('package.json')).version,'Manifest and package versions differ');
check(manifest.manifest_version === 3,'Manifest must use version 3');
check(/^\d+\.\d+\.\d+$/.test(manifest.version),'Invalid version');
check(manifest.default_locale === 'en','Missing default locale');
check(manifest.background?.type === 'module','Background must be a module');
check(manifest.permissions?.every(p => ['debugger','tabs','tabGroups','storage','sidePanel','scripting'].includes(p)),'Unexpected permission');
check(JSON.stringify(manifest.host_permissions) === JSON.stringify(['http://127.0.0.1/*','http://localhost/*']),'Required hosts must be limited to HTTP loopback');
check(JSON.stringify(manifest.optional_host_permissions) === JSON.stringify(['https://*/*','http://*/*']),'Optional hosts must be declared');
const files = [];
async function collect(dir = '') {
  for (const ent of (await readdir(path.join(root,dir),{withFileTypes:true})).sort((a,b) => a.name.localeCompare(b.name,'en'))) {
    const name = path.posix.join(dir,ent.name);
    if (ent.isSymbolicLink()) throw new Error(`Symlink not allowed: ${name}`);
    if (ent.isDirectory() && ['src','icons','_locales'].includes(dir.split('/')[0] || ent.name)) await collect(name);
    else if (ent.isFile() && (dir || ['manifest.json','background.js','panel.html','panel.css','panel.js'].includes(name))) files.push(name);
  }
}
await collect(); files.sort();
const contents = new Map(await Promise.all(files.map(async f => [f,await read(f)])));
function referenced(file) { check(contents.has(file),`Missing referenced file: ${file}`); }
for (const f of [manifest.background.service_worker,manifest.side_panel.default_path,...Object.values(manifest.icons || {}),...Object.values(manifest.action.default_icon || {}),manifest.action.default_popup,manifest.options_page,manifest.options_ui?.page,...(manifest.content_scripts || []).flatMap(c => [...(c.js || []),...(c.css || [])])].filter(Boolean)) referenced(f);
for (const [name,buf] of contents) {
  const text = buf.toString();
  if (/\.(?:html|js|css)$/.test(name)) {
    const refs = name.endsWith('.html') ? [...text.matchAll(/(?:src|href)="([^"#]+)"/g)].map(m => m[1]) : name.endsWith('.js') ? [...text.matchAll(/(?:from\s*|import\s*)['"](\.[^'"]+)['"]/g)].map(m => m[1]) : [...text.matchAll(/url\(['"]?([^)'"#]+)['"]?\)/g)].map(m => m[1]);
    for (const ref of refs) referenced(path.posix.normalize(path.posix.join(path.posix.dirname(name),ref)));
  }
}
const en = JSON.parse(contents.get('_locales/en/messages.json'));
const zh = JSON.parse(contents.get('_locales/zh_CN/messages.json'));
check(JSON.stringify(Object.keys(en).sort()) === JSON.stringify(Object.keys(zh).sort()),'Locale keys differ');
for (const locale of [en,zh]) {
  check(new Set(Object.keys(locale).map(k => k.toLowerCase())).size === Object.keys(locale).length,'Locale keys collide ignoring case');
  for (const [key,entry] of Object.entries(locale)) check(typeof entry.message === 'string' && entry.message.trim(),`Empty translation: ${key}`);
  for (const [,buf] of contents) for (const match of buf.toString().matchAll(/__MSG_([\w]+)__|data-i18n(?:-[\w-]+)?="([\w]+)"|\bt\(['"]([\w]+)['"]/g)) check(locale[match[1] || match[2] || match[3]],`Missing translation: ${match[0]}`);
}
// ZIP32, fixed DOS epoch, no platform metadata. Sorted entries and fixed
// compression make identical source trees produce byte-identical archives.
function crc32(buf) {
  let crc = 0xffffffff;
  for (const b of buf) { crc ^= b; for (let i=0;i<8;i++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1)); }
  return (crc ^ 0xffffffff) >>> 0;
}
const local = [], central = []; let offset = 0;
for (const [name,data] of contents) {
  const filename = Buffer.from(name), compressed = deflateRawSync(data,{level:9}), crc = crc32(data);
  const h = Buffer.alloc(30); h.writeUInt32LE(0x04034b50); h.writeUInt16LE(20,4); h.writeUInt16LE(0x800,6); h.writeUInt16LE(8,8); h.writeUInt16LE(33,12); h.writeUInt32LE(crc,14); h.writeUInt32LE(compressed.length,18); h.writeUInt32LE(data.length,22); h.writeUInt16LE(filename.length,26);
  local.push(h,filename,compressed);
  const c = Buffer.alloc(46); c.writeUInt32LE(0x02014b50); c.writeUInt16LE(20,4); c.writeUInt16LE(20,6); h.copy(c,8,6,30); c.writeUInt32LE(offset,42); central.push(c,filename);
  offset += h.length+filename.length+compressed.length;
}
const directory = Buffer.concat(central), end = Buffer.alloc(22); end.writeUInt32LE(0x06054b50); end.writeUInt16LE(files.length,8); end.writeUInt16LE(files.length,10); end.writeUInt32LE(directory.length,12); end.writeUInt32LE(offset,16);
await mkdir(path.join(root,'dist'),{recursive:true});
const output = `dist/rapid-mlx-chrome-${manifest.version}.zip`;
await writeFile(path.join(root,output),Buffer.concat([...local,directory,end]));
console.log(`Validated ${files.length} files; wrote ${output}`);
