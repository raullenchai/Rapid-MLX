// Reproducible, dependency-free PNGs: a white R on a blue tile.
import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
function crc(bytes) { let c = 0xffffffff; for (const b of bytes) { c ^= b; for (let i=0;i<8;i++) c = (c>>>1)^((c&1)?0xedb88320:0); } return (c^0xffffffff)>>>0; }
function chunk(type, data) { const name = Buffer.from(type), len = Buffer.alloc(4), sum = Buffer.alloc(4); len.writeUInt32BE(data.length); sum.writeUInt32BE(crc(Buffer.concat([name,data]))); return Buffer.concat([len,name,data,sum]); }
const glyph = ['11110','10001','10001','11110','10100','10010','10001'];
for (const size of [16,32,48,128]) {
  const raw = Buffer.alloc((size*4+1)*size);
  for (let y=0;y<size;y++) for (let x=0;x<size;x++) {
    const gx = Math.floor((x/size-.28)/.09), gy = Math.floor((y/size-.18)/.09);
    const white = glyph[gy]?.[gx] === '1';
    const i = y*(size*4+1)+1+x*4;
    raw.set(white ? [255,255,255,255] : [56,93,229,255],i);
  }
  const header = Buffer.alloc(13); header.writeUInt32BE(size); header.writeUInt32BE(size,4); header[8]=8;header[9]=6;
  writeFileSync(fileURLToPath(new URL(`${size}.png`,import.meta.url)),Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]));
}
