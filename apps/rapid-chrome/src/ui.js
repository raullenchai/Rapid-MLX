// Untrusted text is always inserted as text nodes. Links permit web URLs only.
export function webURL(value) {
  try { const url = new URL(value); return ['http:','https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch { return null; }
}
function inline(doc, parent, text) {
  const tokens = /\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^\s)]+)\)|(https?:\/\/[^\s<>]+)/g;
  let end = 0;
  for (const m of text.matchAll(tokens)) {
    parent.append(doc.createTextNode(text.slice(end,m.index)));
    const url = webURL(m[4] || m[5]);
    const node = doc.createElement(m[1] ? 'strong' : m[2] ? 'code' : url ? 'a' : 'span');
    node.textContent = m[1] || m[2] || m[3] || m[5];
    if (url) { node.href = url; node.target = '_blank'; node.rel = 'noopener noreferrer'; }
    parent.append(node); end = m.index + m[0].length;
  }
  parent.append(doc.createTextNode(text.slice(end)));
}
export function markdown(doc, parent, text, {t = key => key, copy = async () => {}, copied = () => {}} = {}) {
  let code = null, list = null, table = null;
  const lines = text.split('\n');
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line.startsWith('```')) {
      if (code) code = null;
      else {
        const block = doc.createElement('div'); block.className = 'code-block';
        const control = doc.createElement('button'); control.type = 'button'; control.textContent = t('copyCode');
        const pre = doc.createElement('pre'); code = doc.createElement('code'); pre.append(code);
        control.onclick = async () => { try { await copy(pre.textContent); copied(control); } catch { control.textContent = t('traceFailed'); } };
        block.append(control,pre); parent.append(block);
      }
      list = null; table = null; continue;
    }
    if (code) { code.append(doc.createTextNode(line+'\n')); continue; }
    if (line.includes('|') && (table || /^\s*\|?\s*:?-{3,}/.test(lines[index+1] || ''))) {
      if (!table) { const wrap = doc.createElement('div'); wrap.className = 'table-scroll'; wrap.tabIndex = 0; table = doc.createElement('table'); wrap.append(table); parent.append(wrap); }
      if (/^\s*\|?\s*:?-{3,}/.test(line)) continue;
      const row = doc.createElement('tr');
      for (const cell of line.replace(/^\s*\||\|\s*$/g,'').split('|')) { const td = doc.createElement(index+1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[index+1]) ? 'th' : 'td'); inline(doc,td,cell.trim()); row.append(td); }
      table.append(row); list = null; continue;
    }
    table = null;
    const item = line.match(/^\s*(?:([-*])|(\d+)\.)\s+(.+)/);
    if (item) {
      const tag = item[1] ? 'ul' : 'ol';
      if (!list || list.tagName.toLowerCase() !== tag) { list = doc.createElement(tag); parent.append(list); }
      const li = doc.createElement('li'); if (item[2]) li.value = Number(item[2]); inline(doc,li,item[3]); list.append(li);
    } else { list = null; const p = doc.createElement('p'); inline(doc,p,line.replace(/^#{1,6}\s+/,'')); parent.append(p); }
  }
}
export function stepLabel(step, t, fallback) {
  const args = step.args || {};
  if (step.label) return String(step.label).slice(0,180);
  if (['read_page','navigate','open_tab'].includes(step.name)) {
    try { const site = new URL(args.url || step.url).hostname; return args.query ? t('stepFinding',[String(args.query).slice(0,60),site]) : t('stepLooking',[site]); } catch { return t('stepReading'); }
  }
  if (step.name === 'click') return step.targetLabel ? t('stepClicking',[String(step.targetLabel).slice(0,100)]) : t('stepClick');
  if (step.name === 'fill') return t('stepTyping');
  return fallback(step.name);
}
