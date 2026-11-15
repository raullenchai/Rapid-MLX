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
export function markdown(doc, parent, text) {
  let code = null, list = null;
  for (const line of text.split('\n')) {
    if (line.startsWith('```')) { if (code) code = null; else { code = doc.createElement('pre'); parent.append(code); } list = null; continue; }
    if (code) { code.append(doc.createTextNode(line+'\n')); continue; }
    const item = line.match(/^\s*(?:([-*])|\d+\.)\s+(.+)/);
    if (item) {
      const tag = item[1] ? 'ul' : 'ol';
      if (!list || list.tagName.toLowerCase() !== tag) { list = doc.createElement(tag); parent.append(list); }
      const li = doc.createElement('li'); inline(doc,li,item[2]); list.append(li);
    } else { list = null; const p = doc.createElement('p'); inline(doc,p,line); parent.append(p); }
  }
}
export function stepLabel(step, t, fallback) {
  const args = step.args || {};
  if (step.label) return String(step.label).slice(0,180);
  if (['read_page','navigate','open_tab'].includes(step.name)) {
    try { return t('stepLooking',[new URL(args.url || step.url).hostname]); } catch { return t('stepReading'); }
  }
  if (step.name === 'click') return step.targetLabel ? t('stepClicking',[String(step.targetLabel).slice(0,100)]) : t('stepClick');
  if (step.name === 'fill') return t('stepTyping');
  return fallback(step.name);
}
