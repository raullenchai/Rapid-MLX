import { patterns } from './i18n-patterns.js';
export const t = (key, substitutions) => globalThis.chrome?.i18n?.getMessage(key,substitutions) || key;
// Runtime diagnostics may contain page labels. Translate extension text while
// leaving page/model content intact. Catalog templates use $1, $2, ... .
export function localize(text) {
  if (typeof text !== 'string') return text;
  const direct = t(text);
  if (direct !== text) return direct;
  for (const [template,key] of patterns) {
    const slots = [];
    const expression = template.split(/(\$\d+)/).map(part => {
      if (/^\$\d+$/.test(part)) { slots.push(Number(part.slice(1))-1); return '([\\s\\S]*?)'; }
      return part.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
    }).join('');
    const match = text.match(new RegExp(`^${expression}$`));
    if (match) {
      const args = []; slots.forEach((slot,i) => { args[slot] = match[i+1] === text ? text : localize(match[i+1]); });
      const result = t(key,args);
      return result === key ? text : result;
    }
  }
  return text.includes('\n') ? text.split('\n').map(localize).join('\n') : text;
}
export function translateDocument(doc = document) {
  doc.documentElement.lang = globalThis.chrome?.i18n?.getUILanguage() || 'en';
  for (const e of doc.querySelectorAll('[data-i18n]')) e.textContent = t(e.dataset.i18n);
  for (const attr of ['placeholder','aria-label','title']) {
    for (const e of doc.querySelectorAll(`[data-i18n-${attr}]`)) e.setAttribute(attr,t(e.getAttribute(`data-i18n-${attr}`)));
  }
}

// Tool results are structured diagnostics around untrusted page observations.
// Translate diagnostic values without rewriting raw AX outlines or JSON keys.
export function displayResult(result) {
  if (!result) return t('working');
  try {
    const visit = value => Array.isArray(value) ? value.map(visit) : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).map(([k,v]) => [k,visit(v)])) : typeof value === 'string' ? localize(value) : value;
    return JSON.stringify(visit(JSON.parse(result)),null,2);
  } catch { return localize(result); }
}
export const toolLabel = name => t(({read_page:'toolReadPage',click:'toolClick',fill:'toolFill',press:'toolPress',navigate:'toolNavigate',open_tab:'toolOpenTab',scroll:'toolScroll',wait:'toolWait',list_tabs:'toolListTabs',screenshot:'toolScreenshot'})[name] || name);
