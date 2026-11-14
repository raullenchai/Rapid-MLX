// Fixed browser-side observer; page text is always untrusted. Runs in an isolated world.
export function watchEffects() {
  const clean = s => String(s || '').replace(/\s+/g,' ').trim().slice(0,700);
  const visible = e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden';
  const snapshot = () => {
    const regions = [...document.querySelectorAll('dialog,[role="dialog"],[role="alertdialog"],[role="alert"],[role="status"],[aria-live]:not([aria-live="off"]),[aria-modal="true"],[data-state="open"],[class*="drawer"],[class*="sheet"],[class*="toast"]')]
      .filter(visible).map(e => clean(`${e.getAttribute('role') || e.tagName}: ${e.getAttribute('aria-label') || ''} ${e.innerText || e.textContent}`)).filter(Boolean).slice(0,30);
    const counters = [...document.querySelectorAll('a,button,[role="link"]')].filter(visible)
      .map(e => ({key:e.getAttribute('href') || e.id || clean(e.getAttribute('aria-label')).replace(/\d+/g,'#'), text:clean(`${e.getAttribute('aria-label') || ''} ${e.innerText || e.textContent} ${e.parentElement?.querySelector('[class*="badge"],[class*="count"],[id*="count"]')?.textContent || ''}`)}))
      .filter(e => /\b(cart|bag|basket)\b|购物车|購物車|购物袋/iu.test(e.text)).slice(0,20);
    return {url:location.href,title:document.title,regions,counters};
  };
  globalThis.rapidEffects?.observer.disconnect();
  const before = snapshot(), samples = [];
  const record = () => { samples.push(snapshot()); if (samples.length > 40) samples.splice(1,1); };
  const observer = new MutationObserver(record);
  observer.observe(document,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:['aria-label','aria-live','open','hidden','class']});
  globalThis.rapidEffects = {observer,samples,snapshot};
  return before;
}
export function collectEffects() {
  const state = globalThis.rapidEffects;
  if (!state) return [];
  state.observer.disconnect();
  const samples = [...state.samples,state.snapshot()];
  delete globalThis.rapidEffects;
  return samples;
}
export function effectReport(before, samples) {
  const after = samples.at(-1) || before;
  const counters = new Map((before.counters || []).map(c => [c.key,c.text]));
  const changed = new Map();
  for (const sample of samples) for (const c of sample.counters || []) {
    if (counters.get(c.key) !== c.text) changed.set(c.key,{before:counters.get(c.key) ?? null,after:c.text});
  }
  return {
    url:before.url === after.url ? null : {before:before.url,after:after.url},
    title:before.title === after.title ? null : {before:before.title,after:after.title},
    regions:[...new Set(samples.flatMap(s => s.regions || []).filter(s => !(before.regions || []).includes(s)))].slice(0,10),
    counters:[...changed.values()].slice(0,10),
    note:'Observed changes only; absence of change does not prove failure. Never repeat an add to verify it.'
  };
}
