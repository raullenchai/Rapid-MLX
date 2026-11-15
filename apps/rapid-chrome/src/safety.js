// Fail closed: exact origins are intentionally stricter than registrable domains.
import { endpoint } from './endpoint.js';
export const engineBase = endpoint;
export function safeURL(value, base) {
  let u;
  try { u = new URL(value); } catch { throw new Error('blocked_url: malformed URL or host; use an absolute HTTP(S) URL without spaces in the host'); }
  const authority = String(value).match(/^https?:\/\/([^/?#]*)/i)?.[1] || '';
  if (!authority || /[%\s\\]/.test(authority) || (!u.hostname.startsWith('[') &&
      u.hostname.replace(/\.$/, '').split('.').some(label => !/^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/i.test(label)))) {
    throw new Error('blocked_url: malformed host; use a valid HTTP(S) hostname without encoded spaces');
  }
  const engine = new URL(base);
  const local = h => ['127.0.0.1', 'localhost', '[::1]'].includes(h);
  if (!['http:', 'https:'].includes(u.protocol) || u.username || u.password ||
      u.hostname === 'chromewebstore.google.com' ||
      (u.hostname === 'chrome.google.com' && /^\/webstore(?:\/|$)/.test(u.pathname)) ||
      u.origin === engine.origin ||
      (local(u.hostname) && local(engine.hostname) && u.port === engine.port)) {
    throw new Error('blocked_url: browser internals, Web Store, credentials, files, and the engine are unavailable');
  }
  return u;
}
export function needsSiteApproval(url, approvedSites) {
  return !approvedSites.has(new URL(url).origin);
}
const consequential = /\b(submit|send|pay|buy|order|checkout|book|reserve|delete|remove|confirm|publish|post|transfer|purchase|enviar|pagar|comprar|eliminar|confirmar|publicar|envoyer|payer|acheter|supprimer|confirmer|publier|senden|bezahlen|kaufen|löschen|bestätigen)\b|发送|發送|支付|付款|购买|購買|下单|下單|订单|訂單|结账|結帳|删除|刪除|移除|确认|確認|发布|發佈|發表|转账|轉帳|送信|購入|削除|送金|결제|삭제|전송/iu;
const sensitive = /password|passwd|credit.?card|card.?number|card.?holder|security.?code|\bcvv\b|\bcvc\b|cc-|密码|密碼|信用卡|银行卡|銀行卡|卡号|卡號/iu;
// Financial commits always require human activation, including opaque JS state.
const financialCommit = /\b(?:buy|purchase|pay|payment|place\s+(?:an?\s+)?order|confirm\s+(?:an?\s+)?order|checkout\s+complete|complete\s+checkout|donate|subscribe|authorize(?:\s+payment)?|finalize|complete\s+purchase|transfer|send\s+money|tip|pagar|comprar|payer|acheter|bezahlen|kaufen)\b|支付|付款|购买|購買|下单|下單|订单|訂單|结账|結帳|转账|轉帳|購入|送金|결제/iu;
const destructive = /\b(?:deactivate|close\s+account|delete|erase|remove\s+account|unsubscribe|cancel\s+(?:subscription|membership)|revoke|reset|eliminar|supprimer|löschen)\b|关闭账户|關閉帳戶|删除|刪除|移除|削除|삭제/iu;
function financialContext(target) {
  // Match paths only: a merchant hostname or query parameter is not an action.
  let path = '';
  try { path = decodeURIComponent(new URL(target.action, 'https://page.invalid').pathname); } catch {}
  return financialCommit.test([target.name,target.label,target.form,target.formName,target.submitter].filter(Boolean).join(' ')) ||
    financialCommit.test(path.replace(/[-_/]+/g,' ')) || /(?:^|\/)(?:checkout|order)\b/i.test(path);
}
function consequentialContext(target) {
  // A lookup submitter may repeat its label in form text. Only this exact
  // phrase is benign; other effects in the same context must still win.
  const context = [target.form,target.formName,target.submitter,target.action,target.label,target.id]
    .filter(Boolean).join(' ').replace(/\bsubmit search\b/gi,'search');
  return consequential.test(context) || destructive.test(context) || financialCommit.test(context) || financialContext(target);
}
// Positive browse context is required for ambiguous location/Go controls.
// Account writes, opaque endpoints and explicit effects always keep the gate.
export function preparatoryContext(t = {}) {
  const context = [t.landmark,t.heading,t.formName,t.form,t.action,t.id,t.autocomplete].join(' ');
  const own = String(t.name || t.label || '').trim();
  const account = /^(?:continue|next|save|submit|confirm)$/i.test(own) || /saved|account|profile|billing|payment|checkout|address[-_/ ]?book|save (?:address|location)|update|delete|reserve|booking|purchase/i.test(context+' '+own);
  const writes = t.method && !['get','dialog'].includes(t.method.toLowerCase());
  let path = '';
  try { path = new URL(t.action || t.href || '/', 'https://page.invalid').pathname; } catch {}
  // One editable query is structural evidence; GET alone does not make a write safe.
  const queryLookup = t.method?.toLowerCase() === 'get' && t.singleQuery && /search|lookup|forecast|zipcity|(?:find|enter).*?(?:location|city|zip)/i.test(context);
  const endpoint = (queryLookup && !/update|save|submit|delete|account|profile|address[-_/ ]?book/i.test(path)) || !t.action || /(?:^|\/)(?:search|s|browse|discover|find|restaurants?)(?:\/|$)/i.test(path) || path === '/';
  if (!t.tag || account || writes || consequentialContext(t) || !endpoint || t.paymentForm || t.sensitiveForm || t.longName) return false;
  if (queryLookup && /^(?:go|search|find|submit search)$/i.test(own)) return true;
  const headerLocation = /^(?:HEADER|NAV)\b/.test(t.landmark || '') && /^(?:location .+|current address: .+\. Or select a new address\.)$/i.test(own);
  const cityChoice = /^(?:discover restaurants|order groceries for delivery or pickup)/i.test(t.heading || '') && /^(?:[A-Z][a-z]+)(?:[ -][A-Z][a-z]+){0,5}(?:, [A-Z]{2})?$/.test(own);
  const browse = /search|browse|discover|find|delivery near|shop(?:ping)?|restaurants?|groceries|filters?|results|autocomplete|(?:choose|select|search|browse).*?(?:location|city|region|area)/i.test(context);
  const control = ['BUTTON','INPUT','SELECT','A'].includes(t.tag) || ['option','combobox','slider','menuitem','button'].includes(t.role);
  return control && (headerLocation || cityChoice ||
    (/^(?:main (?:navigation )?menu|menu|navigation)$/i.test(own) && /navigation|menu/i.test(context)) ||
    (Boolean(t.suggestionQuery) && own.toLocaleLowerCase().includes(t.suggestionQuery.toLocaleLowerCase())) ||
    (browse && (/^(?:location .+|current address: .+|all filters(?: \(\d+\))?|filters?|go|search|find|apply(?: filters)?|location|change location|region|address|delivery address|select (?:location|city|region)|date|dates|time|party size|guests?|people)$/i.test(own) ||
      ['option','combobox','slider'].includes(t.role) || (t.choiceContext && /location|region|area|city/i.test(context) && ['button','menuitem'].includes(t.role) && /^(?:choose|select|search|browse|delivery|shop|location)/i.test(t.heading || '')) || ['search','date','time','range'].includes(t.type) || /location|region|city|zip|postal|address|date|time|party|guest/i.test(t.id || ''))));
}
export function classifyTarget(target = {}) {
  const text = [target.name, target.form, target.label, target.type, target.autocomplete, target.id].filter(Boolean).join(' ');
  // Form context is scoped to the target's form, never the entire page.
  // Explicit effects win over search cues (e.g. Search + Confirm purchase).
  const effect = [target.name, target.label, target.form, target.formName, target.submitter, target.action, target.id].filter(Boolean).join(' ');
  // Inspect the complete label: padding and sentence breaks must not hide intent.
  const own = String(target.name || target.label || '').trim();
  // Page-owned ARIA is not evidence that a control merely discloses content.
  const disclosure = /^(?:flight details|view details|show details|details about|expand|collapse|show more|load more)\b/i.test(own);
  const pagination = /^(?:next(?: page)?|previous(?: page)?|page \d+)$/i.test(own);
  const filterControl = /^(?:filters?|filter by .+|apply(?: filters)?|clear filters|price(?: filter)?(?:, (?:Not selected|Selected))?|nonstop|airlines|stops)$/i.test(own);
  const browsing = /^(?:clear|reply|compose|list|grid|arrow (?:left|right)|cart options|view cart\. Items in cart: \d+, View cart|(?:view )?cart(?:[., ]+\d+ unique items in your cart)?|\d+ items?|\d\+ rating, Guest rating, (?:Not selected|Selected))$/i.test(own);
  const harmless = /^(?:done|reject optional|reject all non-essential cookies|close(?: dialog| modal)?|dismiss|cancel|not now|back|got it|acknowledge|stops(?:,.*)?|sort(?: by.*)?|关闭(?:对话框|弹窗)?|關閉(?:對話框|彈窗)?|取消|暂不|暫不|经停|經停|排序)$/iu.test(own.trim());
  const filter = ['checkbox','radio','tab','option','menuitem','switch'].includes(target.role);
  // Filter and dismissal controls do not inherit unrelated form submitters.
  // Explicit destructive/purchase names still win (including close account).
  const filterName = /^(?:order by|order history|confirm filters?|订单记录|訂單記錄|确认筛选|確認篩選)/iu.test(own)
    ? own.replace(/\b(?:order|confirm)\b|订单|訂單|确认|確認/giu,'') : own;
  const financial = financialContext(target);
  const contextEffect = consequentialContext(target);
  const cart = isCartAddition(own) && !contextEffect && !target.sensitiveForm && !target.paymentForm;
  const searchSubmit = /^submit search$/i.test(own) && !contextEffect && !target.sensitiveForm && !target.paymentForm && (!target.method || target.method.toLowerCase() === 'get');
  const explicit = (!searchSubmit && consequential.test(filter ? filterName : own.replace(/^add \d+ to order(?=\s*(?:[•·]\s*[$€£¥]\s*[\d.,]+)?$)/i, 'add to cart'))) || /\b(?:place|submit|confirm)\s+(?:an?\s+)?(?:order|payment)\b|确认订单|確認訂單/iu.test(own) || /\bclose account\b|关闭账户|關閉帳戶/iu.test(own);
  const search = searchSubmit || /\b(search|find)\b|搜索|搜尋|查找|查询|查詢/iu.test(effect) || target.type === 'search' || (/^go$/i.test(own) && (!target.tag || preparatoryContext(target)));
  const genericSubmit = /\bsubmit\b|提交/iu.test([target.name,target.label].filter(Boolean).join(' ')) || (target.type === 'submit' && Boolean(target.method || target.action || target.form || target.submitter));
  // DOM text can be just '22' while its owning AX gridcell names the full date.
  const fullDate = /^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:[\s,]+(?:\d{1,4}|MON|TUE|WED|THU|FRI|SAT|SUN|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC))*$/i;
  const shortDate = /^(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?(?:[1-9]|[12]\d|3[01])$/i.test(own);
  // Calendar labels can carry fare/availability hints. AX may also append
  // opaque annotations (r14: 'Sunday, November 22, 2026 ????'). Use its date
  // prefix as structural evidence, but validate the clicked label separately.
  const fullDatePrefix = /^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:3[01]|[12]\d|[1-9])(?:,?\s+\d{4})?(?=$|[\s,])/i;
  const shortDatePrefix = /^(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+)?(?:3[01]|[12]\d|[1-9])(?=$|[\s,])/i;
  const datePrefix = own.match(fullDatePrefix)?.[0] || own.match(shortDatePrefix)?.[0];
  const hint = datePrefix ? own.slice(datePrefix.length) : '';
  const dateHint = /^(?:[\s,]+|[$€£¥]\s*\d[\d,.]*|\d[\d,.]*\s*(?:US dollars?|dollars?|USD|EUR|GBP|seats?|tickets?)\b|\b(?:lowest|cheapest|price|fare|available|unavailable|availability|sold out|no flights)\b)+$/i;
  const hintedDate = Boolean(datePrefix && hint && dateHint.test(hint)) && ['button','gridcell'].includes(target.role);
  const dateEvidence = target.datePicker || (target.dateCell && (fullDatePrefix.test(target.dateCell) || fullDatePrefix.test(target.dateCellAX || '')));
  const calendarDate = fullDate.test(own) || ((shortDate || hintedDate) && dateEvidence && !target.form && !target.formName && !target.submitter && !target.action && !target.method && target.type !== 'submit');
  // Exact preparatory labels on non-submitting buttons outside forms. Final
  // saves/cancellations and all financial/sensitive context remain gated.
  const preparation = target.type === 'button' && !target.form && !target.formName && !target.action && !target.method && !target.submitter &&
    /^(?:Create event|[A-Z]{1,3}\d{1,3}\s+[—–-]\s+[$€£¥]\s*\d+(?:[.,]\d{2})?)$/i.test(own);
  // Error-page reload outside any form re-renders the page; "Retry" can replay a payment and stays gated.
  const reload = /^(?:reload|refresh|try again|重新加载|重新載入|刷新)$/iu.test(own) && target.type !== 'submit' && !target.form && !target.formName && !target.action && !target.method && !target.submitter && !financialContext(target) && !target.sensitiveForm && !target.paymentForm;
  const writes = target.method && !['get','dialog'].includes(target.method.toLowerCase());
  // ARIA roles and benign labels cannot turn a submitting control into a
  // dismissal/filter when its form declares an effect. Non-submitting filters
  // still avoid inheriting unrelated form submitters.
  const submitEffect = ['submit','image'].includes(target.type) && (contextEffect ||
    (filter && target.method && target.method.toLowerCase() !== 'get'));
  const contextual = preparatoryContext(target);
  const accountWrite = /saved|account|profile|billing|address[-_/ ]?book/i.test([target.landmark,target.heading,target.formName,target.action].join(' '));
  const ambiguous = (target.role === 'button' || target.type === 'button') && !(contextual || preparation || reload || disclosure || pagination || filterControl || harmless || browsing || calendarDate || filter || cart || search);
  return { search, financial, consequential: Boolean(submitEffect) || destructive.test(own) || explicit || accountWrite || ambiguous || Boolean(target.longName || target.sensitiveForm || target.paymentForm) || (!(contextual || reload || searchSubmit || disclosure || pagination || filterControl || harmless || browsing || calendarDate || filter || cart) && (consequential.test(effect) || destructive.test(effect) || (!search && Boolean(genericSubmit || writes)))), sensitive: sensitive.test(text) };
}

export function isCartAddition(name = '') {
  return /^add \d+ to order(?:\s*[•·]\s*[$€£¥]\s*[\d.,]+)?$/i.test(name.trim()) || /^add \d+ (?:ct|count|pack) .+/i.test(name.trim()) || /\b(?:add(?:[ -]+to)?[ -]+(?:cart|bag|basket)|quick[ -]+add|increase(?:[ -]+quantity)?|increment(?:[ -]+quantity)?|add another)\b|^(?:\+|plus)$|加入(?:购物车|購物車|购物袋)|添加到(?:购物车|購物車)|快速添加|增加(?:数量|數量)/iu.test(name);
}
export function additionKey(url, target) {
  if (!isCartAddition(target.name)) return null;
  const u = new URL(url);
  // Stable across ref refreshes, tracking parameters and product-page aliases.
  const product = target.product || u.pathname.replace(/\/ref=.*$/, '');
  return JSON.stringify([u.origin,product,target.name.toLocaleLowerCase().replace(/\s+/g,' ').trim()]);
}

// Only explicit user URLs, observed outline links, and root/search entry points
// may be typed by the model. Site approval still runs separately for new origins.
export class NavigationPolicy {
  constructor(task = '') {
    this.seen = new Set(); this.given = new Set();
    for (const match of task.matchAll(/https?:\/\/[^\s<>"']+/gi)) {
      try { this.given.add(new URL(match[0].replace(/[.,;!?)]+$/, '')).href); } catch {}
    }
  }
  observe(href) { try { this.seen.add(new URL(href).href); } catch {} }
  check(url) {
    if (this.given.has(url.href) || this.seen.has(url.href)) return;
    const root = url.pathname === '/' && !url.search;
    const searchPath = /^(?:\/s|\/search|\/search\/|\/site\/searchpage\.jsp|\/travel\/flights)$/i.test(url.pathname);
    const search = searchPath && ['q','k','st','query','search'].some(k => url.searchParams.get(k)?.trim());
    const wikiSearch = url.pathname === '/w/index.php' && Boolean(url.searchParams.get('search')?.trim()) &&
      [...url.searchParams.keys()].every(k => k === 'search');
    if (root || search || wikiSearch) return; // A new origin must then receive site approval.
    throw new Error('unobserved_url: URL was not supplied by the user or seen as an href in an outline in this task. Click a link from the page or use site search. Use an observed link or open the site root and search; this deep URL was not visited. If no link or search ref is available, request a focused read.');
  }
}

// Deliberately small list; exact hosts and their subdomains only.
export function sensitivePage(url) {
  const u = new URL(url), host = u.hostname.toLowerCase().replace(/\.$/, '');
  const vaults = ['1password.com','bitwarden.com','lastpass.com','dashlane.com','keepersecurity.com','proton.me'];
  if (vaults.some(d => host === d || host.endsWith('.'+d)) &&
      (host !== 'proton.me' && !host.endsWith('.proton.me') || /pass|vault/.test(host+u.pathname))) return true;
  const banks = ['chase.com','bankofamerica.com','wellsfargo.com','citi.com','capitalone.com','paypal.com'];
  return banks.some(d => host === d || host.endsWith('.'+d)) && /login|logon|signin|sign-in|auth|banking|secure/i.test(host+u.pathname);
}

// Classify only the bound target name, never arbitrary form values in the card.
export function approvalTitleKey(name = '') {
  if (/\b(?:buy|pay|purchase|checkout|book|reserve|(?:place|confirm|submit) (?:an? )?order)\b|^order$|购买|購買|支付|付款|下单|下單|结账|結帳/iu.test(name)) return 'allowPurchase';
  if (destructive.test(name) || /\bremove\b/iu.test(name)) return 'allowDelete';
  if (/\b(?:send|publish|post)\b|发送|發送|发布|發佈|送信/iu.test(name)) return 'allowSend';
  if (/\b(?:sign in|log in|login|sign up)\b|登录|登入|登錄/iu.test(name)) return 'allowSignIn';
  if (/\bsubmit\b|提交/iu.test(name)) return 'allowSubmit';
  if (isCartAddition(name)) return 'allowCart';
  return 'allowAction';
}
