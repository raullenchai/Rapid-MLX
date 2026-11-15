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
const consequential = /\b(send|pay|buy|order|checkout|book|reserve|delete|remove|confirm|publish|post|transfer|purchase|enviar|pagar|comprar|eliminar|confirmar|publicar|envoyer|payer|acheter|supprimer|confirmer|publier|senden|bezahlen|kaufen|löschen|bestätigen)\b|发送|發送|支付|付款|购买|購買|下单|下單|订单|訂單|结账|結帳|删除|刪除|移除|确认|確認|发布|發佈|發表|转账|轉帳|送信|購入|削除|送金|결제|삭제|전송/iu;
const sensitive = /password|passwd|credit.?card|card.?number|card.?holder|security.?code|\bcvv\b|\bcvc\b|cc-|密码|密碼|信用卡|银行卡|銀行卡|卡号|卡號/iu;
export function classifyTarget(target = {}) {
  const text = [target.name, target.form, target.label, target.type, target.autocomplete, target.id].filter(Boolean).join(' ');
  // Form context is scoped to the target's form, never the entire page.
  // Explicit effects win over search cues (e.g. Search + Confirm purchase).
  const effect = [target.name, target.label, target.form, target.formName, target.submitter, target.action, target.id].filter(Boolean).join(' ');
  // Long AX names include itinerary descriptions and unrelated booking copy.
  const own = String(target.name || target.label || '').trim().slice(0,100).split(/[.!?]\s/)[0];
  const disclosure = target.expanded === true || target.expanded === false ||
    ['true','false'].includes(target.expanded) || /^(?:flight details|view details|show details|expand|collapse)\b/i.test(own);
  const harmless = /^(?:close(?: dialog| modal)?|dismiss|cancel|not now|stops(?:,.*)?|sort(?: by.*)?|关闭(?:对话框|弹窗)?|關閉(?:對話框|彈窗)?|取消|暂不|暫不|经停|經停|排序)$/iu.test(own.trim());
  const filter = ['checkbox','radio','tab','option','menuitem','switch'].includes(target.role);
  // Filter and dismissal controls do not inherit unrelated form submitters.
  // Explicit destructive/purchase names still win (including close account).
  const filterName = own.replace(/\b(?:order|confirm)\b|订单|訂單|确认|確認/giu,'');
  const explicit = consequential.test(filter ? filterName : own) || /\b(?:place|submit|confirm)\s+(?:an?\s+)?(?:order|payment)\b|确认订单|確認訂單/iu.test(own) || /\bclose account\b|关闭账户|關閉帳戶/iu.test(own);
  const search = /\b(search|go|find)\b|搜索|搜尋|查找|查询|查詢/iu.test(effect) || target.type === 'search';
  const genericSubmit = /\bsubmit\b|提交/iu.test(text) || target.type === 'submit';
  const writes = target.method && !['get','dialog'].includes(target.method.toLowerCase());
  return { search, consequential: (!disclosure && explicit) || (!(disclosure || harmless || filter || isCartAddition(own)) && (consequential.test(effect) || (!search && Boolean(genericSubmit || writes)))), sensitive: sensitive.test(text) };
}

export function isCartAddition(name = '') {
  return /\b(?:add(?:[ -]+to)?[ -]+(?:cart|bag|basket)|quick[ -]+add|increase(?:[ -]+quantity)?|increment(?:[ -]+quantity)?|add another)\b|^(?:\+|plus)$|加入(?:购物车|購物車|购物袋)|添加到(?:购物车|購物車)|快速添加|增加(?:数量|數量)/iu.test(name);
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
    if (root || search) return; // A new origin must then receive site approval.
    throw new Error('unobserved_url: URL was not supplied by the user or seen as an href in an outline in this task. Click a link from the page or use site search; do not invent product or venue URLs.');
  }
}
