// Fail closed: exact origins are intentionally stricter than registrable domains.
export function engineBase(value) {
  const u = new URL(value);
  if (u.protocol !== 'http:' || !['127.0.0.1', 'localhost'].includes(u.hostname) ||
      u.username || u.password || u.search || u.hash || !['', '/'].includes(u.pathname)) {
    throw new Error('Engine URL must be an HTTP loopback origin, e.g. http://127.0.0.1:8000');
  }
  return u.origin;
}
export function safeURL(value, base) {
  let u;
  try { u = new URL(value); } catch { throw new Error('blocked_url: use an absolute HTTP(S) URL'); }
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
const consequential = /\b(send|pay|buy|order|checkout|delete|remove|confirm|publish|post|transfer|purchase|enviar|pagar|comprar|eliminar|confirmar|publicar|envoyer|payer|acheter|supprimer|confirmer|publier|senden|bezahlen|kaufen|löschen|bestätigen)\b|发送|發送|支付|付款|购买|購買|下单|下單|订单|訂單|结账|結帳|删除|刪除|移除|确认|確認|发布|發佈|發表|转账|轉帳|送信|購入|削除|送金|결제|삭제|전송/iu;
const sensitive = /password|passwd|credit.?card|card.?number|card.?holder|security.?code|\bcvv\b|\bcvc\b|cc-|密码|密碼|信用卡|银行卡|銀行卡|卡号|卡號/iu;
export function classifyTarget(target = {}) {
  const text = [target.name, target.form, target.label, target.type, target.autocomplete, target.id].filter(Boolean).join(' ');
  // Form context is scoped to the target's form, never the entire page.
  // Explicit effects win over search cues (e.g. Search + Confirm purchase).
  const effect = [target.name, target.label, target.form, target.formName, target.submitter, target.action, target.id].filter(Boolean).join(' ');
  const search = /\b(search|go|find)\b|搜索|搜尋|查找|查询|查詢/iu.test(effect) || target.type === 'search';
  const genericSubmit = /\bsubmit\b|提交/iu.test(text) || target.type === 'submit';
  const writes = target.method && !['get','dialog'].includes(target.method.toLowerCase());
  return { search, consequential: consequential.test(effect) || (!search && Boolean(genericSubmit || writes)), sensitive: sensitive.test(text) };
}
