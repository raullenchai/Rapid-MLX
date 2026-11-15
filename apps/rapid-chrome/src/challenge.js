// Detection only: never interact with, solve, or bypass a verification widget.
export function detectChallenge({url = '', title = '', nodes = [], frames = [], error = ''} = {}) {
  let site = 'this site';
  try { site = new URL(url).hostname || site; } catch {}
  const text = [title,...nodes.filter(n => !n.ignored && n.role?.value !== 'InlineTextBox').map(n => n.name?.value || '')].join(' ');
  if (/chrome-error:|^about:neterror/i.test(url) || /ERR_[A-Z_]+|HTTP[ /](?:4\d\d|5\d\d)|navigation_failed/i.test(error + ' ' + text)) {
    return {kind:'load_error',message:`${site} could not load. Reload or resolve the error in that tab, then press Continue. Rapid will verify the page again.`};
  }
  const challengeText = /press\s+(?:and|&)\s+hold|verify (?:that )?you(?: are|’re|'re) (?:a )?human|human verification|performing security verification|checking your browser|just a moment|complete the (?:captcha|security check)|\bi(?: am|'m|’m) not a robot\b|请完成.{0,10}验证|請完成.{0,10}驗證/i.test(text);
  // Only visible widgets, not background reCAPTCHA telemetry/hidden anchors.
  const widget = frames.some(f => f.visible && /challenges\.cloudflare\.com|\/turnstile\/|(?:google\.com|recaptcha\.net)\/recaptcha\/|hcaptcha\.com|captcha-delivery\.com|datadome/i.test(f.url));
  const embedded = nodes.some(n => /checkbox|button/i.test(n.role?.value || '') && /captcha|not a robot|human verification/i.test(n.name?.value || ''));
  if (challengeText || widget || embedded) return {kind:'human_check',message:`${site} requires a human check. Complete the verification in that tab, then press Continue. Rapid will verify that the check is gone.`};
  return null;
}

// Conservative: unknown or action-bearing requests keep the human pause. Only
// user text determines task intent; page content must never opt into this path.
export function informationTask(task = '') {
  const text = task.replace(/https?:\/\/[^\s]+/gi,'')
    .replace(/\b(?:do not|don't|never)\s+(?:buy|bid|book|reserve|purchase|pay|checkout)\b/gi,'');
  return /\b(?:what|when|where|how|weather|find|research|read|summarize|compare|tell|list|look up)\b/i.test(text) &&
    !/\b(?:add|cart|buy|bid|purchase|pay|order|book|reserve|cancel|subscribe|unsubscribe|send|reply|submit|save|delete|remove|change|update|fill|sign|log|schedule|refund|transfer|checkout)\b/i.test(text);
}
