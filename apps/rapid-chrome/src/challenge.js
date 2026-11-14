// Detection only: never interact with, solve, or bypass a verification widget.
export function detectChallenge({url = '', title = '', nodes = [], frames = [], error = ''} = {}) {
  let site = 'this site';
  try { site = new URL(url).hostname || site; } catch {}
  const text = [title,...nodes.filter(n => !n.ignored && n.role?.value !== 'InlineTextBox').map(n => n.name?.value || '')].join(' ');
  if (/chrome-error:|^about:neterror/i.test(url) || /ERR_[A-Z_]+|HTTP[ /](?:4\d\d|5\d\d)|navigation_failed/i.test(error + ' ' + text)) {
    return {kind:'load_error',message:`${site} could not load. Reload or resolve the error in that tab, then press Continue. Rapid will verify the page again.`};
  }
  const challengeText = /press\s+(?:and|&)\s+hold|verify (?:that )?you(?: are|’re|'re) (?:a )?human|human verification|performing security verification|checking your browser|just a moment|complete the (?:captcha|security check)|\bi am not a robot\b/i.test(text);
  // Only visible widgets, not background reCAPTCHA telemetry/hidden anchors.
  const widget = frames.some(f => f.visible && /challenges\.cloudflare\.com|\/turnstile\/|(?:google\.com|recaptcha\.net)\/recaptcha\/|hcaptcha\.com/i.test(f.url));
  if (challengeText || widget) return {kind:'human_check',message:`${site} requires a human check. Complete the verification in that tab, then press Continue. Rapid will verify that the check is gone.`};
  return null;
}
