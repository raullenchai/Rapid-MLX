import { markdown, stepLabel } from './src/ui.js';
import { renderSteps, clip } from './src/transcript.js';
import { t, localize, translateDocument, displayResult, toolLabel } from './src/i18n.js';
import { endpoint, originPermission, LOCAL_ENDPOINTS, redact } from './src/endpoint.js';
translateDocument();
const $ = id => document.getElementById(id);
let port, state, initialized = false, editing = false;
let connectionBusy = false, messageSignature = '', unread = 0, contextTab = null, contextAdded = false;
const thread = $('transcript');
const nearBottom = () => thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
function scrollBottom() { thread.scrollTop = thread.scrollHeight; unread = 0; $('new-messages').hidden = true; }
function grow() { $('message').style.height = 'auto'; $('message').style.height = Math.min(176,$('message').scrollHeight) + 'px'; }

function send(message) { port?.postMessage(message); }
function connect() {
  port = chrome.runtime.connect({name:'rapid-panel'});
  port.onMessage.addListener(render);
  port.onDisconnect.addListener(() => { $('status').textContent = t('reconnecting'); setUnavailable(true); setTimeout(connect,500); });
}
function element(tag,text,className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
function button(text,fn) { const b = element('button',text); b.type = 'button'; b.onclick = fn; return b; }
function populate(s) {
  $('endpoint-mode').value = s.settings.endpointMode || 'local'; $('reasoning').value = s.settings.reasoningEffort || 'off';
  $('base').value = s.settings.base; $('api-key').value = s.settings.apiKey;
  $('vision').checked = s.settings.vision; $('step-cap').value = s.settings.stepCap;
  $('model').replaceChildren(...[...new Set([s.settings.model,...s.models])].filter(Boolean).map(id => {const o = element('option',id); o.value = id; return o;}));
  if (!$('model').options.length) { const o = element('option',t('firstModel')); o.value = ''; $('model').append(o); }
  $('model').value = s.settings.model; updateEndpoint();
}
function render(s) {
  const wasBottom = nearBottom();
  const previous = state;
  state = s;
  $('settings-toggle').disabled = false;
  document.body.dataset.ready = 'true';
  if (!initialized || !editing) { populate(s); initialized = true; }
  $('status').textContent = localize(s.status);
  $('settings-status').textContent = localize(s.status);
  document.body.dataset.connected = String(!!s.connected);
  $('availability').textContent = t(s.connected ? 'browserAvailable' : 'notConnected');
  $('model-name').textContent = s.settings.model || '';
  $('model-name').title = s.settings.model || '';
  $('setup-hint').hidden = !!s.connected;
  $('onboard-connect').hidden = !!s.connected;
  $('onboarding').hidden = s.messages.some(m => m.role === 'user');
  connectionBusy = s.status === t('connecting');
  $('chat').setAttribute('aria-busy',String(s.running));
  $('settings-form').setAttribute('aria-busy',String(connectionBusy));
  $('onboard-connect').textContent = t(s.connected ? 'stepConnected' : 'stepConnect');
  $('open-tab').disabled = s.running;
  $('stop').disabled = !s.running; $('stop').hidden = !s.running; $('send').hidden = !!s.running; $('send').disabled = s.running || !s.connected;
  const granted = s.tabs.some(tab => tab.id === contextTab?.id);
  $('share').textContent = t(granted ? 'tabGranted' : 'grantTab');
  $('share').disabled = !contextTab || (s.running && !granted);
  $('add-context').disabled = !contextTab;
  $('history-toggle').disabled = !!s.running;
  for (const field of $('settings-form').elements) field.disabled = s.running || connectionBusy;
  $('new-chat').disabled = s.running;
  const signature = JSON.stringify(s.messages);
  const changed = signature !== messageSignature;
  if (changed) {
    messageSignature = signature;
    $('messages').replaceChildren(...s.messages.slice(-60).flatMap(m => {
      const article = element('article',undefined,`bubble ${m.role}`);
      if (m.role === 'assistant') markdown(document,article,clip(['notice'].includes(m.kind) ? localize(m.content) : m.content,12000));
      else article.append(element('p',clip(m.content,12000)));
      let time;
      if (m.at && m.role === 'user') { time = element('time',new Date(m.at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),'timestamp'); time.dateTime = m.at; }
      const actions = element('div',undefined,'message-actions');
      actions.append(button(t('copyMessage'),async () => { try { await navigator.clipboard.writeText(m.content); $('status').textContent = t('messageCopied'); } catch { $('status').textContent = t('traceFailed'); } }));
      const more = element('details',undefined,'message-more');
      more.append(element('summary',t('moreActions')));
      more.append(button(t('useAsPrompt'),() => { $('message').value = m.content; grow(); $('message').focus(); more.open = false; }));
      actions.append(more);
      article.append(actions); return time ? [time,article] : [article];
    }));
  }
  $('owned').hidden = s.tabs.length <= 1;
  $('owned').replaceChildren(...s.tabs.map(tab => {
    const row = element('div',`${tab.id} · ${localize(tab.attached ? 'attached' : 'shared')} `);
    const b = button(t('release'),() => send({type:s.running ? 'take-over' : 'release',id:tab.id})); row.append(b); return row;
  }));
  $('work-card').hidden = !s.steps.length && !s.running;
  $('activity-summary').textContent = t('activitySummary',[String(new Set(s.steps.map(step => { try { return new URL(step.args?.url || step.url).hostname; } catch { return null; } }).filter(Boolean)).size),String(s.steps.length)]);
  const current = s.steps.findLast(step => step.status === 'running');
  $('current-step').textContent = s.running && !s.paused ? (current ? stepLabel(current,t,toolLabel) : t('thinking')) : '';
  renderSteps(document,$('steps'),s.steps,{label:toolLabel,result:displayResult,status:localize,older:n => t('olderSteps',[String(n)]),stepLabel:step => stepLabel(step,t,toolLabel)});
  $('history-warning').textContent = s.historyWarning || '';
  $('history-warning').hidden = !s.historyWarning;
  $('history-list').replaceChildren(...(s.chats || []).map(chat => button(chat.title,() => { send({type:'restore-chat',id:chat.id}); $('history').hidden = true; $('history-toggle').setAttribute('aria-expanded','false'); $('message').focus(); })));
  if (!(s.chats || []).length) $('history-list').append(element('p',t('noHistory')));
  // Preserve approval buttons/focus while unrelated step updates arrive.
  const ids = s.approvals.map(a => a.id).join(',');
  if ($('approvals').dataset.ids !== ids) {
    $('approvals').dataset.ids = ids;
    $('approvals').replaceChildren(...s.approvals.map(a => {
      const card = element('article',undefined,'approval'); card.dataset.id = a.id;
      card.append(element('p',t('approvalEyebrow'),'eyebrow'),element('h2',t(/order|buy|purchase|checkout/i.test(a.action) ? 'allowOrder' : 'allowAction')),element('p',localize(a.action)),element('small',a.site));
      const allow = button(t('allowOnce'),() => send({type:'approve',id:a.id,allow:true})); allow.className = 'primary';
      card.append(allow,button(t('deny'),() => send({type:'approve',id:a.id,allow:false}))); return card;
    }));
  }
  $('paused').hidden = !s.paused;
  $('pause-title').textContent = t(s.pauseReason === 'takeover' ? 'takeoverTitle' : 'challengeTitle');
  $('pause-help').textContent = t(s.pauseReason === 'takeover' ? 'takeoverHelp' : 'challengeHelp');
  $('resume').textContent = t(s.pauseReason === 'takeover' ? 'resumeTask' : 'continueTask');
  const lastAnswer = s.messages.findLast(m => m.role === 'assistant');
  const incomplete = !s.running && /completion is unconfirmed|model did not inspect|no final answer|step cap/i.test(lastAnswer?.content || '');
  const failed = !s.running && (s.status === 'Error' || s.status === t('errorStatus') || incomplete || (!s.connected && s.messages.length));
  $('recovery').hidden = !failed;
  $('recovery-help').textContent = t(incomplete ? 'incompleteHelp' : 'connectionHelp');
  $('recover').textContent = t(incomplete ? 'reviseTask' : 'reviewSettings');
  $('recover').onclick = () => { if (incomplete) { $('message').value = s.messages.findLast(m => m.role === 'user')?.content || ''; grow(); $('message').focus(); } else openSettings(); };
  $('result-tab').hidden = s.running || !s.resultTab;
  if (!s.messages.length) scrollBottom();
  if (s.messages.length && changed && previous && !wasBottom) { unread += Math.max(1,s.messages.length - previous.messages.length); $('new-messages').textContent = t('newMessages',[String(unread)]); $('new-messages').hidden = false; }
  if (wasBottom) scrollBottom();
}
function openSettings() {
  const hidden = !$('settings').hidden; $('settings').hidden = hidden;
  $('settings-toggle').setAttribute('aria-expanded',String(!hidden));
  if (!hidden && state) populate(state); editing = !hidden;
  thread.inert = !hidden; document.querySelector('footer').inert = !hidden; document.querySelector('.thread-head').inert = !hidden;
  if (!hidden) $('settings').focus(); else $('settings-toggle').focus();
}
$('settings-toggle').onclick = openSettings;
$('settings-close').onclick = () => { if (!$('settings').hidden) openSettings(); };
$('settings').onkeydown = e => {
  if (e.key === 'Escape') { e.preventDefault(); openSettings(); }
  if (e.key === 'Tab') {
    const controls = [...$('settings').querySelectorAll('button:not(:disabled),input:not(:disabled),select:not(:disabled)')].filter(node => !node.hidden && node.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (e.shiftKey && (document.activeElement === first || document.activeElement === $('settings'))) { e.preventDefault(); last?.focus(); }
    else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first?.focus(); }
  }
};
function updateEndpoint() {
  const local = $('endpoint-mode').value === 'local';
  $('base').readOnly = local;
  $('http-warning').hidden = !$('base').value.startsWith('http:');
}
function setUnavailable(value) {
  for (const id of ['send','share','open-tab','refresh','settings-toggle']) $(id).disabled = value;
}
$('endpoint-mode').onchange = () => { if ($('endpoint-mode').value === 'local') $('base').value = LOCAL_ENDPOINTS[0]; updateEndpoint(); };
$('base').oninput = updateEndpoint;
$('settings-form').onsubmit = async e => {
  e.preventDefault();
  try {
    const base = endpoint($('base').value);
    const settings = {endpointMode:$('endpoint-mode').value,reasoningEffort:$('reasoning').value,base,apiKey:$('api-key').value,model:$('model').value,vision:$('vision').checked,stepCap:$('step-cap').value};
    // These HTTP loopback hosts are granted by the manifest across all ports.
    // Request other origins in the click gesture, before messaging the worker.
    const bundled = /^http:\/\/(?:127\.0\.0\.1|localhost)(?::\d+)?$/.test(base);
    const allowed = bundled || await chrome.permissions.request({origins:[originPermission(base)]});
    if (!allowed) throw new Error('permissionMissing');
    editing = false;
    $('status').textContent = t('connecting');
    send({type:'settings',settings});
  } catch (e) { $('status').textContent = localize(e.message); }
};
$('refresh').onclick = () => { editing = false; send({type:'connect'}); };
$('onboard-connect').onclick = () => { if ($('settings').hidden) $('settings-toggle').onclick(); $('endpoint-mode').focus(); };
$('open-tab').onclick = () => send({type:'open-tab'});
$('share').onclick = () => { const granted = state?.tabs.some(tab => tab.id === contextTab?.id); send(granted ? {type:state.running ? 'take-over' : 'release',id:contextTab.id} : {type:'share'}); };
$('stop').onclick = $('pause-stop').onclick = () => send({type:'stop'});
$('resume').onclick = () => send({type:'resume',allow:true});
$('chat').onsubmit = e => { e.preventDefault(); const text = $('message').value.trim(); if (!text || state?.running || !state?.connected) return; send({type:'send',text:contextAdded && contextTab ? `${text}\n\n${t('pageContext')}: ${contextTab.title}\n${contextTab.url}` : text}); $('message').value = ''; contextAdded = false; $('add-context').setAttribute('aria-pressed','false'); $('context-hint').textContent = ''; grow(); scrollBottom(); };
$('message').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('chat').requestSubmit(); } };
$('new-chat').onclick = () => send({type:'new-chat'});
$('copy-trace').onclick = async () => {
  try {
    const {debugTrace} = await chrome.storage.session.get('debugTrace');
    await navigator.clipboard.writeText(JSON.stringify(redact(debugTrace ?? {},state?.settings.apiKey),null,2));
    $('status').textContent = t('traceCopied');
  } catch (e) { $('status').textContent = t('traceFailed'); }
};
for (const kind of ['shopping','reservations','summarize','formTask']) {
  const chip = button(t(kind),() => { $('message').value = t(`${kind}Example`); grow(); $('message').focus(); });
  $('examples').append(chip);
}
$('message').oninput = grow;
$('new-messages').onclick = scrollBottom;
thread.onscroll = () => { if (nearBottom()) { unread = 0; $('new-messages').hidden = true; } };
$('history-toggle').onclick = () => { $('history').hidden = !$('history').hidden; $('history-toggle').setAttribute('aria-expanded',String(!$('history').hidden)); };
$('help').onclick = () => { $('help-text').hidden = !$('help-text').hidden; };
$('result-tab').onclick = () => send({type:'result-tab'});
$('add-context').onclick = () => { contextAdded = !contextAdded; $('context-hint').textContent = contextAdded ? t('contextAdded') : ''; $('add-context').setAttribute('aria-pressed',String(contextAdded)); $('message').focus(); };
async function activeTab() {
  try {
    const [tab] = await chrome.tabs.query({active:true,lastFocusedWindow:true});
    contextTab = tab && /^https?:/.test(tab.url) ? tab : null;
    $('tab-title').textContent = contextTab ? tab.title || new URL(tab.url).hostname : t('noTab');
    $('tab-title').title = contextTab?.url || '';
    // Use a local site initial without requesting a remote favicon.
    $('tab-icon').textContent = contextTab ? new URL(tab.url).hostname[0].toUpperCase() : '▤';
    if (state) render(state);
  } catch { /* Chrome internal pages cannot be shared. */ }
}
chrome.tabs?.onActivated?.addListener(activeTab);
chrome.tabs?.onUpdated?.addListener(activeTab);
void activeTab();
setUnavailable(true);
connect();

export { render };
