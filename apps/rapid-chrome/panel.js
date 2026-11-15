import { approvalTitleKey } from './src/safety.js';
import { ScrollFollow, acceptsSnapshot, activityLabels, quietSteps, errorType } from './src/ui-state.js';
import { markdown, stepLabel } from './src/ui.js';
import { renderSteps, clip } from './src/transcript.js';
import { t, localize, translateDocument, displayResult, toolLabel } from './src/i18n.js';
import { endpoint, originPermission, LOCAL_ENDPOINTS, redact } from './src/endpoint.js';
translateDocument();
const $ = id => document.getElementById(id);
let port, state, initialized = false, editing = false;
let connectionBusy = false, messageSignature = '', unread = 0, contextTab = null, contextAdded = false;
const thread = $('transcript');
const follow = new ScrollFollow();
const selected = () => !!document.getSelection?.()?.toString();
const nearBottom = () => follow.follow && !selected() && thread.scrollHeight - thread.scrollTop - thread.clientHeight <= 40;
function scrollBottom() { thread.scrollTop = thread.scrollHeight; unread = 0; follow.bottom(); $('new-messages').hidden = true; }
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
  $('endpoint-mode').value = s.settings.endpointMode || 'local'; $('page-detail').value = s.settings.pageDetail || 'auto'; $('reasoning').value = s.settings.reasoningEffort || 'off';
  $('base').value = s.settings.base; $('api-key').value = s.settings.apiKey;
  document.body.dataset.theme = s.settings.theme || 'dark'; $('theme').value = s.settings.theme || 'dark';
  $('vision').checked = s.settings.vision; $('step-cap').value = s.settings.stepCap;
  $('model').replaceChildren(...[...new Set([s.settings.model,...s.models])].filter(Boolean).map(id => {const o = element('option',id); o.value = id; return o;}));
  if (!$('model').options.length) { const o = element('option',t('firstModel')); o.value = ''; $('model').append(o); }
  $('model').value = s.settings.model;
  $('quick-model').replaceChildren(...s.models.map(id => { const option = element('option',id); option.value = id; return option; }));
  $('quick-model').value = s.settings.model; $('quick-model').disabled = s.running || !s.models.length;
  updateEndpoint();
}
function render(s, force = false) {
  if (!force && !acceptsSnapshot(state,s)) return;
  if (!initialized || thread.scrollHeight - thread.scrollTop - thread.clientHeight > 40 || selected()) follow.scroll(thread,selected());
  const wasBottom = nearBottom();
  if (!wasBottom) follow.follow = false;
  state = s;
  $('settings-toggle').disabled = false;
  document.body.dataset.ready = 'true';
  if (!initialized || !editing) { populate(s); initialized = true; }
  if ($('status').textContent !== localize(s.status)) $('status').textContent = localize(s.status);
  $('settings-status').textContent = localize(s.status);
  document.body.dataset.connected = String(!!s.connected);
  $('availability').textContent = t(s.connected ? 'engineReady' : 'notConnected');
  $('browser-state').textContent = t(s.status === 'Stopped' || s.status === t('stoppedStatus') ? 'stoppedStatus' : s.paused ? 'browserPaused' : s.running && s.tabs.some(tab => tab.attached) ? 'browserWorking' : s.tabs.length ? 'browserGranted' : 'browserIdle');
  $('model-name').textContent = s.settings.model || '';
  $('model-name').title = s.settings.model || '';
  $('setup-hint').hidden = !!s.connected;
  $('onboard-connect').hidden = !!s.connected;
  $('onboarding').hidden = s.messages.some(m => m.role === 'user');
  connectionBusy = s.status === t('connecting');
  $('chat').setAttribute('aria-busy',String(s.running));
  $('settings-form').setAttribute('aria-busy',String(connectionBusy));
  $('onboard-connect').textContent = t(s.connected ? 'stepConnected' : 'stepConnect');
  $('open-tab').disabled = s.running; $('settings-stop').hidden = !s.running;
  $('stop').disabled = !s.running; $('stop').hidden = !s.running; $('send').hidden = !!s.running; $('send').disabled = s.running || !s.connected;
  const granted = s.tabs.some(tab => tab.id === contextTab?.id);
  $('share').textContent = t(granted ? 'tabGranted' : 'grantTab');
  $('share').disabled = !contextTab || (s.running && !granted);
  $('add-context').disabled = !contextTab;
  $('history-toggle').disabled = !!s.running;
  for (const field of $('settings-form').elements) field.disabled = s.running || connectionBusy;
  $('new-chat').disabled = s.running; $('menu-new-chat').disabled = s.running; $('handover-actions').hidden = !s.running || s.paused; $('menu-takeover').hidden = !s.running || s.paused;
  const signature = JSON.stringify(s.messages);
  const changed = signature !== messageSignature;
  if (changed && !selected()) {
    const oldMessages = [...($('messages').children || [])];
    messageSignature = signature;
    $('messages').replaceChildren(...s.messages.slice(-60).flatMap((m,index) => {
      if (m.kind === 'notice' && /^Shared tab:/.test(m.content)) return [];
      const key = m.id || `${m.at || ''}:${m.role}:${index}`;
      const existing = oldMessages.find(node => node.dataset?.key === key && node.dataset?.content === m.content);
      if (existing) { const timestamp = oldMessages.find(node => node.dataset?.timeFor === key); return timestamp ? [timestamp,existing] : [existing]; }
      const article = element('article',undefined,`bubble ${m.role}${m.kind === 'notice' ? ' system-note' : ''}`);
      article.dataset.key = key; article.dataset.content = m.content;
      if (m.role === 'assistant') markdown(document,article,clip(['notice'].includes(m.kind) ? localize(m.content) : m.content,12000),{t,copy:text => navigator.clipboard.writeText(text),copied:copiedControl});
      else article.append(element('p',clip(m.content,12000)));
      let time;
      if (m.at && m.role === 'user') { time = element('time',new Date(m.at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),'timestamp'); time.dateTime = m.at; time.dataset.timeFor = key; }
      const actions = element('div',undefined,'message-actions');
      const copyButton = button(t('copyMessage'),async () => { try { await navigator.clipboard.writeText(m.content); copiedControl(copyButton); } catch { $('status').textContent = t('traceFailed'); } });
      actions.append(copyButton);
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
  const elapsed = Math.max(0,Math.round(((s.run?.endedAt || Date.now()) - (s.run?.startedAt || Date.now())) / 1000));
  $('activity-summary').textContent = activityLabels(s.steps,t,s.running,elapsed) + (s.running ? ` · ${t('elapsedTime',[String(elapsed)])}` : '');
  $('activity-summary').title = activityLabels(s.steps,t,true,elapsed);
  const current = s.steps.findLast(step => step.status === 'running');
  $('current-step').textContent = s.running && !s.paused ? (current ? stepLabel(current,t,toolLabel) : t('thinking')) : '';
  renderSteps(document,$('steps'),quietSteps(s.steps),{label:toolLabel,result:displayResult,status:localize,older:n => t('olderSteps',[String(n)]),stepLabel:step => stepLabel(step,t,toolLabel)});
  $('history-warning').textContent = s.historyWarning || '';
  $('history-warning').hidden = !s.historyWarning;
  $('history-list').replaceChildren(...(s.chats || []).map(chat => button(chat.title,() => { send({type:'restore-chat',id:chat.id}); follow.seen.clear(); scrollBottom(); $('history').hidden = true; $('history-toggle').setAttribute('aria-expanded','false'); $('message').focus(); })));
  if (!(s.chats || []).length) $('history-list').append(element('p',t('noHistory')));
  // Preserve approval buttons/focus while unrelated step updates arrive.
  const approvalFocused = $('approvals').contains?.(document.activeElement);
  const ids = s.approvals.map(a => a.id).join(',');
  if ($('approvals').dataset.ids !== ids) {
    $('approvals').dataset.ids = ids;
    $('approvals').replaceChildren(...s.approvals.map(a => {
      const card = element('article',undefined,'approval'); card.dataset.id = a.id;
      card.append(element('p',t('approvalEyebrow'),'eyebrow'),element('h2',t(a.titleKey || approvalTitleKey(a.target))),element('p',localize(a.action)),element('small',a.site));
      const allow = button(t('allowOnce'),() => send({type:'approve',id:a.id,allow:true})); allow.className = 'primary';
      card.append(allow,button(t('deny'),() => send({type:'approve',id:a.id,allow:false}))); card.append(element('p',a.effect ? t('approvalEffect',[localize(a.effect)]) : t(a.kind === 'grant' ? 'grantEffect' : 'actionEffect'),'approval-effect')); return card;
    }));
  }
  if (approvalFocused && !s.approvals.length) $('message').focus();
  if (!s.running && document.activeElement === $('stop')) $('message').focus();
  $('paused').hidden = !s.paused;
  $('pause-title').textContent = s.handover ? t('handoverTitle',[s.handover.reason || t(({login:'loginHandover',payment:'paymentHandover',permission:'permissionHandover',human_check:'checkHandover',load_error:'loadHandover'})[s.handover.kind] || 'checkHandover')]) : t(s.pauseReason === 'takeover' ? 'takeoverTitle' : 'challengeTitle');
  $('pause-help').textContent = s.handover ? [s.handover.message, t('handoverHelp')].filter(Boolean).join(' ') : t(s.pauseReason === 'takeover' ? 'takeoverHelp' : 'challengeHelp');
  $('resume').textContent = t(s.handover ? 'handoverDone' : s.pauseReason === 'takeover' ? 'resumeTask' : 'continueTask');
  const lastAnswer = s.messages.findLast(m => m.role === 'assistant');
  const incomplete = !s.running && /completion is unconfirmed|model did not inspect|no final answer|step cap/i.test(lastAnswer?.content || '');
  const failed = !s.running && (s.status === 'Error' || s.status === t('errorStatus') || incomplete || !!s.error || !!s.run?.interrupted || (!s.connected && s.messages.length));
  $('recovery').hidden = !failed;
  const type = s.error?.type || errorType(lastAnswer?.content || s.status);
  $('recovery-help').textContent = t(s.run?.interrupted ? 'interruptedRun' : ({unreachable:'errorUnreachable',cors:'errorCors',noTools:'errorNoTools',stepCap:'errorStepCap'})[type] || (incomplete ? 'incompleteHelp' : 'connectionHelp'));
  $('recovery-command').hidden = !['cors','unreachable'].includes(type);
  const model = s.settings.model || 'qwen3.5-4b-4bit';
  const extensionId = chrome.runtime.id || '';
  $('recovery-command').textContent = type === 'cors' ? `rapid-mlx serve ${model} --cors-origins chrome-extension://${extensionId}` : `rapid-mlx serve ${model}`;
  $('recover').textContent = t(incomplete || s.run?.interrupted ? 'reviseTask' : 'reviewSettings');
  $('recover').onclick = () => { if (incomplete || s.run?.interrupted) { $('message').value = s.messages.findLast(m => m.role === 'user')?.content || ''; grow(); $('message').focus(); } else openSettings(); };
  if (!s.messages.length) scrollBottom();
  const shouldFollow = follow.update(s.messages,selected());
  unread = follow.unread;
  $('new-messages').textContent = t(unread === 1 ? 'oneNewMessage' : 'newMessages',[String(unread)]);
  $('new-messages').hidden = !unread;
  if (wasBottom && shouldFollow) scrollBottom();
}
function openSettings() {
  const hidden = !$('settings').hidden; $('settings').hidden = hidden;
  $('settings-toggle').setAttribute('aria-expanded',String(!hidden));
  if (!hidden && state) populate(state); editing = !hidden;
  thread.inert = !hidden; document.querySelector('footer').inert = !hidden; document.querySelector('.thread-head').inert = !hidden;
  if (!hidden) $('settings').focus(); else $('settings-toggle').focus();
}
function closeMenu() { $('panel-menu').hidden = true; $('settings-toggle').setAttribute('aria-expanded','false'); }
$('settings-toggle').onclick = () => {
  if (!$('settings').hidden) { openSettings(); return; }
  $('panel-menu').hidden = !$('panel-menu').hidden;
  $('settings-toggle').setAttribute('aria-expanded',String(!$('panel-menu').hidden));
  if (!$('panel-menu').hidden) $('menu-settings').focus();
};
$('menu-settings').onclick = () => { closeMenu(); openSettings(); };
$('menu-new-chat').onclick = () => { closeMenu(); $('new-chat').onclick(); };
$('menu-help').onclick = () => { closeMenu(); openSettings(); $('help-text').hidden = false; };
$('panel-menu').onkeydown = event => { if (event.key === 'Escape') { closeMenu(); $('settings-toggle').focus(); } };
$('menu-takeover').onclick = () => { closeMenu(); send({type:'take-over'}); };
for (const kind of ['login','payment','permission']) $('handover-'+kind).onclick = () => { closeMenu(); send({type:'handover',kind}); };
$('quick-model').onchange = () => { if (state && !state.running) send({type:'settings',settings:{...state.settings,model:$('quick-model').value}}); };
$('theme').onchange = () => { document.body.dataset.theme = $('theme').value; };

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
    const settings = {pageDetail:$('page-detail').value,theme:$('theme').value,endpointMode:$('endpoint-mode').value,reasoningEffort:$('reasoning').value,base,apiKey:$('api-key').value,model:$('model').value,vision:$('vision').checked,stepCap:$('step-cap').value};
    // These HTTP loopback hosts are granted by the manifest across all ports.
    // Request other origins in the click gesture, before messaging the worker.
    const bundled = /^http:\/\/(?:127\.0\.0\.1|localhost)(?::\d+)?$/.test(base);
    const allowed = bundled || await chrome.permissions.request({origins:[originPermission(base)]});
    if (!allowed) throw new Error('permissionMissing');
    editing = false;
    $('status').textContent = t('connecting');
    document.body.dataset.connected = 'false';
    send({type:'settings',settings});
  } catch (e) { $('status').textContent = localize(e.message); }
};
$('refresh').onclick = () => { editing = false; send({type:'connect'}); };
$('onboard-connect').onclick = () => { if ($('settings').hidden) openSettings(); $('endpoint-mode').focus(); };
$('open-tab').onclick = () => send({type:'open-tab'});
$('share').onclick = () => { const granted = state?.tabs.some(tab => tab.id === contextTab?.id); send(granted ? {type:'revoke',id:contextTab.id} : {type:'share'}); };
$('stop').onclick = $('pause-stop').onclick = $('settings-stop').onclick = () => send({type:'stop'});
$('resume').onclick = () => send({type:'resume',allow:true});
$('chat').onsubmit = e => { e.preventDefault(); const text = $('message').value.trim(); if (!text || state?.running || !state?.connected) return; send({type:'send',tabId:contextTab?.id,text:contextAdded && contextTab ? `${text}\n\n${t('pageContext')}: ${contextTab.title}\n${contextTab.url}` : text}); $('message').value = ''; contextAdded = false; $('add-context').setAttribute('aria-pressed','false'); $('context-hint').textContent = ''; grow(); scrollBottom(); };
$('message').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); $('chat').requestSubmit(); } };
$('new-chat').onclick = () => $('new-chat-choice').showModal();
function startNewChat(deleteHistory) {
  $('new-chat-choice').close(); send({type:'new-chat',deleteHistory});
  follow.seen.clear(); scrollBottom(); $('message').focus();
}
$('archive-chat').onclick = () => startNewChat(false);
$('delete-chats').onclick = () => startNewChat(true);
$('cancel-new-chat').onclick = () => $('new-chat-choice').close();
$('copy-trace').onclick = async () => {
  try {
    const {debugTrace} = await chrome.storage.session.get('debugTrace');
    await navigator.clipboard.writeText(JSON.stringify(redact(debugTrace ?? {},state?.settings.apiKey),null,2));
    $('status').textContent = t('traceCopied');
  } catch (e) { $('status').textContent = t('traceFailed'); }
};
for (const kind of ['shopping','reservations','summarize','formTask']) {
  const chip = button(t(kind),() => { $('message').value = t(`${kind}Example`) + (contextTab ? `\n\n${t('pageContext')}: ${contextTab.title}\n${contextTab.url}` : ''); grow(); $('message').focus(); });
  chip.dataset.kind = kind; $('examples').append(chip);
}
$('message').oninput = grow;
$('new-messages').onclick = scrollBottom;
thread.onscroll = () => { follow.scroll(thread,selected()); if (follow.follow) { unread = 0; $('new-messages').hidden = true; } };
thread.onwheel = e => { if (e.deltaY < 0) follow.follow = false; };
thread.onpointerdown = () => { follow.follow = false; };
document.onselectionchange = () => { if (selected()) follow.follow = false; else if (state) { messageSignature = ''; render(state,true); } };
$('history-toggle').onclick = () => { $('history').hidden = !$('history').hidden; $('history-toggle').setAttribute('aria-expanded',String(!$('history').hidden)); };
$('help').onclick = () => { $('help-text').hidden = !$('help-text').hidden; };
$('add-context').onclick = () => { contextAdded = !contextAdded; $('context-hint').textContent = contextAdded ? t('contextAdded') : ''; $('add-context').setAttribute('aria-pressed',String(contextAdded)); $('message').focus(); };
async function activeTab() {
  try {
    const [tab] = await chrome.tabs.query({active:true,lastFocusedWindow:true});
    contextTab = tab && /^https?:/.test(tab.url) ? tab : null;
    $('tab-title').textContent = contextTab ? tab.title || new URL(tab.url).hostname : t('noTab');
    $('tab-title').title = contextTab?.url || '';
    const starter = [...$('examples').children].find(chip => chip.dataset.kind === 'summarize');
    if (starter) starter.textContent = contextTab ? t('pageStarter',[clip(contextTab.title || new URL(tab.url).hostname,50)]) : t('summarize');
    const summaryStarter = $('examples').children?.[2];
    if (summaryStarter) summaryStarter.textContent = contextTab ? t('pageStarter',[(contextTab.title || new URL(contextTab.url).hostname).slice(0,40)]) : t('summarize');
    // Keep the site initial until Chrome supplies its cached favicon.
    $('tab-icon').textContent = contextTab ? new URL(tab.url).hostname[0].toUpperCase() : '';
    if (contextTab && chrome.runtime.getURL) {
      const favicon = element('img'); favicon.alt = ''; favicon.width = favicon.height = 16;
      favicon.src = chrome.runtime.getURL(`_favicon/?pageUrl=${encodeURIComponent(tab.url)}&size=32`);
      favicon.onload = () => { if (contextTab?.id === tab.id && contextTab?.url === tab.url) $('tab-icon').replaceChildren(favicon); };
    }
    if (state) render(state,true);
  } catch { /* Chrome internal pages cannot be shared. */ }
}
chrome.tabs?.onActivated?.addListener(activeTab);
chrome.tabs?.onUpdated?.addListener(activeTab);
void activeTab();
setUnavailable(true);
connect();

function copiedControl(control) {
  const original = control.textContent; control.textContent = t('copied');
  control.setAttribute('aria-label',t('messageCopied'));
  setTimeout(() => { control.textContent = original; control.setAttribute('aria-label',original); },1600);
}
globalThis.setInterval?.(() => {
  if (state?.running && state.run?.startedAt) {
    const seconds = Math.max(0,Math.floor((Date.now()-state.run.startedAt)/1000));
    $('activity-summary').textContent = `${activityLabels(state.steps,t,true,seconds)} · ${t('elapsedTime',[String(seconds)])}`;
  }
},1000);
export { render };
