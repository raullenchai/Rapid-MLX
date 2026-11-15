import { t, localize, translateDocument, displayResult, toolLabel } from './src/i18n.js';
import { endpoint, originPermission, LOCAL_ENDPOINTS, redact } from './src/endpoint.js';
translateDocument();
const $ = id => document.getElementById(id);
let port, state, initialized = false, editing = false;
let connectionBusy = false;
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
  state = s;
  $('settings-toggle').disabled = false;
  document.body.dataset.ready = 'true';
  if (!initialized || !editing) { populate(s); initialized = true; }
  $('status').textContent = localize(s.status);
  $('onboarding').hidden = s.messages.some(m => m.role === 'user');
  connectionBusy = s.status === t('connecting');
  $('chat').setAttribute('aria-busy',String(s.running));
  $('settings-form').setAttribute('aria-busy',String(connectionBusy));
  $('onboard-connect').textContent = t(s.connected ? 'stepConnected' : 'stepConnect');
  $('open-tab').disabled = s.running;
  $('stop').disabled = !s.running; $('send').disabled = s.running || !s.connected; $('share').disabled = s.running;
  for (const field of $('settings-form').elements) field.disabled = s.running || connectionBusy;
  $('new-chat').disabled = s.running;
  $('messages').replaceChildren(...s.messages.map(m => element('article',m.role === 'assistant' ? localize(m.content) : m.content,`bubble ${m.role}`)));
  $('owned').replaceChildren(...s.tabs.map(t => { const row = element('div',`${t.id} · ${t.attached ? localize('attached') : localize('shared')} `); const b = button(localize('release'),() => send({type:'release',id:t.id})); b.disabled = s.running; row.append(b); return row; }));
  const open = new Set([...$('steps').querySelectorAll('details[open]')].map(e => e.dataset.id));
  $('steps').replaceChildren(...s.steps.map(step => {
    const d = element('details'); d.dataset.id = step.id; d.open = open.has(step.id);
    d.append(element('summary',`${toolLabel(step.name)} · ${localize(step.status)} · ${JSON.stringify(step.args)} · ${displayResult(step.result).slice(0,200)}`),element('pre',JSON.stringify(step.args,null,2)),element('p',displayResult(step.result))); return d;
  }));
  // Preserve approval buttons/focus while unrelated step updates arrive.
  const ids = s.approvals.map(a => a.id).join(',');
  if ($('approvals').dataset.ids !== ids) {
    $('approvals').dataset.ids = ids;
    $('approvals').replaceChildren(...s.approvals.map(a => {
      const card = element('article',undefined,'approval'); card.dataset.id = a.id;
      card.append(element('h2',localize(a.action)),element('small',a.site),button(t('allowOnce'),() => send({type:'approve',id:a.id,allow:true})),button(t('deny'),() => send({type:'approve',id:a.id,allow:false}))); return card;
    }));
  }
  $('paused').hidden = !s.paused;
}
$('settings-toggle').onclick = () => { const hidden = !$('settings').hidden; $('settings').hidden = hidden; $('settings-toggle').setAttribute('aria-expanded',String(!hidden)); if (!hidden && state) populate(state); editing = !hidden; };
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
    // Request in the click gesture, before messaging the service worker.
    const allowed = await chrome.permissions.request({origins:[originPermission(base)]});
    if (!allowed) throw new Error('permissionMissing');
    editing = false;
    $('status').textContent = t('connecting');
    send({type:'settings',settings});
  } catch (e) { $('status').textContent = localize(e.message); }
};
$('refresh').onclick = () => { editing = false; send({type:'connect'}); };
$('onboard-connect').onclick = () => { if ($('settings').hidden) $('settings-toggle').onclick(); $('endpoint-mode').focus(); };
$('open-tab').onclick = () => send({type:'open-tab'});
$('share').onclick = () => send({type:'share'});
$('stop').onclick = $('pause-stop').onclick = () => send({type:'stop'});
$('resume').onclick = () => send({type:'resume',allow:true});
$('chat').onsubmit = e => { e.preventDefault(); const text = $('message').value.trim(); if (!text || state?.running || !state?.connected) return; send({type:'send',text}); $('message').value = ''; };
$('message').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('chat').requestSubmit(); } };
$('new-chat').onclick = () => send({type:'new-chat'});
$('copy-trace').onclick = async () => {
  try {
    const {debugTrace} = await chrome.storage.session.get('debugTrace');
    await navigator.clipboard.writeText(JSON.stringify(redact(debugTrace ?? {},state?.settings.apiKey),null,2));
    $('status').textContent = t('traceCopied');
  } catch (e) { $('status').textContent = t('traceFailed'); }
};
for (const kind of ['shopping','travel','reservations','mail']) {
  const chip = button(t(kind),() => { $('message').value = t(`${kind}Example`); $('message').focus(); });
  $('examples').append(chip);
}
setUnavailable(true);
connect();
