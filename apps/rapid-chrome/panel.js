const $ = id => document.getElementById(id);
let port, state, initialized = false, editing = false;
function send(message) { port?.postMessage(message); }
function connect() {
  port = chrome.runtime.connect({name:'rapid-panel'});
  port.onMessage.addListener(render);
  port.onDisconnect.addListener(() => { $('status').textContent = 'Reconnecting…'; setTimeout(connect,500); });
}
function element(tag,text,className) { const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (className) e.className = className; return e; }
function button(text,fn) { const b = element('button',text); b.type = 'button'; b.onclick = fn; return b; }
function populate(s) {
  $('base').value = s.settings.base; $('api-key').value = s.settings.apiKey;
  $('vision').checked = s.settings.vision; $('step-cap').value = s.settings.stepCap;
  $('model').replaceChildren(...[...new Set([s.settings.model,...s.models])].filter(Boolean).map(id => {const o = element('option',id); o.value = id; return o;}));
  if (!$('model').options.length) { const o = element('option','First available model'); o.value = ''; $('model').append(o); }
  $('model').value = s.settings.model;
}
function render(s) {
  state = s;
  $('settings-toggle').disabled = false;
  document.body.dataset.ready = 'true';
  if (!initialized || !editing) { populate(s); initialized = true; }
  $('status').textContent = s.status;
  $('stop').disabled = !s.running; $('send').disabled = s.running; $('share').disabled = s.running;
  for (const field of $('settings-form').elements) field.disabled = s.running;
  $('new-chat').disabled = s.running;
  $('messages').replaceChildren(...s.messages.map(m => element('article',m.content,`bubble ${m.role}`)));
  $('owned').replaceChildren(...s.tabs.map(t => { const row = element('div',`Tab ${t.id} · ${t.attached ? 'attached' : 'shared'} `); const b = button('Release',() => send({type:'release',id:t.id})); b.disabled = s.running; row.append(b); return row; }));
  const open = new Set([...$('steps').querySelectorAll('details[open]')].map(e => e.dataset.id));
  $('steps').replaceChildren(...s.steps.map(step => {
    const d = element('details'); d.dataset.id = step.id; d.open = open.has(step.id);
    d.append(element('summary',`${step.name} · ${step.status} · ${JSON.stringify(step.args)} · ${(step.result || 'Working…').slice(0,200)}`),element('pre',JSON.stringify(step.args,null,2)),element('p',step.result || 'Working…')); return d;
  }));
  // Preserve approval buttons/focus while unrelated step updates arrive.
  const ids = s.approvals.map(a => a.id).join(',');
  if ($('approvals').dataset.ids !== ids) {
    $('approvals').dataset.ids = ids;
    $('approvals').replaceChildren(...s.approvals.map(a => {
      const card = element('article',undefined,'approval'); card.dataset.id = a.id;
      card.append(element('h2',a.action),element('small',a.site),button('Allow once',() => send({type:'approve',id:a.id,allow:true})),button('Deny',() => send({type:'approve',id:a.id,allow:false}))); return card;
    }));
  }
  $('paused').hidden = !s.paused;
}
$('settings-toggle').onclick = () => { const hidden = !$('settings').hidden; $('settings').hidden = hidden; $('settings-toggle').setAttribute('aria-expanded',String(!hidden)); if (!hidden && state) populate(state); editing = !hidden; };
$('settings-form').onsubmit = e => { e.preventDefault(); editing = false; send({type:'settings',settings:{base:$('base').value,apiKey:$('api-key').value,model:$('model').value,vision:$('vision').checked,stepCap:$('step-cap').value}}); };
$('refresh').onclick = () => { editing = false; send({type:'connect'}); };
$('share').onclick = () => send({type:'share'});
$('stop').onclick = $('pause-stop').onclick = () => send({type:'stop'});
$('resume').onclick = () => send({type:'resume',allow:true});
$('chat').onsubmit = e => { e.preventDefault(); const text = $('message').value.trim(); if (!text || state?.running) return; send({type:'send',text}); $('message').value = ''; };
$('message').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('chat').requestSubmit(); } };
$('new-chat').onclick = () => send({type:'new-chat'});
$('copy-trace').onclick = async () => {
  try {
    const {debugTrace} = await chrome.storage.session.get('debugTrace');
    await navigator.clipboard.writeText(JSON.stringify(debugTrace ?? {},null,2));
    $('status').textContent = 'Debug trace copied';
  } catch (e) { $('status').textContent = `Could not copy trace: ${e.message}`; }
};
connect();
