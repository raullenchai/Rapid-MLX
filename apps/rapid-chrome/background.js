import { BrowserHand, ChromeAdapter } from './src/cdp.js';
import { runAgent, headers } from './src/agent.js';
import { engineBase, safeURL } from './src/safety.js';

const defaults = {base:'http://127.0.0.1:8000',apiKey:'',model:'',vision:false,stepCap:60};
const state = {messages:[],steps:[],approvals:[],running:false,paused:false,status:'Ready',settings:{...defaults},models:[],tabs:[]};
const ports = new Set(), pending = new Map();
let controller = null, runPromise = null, resumeGate = null, pausePromise = null;
const hand = new BrowserHand(new ChromeAdapter(), {settings:state.settings,approve,emit,stop,challenge});
const ready = (async () => {
  const {settings} = await chrome.storage.local.get('settings');
  if (settings) { try { engineBase(settings.base); state.settings = {...defaults,...settings}; } catch {} }
  hand.settings = state.settings;
  const saved = await chrome.storage.session.get(['transcript','owned']);
  state.messages = saved.transcript || [];
  for (const tab of saved.owned || []) {
    try {
      const current = await chrome.tabs.get(tab.id);
      safeURL(current.url,state.settings.base);
      if (tab.created && current.groupId !== tab.groupId) continue;
      hand.tabs.set(tab.id,{...tab,attached:false,refs:new Map(),epoch:0});
      hand.active = tab.id;
    } catch {}
  }
  refreshTabs();
})();
function publish() {
  for (const port of ports) { try { port.postMessage(state); } catch { ports.delete(port); } }
}
function refreshTabs() {
  state.tabs = [...hand.tabs].map(([id,t]) => ({id,attached:t.attached})); publish();
  void chrome.storage.session.set({owned:[...hand.tabs].map(([id,t]) => ({id,created:t.created,groupId:t.groupId,binding:t.binding}))});
}
function emit(event) {
  if (event.kind === 'step') {
    const existing = state.steps.findIndex(s => s.id === event.id);
    if (existing < 0) state.steps.push(event); else state.steps[existing] = event;
  } else if (event.kind === 'answer' || event.kind === 'notice' || event.kind === 'progress') {
    state.messages.push({role:'assistant',kind:event.kind,content:event.text});
    void chrome.storage.session.set({transcript:state.messages.slice(-60)});
  }
  publish();
}
function approve(card) {
  if (!controller || controller.signal.aborted) return Promise.resolve(false);
  return new Promise(resolve => {
    const id = crypto.randomUUID();
    const signal = controller.signal;
    const finish = allowed => {
      signal.removeEventListener('abort',cancel); pending.delete(id);
      state.approvals = state.approvals.filter(a => a.id !== id); publish(); resolve(allowed);
    };
    const cancel = () => finish(false);
    pending.set(id,finish); state.approvals.push({id,...card});
    signal.addEventListener('abort',cancel,{once:true}); publish();
  });
}
async function challenge(info) {
  controller.signal.throwIfAborted();
  if (!state.paused) pausePromise = new Promise(resolve => { resumeGate = resolve; });
  state.paused = true; hand.paused = true;
  state.status = info.message;
  emit({kind:'notice',text:info.message});
  // Release debugger and overlay so the human can interact with the check.
  await hand.finish();
  await checkpoint();
  state.status = 'Working'; publish();
}
function stop() {
  controller?.abort(new DOMException('Stopped by user','AbortError'));
  resumeGate?.(false); resumeGate = null;
  for (const finish of [...pending.values()]) finish(false);
  state.paused = false; hand.paused = false; state.status = 'Stopping…'; publish();
  // Abort prevents new commands; detach also stops outstanding browser operations.
  void hand.finish();
}
async function checkpoint(messages) {
  while (state.paused) {
    const allowed = await pausePromise;
    resumeGate = null;
    if (!allowed) { controller.signal.throwIfAborted(); throw new Error('Stopped'); }
    hand.paused = false;
    for (const [id,t] of hand.tabs) if (!t.attached) { await hand.own(id); await hand.showOverlay(id,true); }
    state.paused = false; publish();
  }
  if (messages && hand.notices.length) messages.push({role:'user',content:`Browser ownership notice (from extension): ${hand.notices.splice(0).join('\n')}`});
}
async function models(signal) {
  const response = await fetch(`${engineBase(state.settings.base)}/v1/models`,{headers:headers(state.settings.apiKey),credentials:'omit',redirect:'error',signal:signal ? AbortSignal.any([signal,AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}; check API key and --cors-origins chrome-extension://${chrome.runtime.id}`);
  const data = await response.json();
  state.models = (data.data || []).map(m => m.id).filter(id => typeof id === 'string');
  if (!state.models.includes(state.settings.model)) state.settings.model = state.models[0] || '';
  if (!state.settings.model) throw new Error('Engine has no loaded models');
  state.status = 'Connected'; await chrome.storage.local.set({settings:state.settings}); publish();
}
async function run(task) {
  if (state.running) throw new Error('A task is already running');
  if (!task.trim()) return;
  hand.paused = false; state.running = true; state.steps = []; state.status = 'Working';
  controller = new AbortController();
  const signal = controller.signal;
  const history = state.messages.filter(m => m.kind !== 'progress').slice(-20);
  const debugTrace = {version:1,id:crypto.randomUUID(),task,startedAt:new Date().toISOString(),events:[]};
  const started = Date.now();
  const trace = async event => {
    debugTrace.events.push({atMs:Date.now()-started,...event});
    await chrome.storage.session.set({debugTrace});
  };
  state.messages.push({role:'user',content:task}); publish();
  // A Chrome API call keeps a waiting approval / local inference run alive.
  const keepAlive = setInterval(() => { void chrome.runtime.getPlatformInfo(); },20000);
  try {
    await trace({kind:'start'});
    if (!state.settings.model) await models(signal);
    signal.throwIfAborted();
    await hand.begin(signal, task, history); refreshTabs();
    await runAgent({settings:{...state.settings},task,history,signal,execute:(...a) => hand.execute(...a),emit,checkpoint,trace});
    await trace({kind:'answer',text:state.messages.at(-1)?.content});
    state.status = 'Ready';
  } catch (e) {
    await trace({kind:'error',error:e.message,aborted:signal.aborted});
    state.status = signal.aborted ? 'Stopped' : 'Error';
    emit({kind:'notice',text:signal.aborted ? 'Stopped. No further actions will run.' : e.message});
  } finally {
    clearInterval(keepAlive);
    await hand.finish();
    await chrome.storage.session.set({transcript:state.messages.slice(-60)});
    try { await trace({kind:'end',status:state.status,steps:state.steps.length,ms:Date.now()-started}); }
    finally { controller = null; state.running = false; state.paused = false; refreshTabs(); }
  }
}
async function handle(message) {
  await ready;
  if (message.type === 'stop') { stop(); return; }
  if (message.type === 'approve') { pending.get(message.id)?.(message.allow === true); return; }
  if (message.type === 'resume') { if (message.allow) resumeGate?.(true); else stop(); return; }
  if (message.type === 'send') {
    if (state.running) throw new Error('A task is already running');
    runPromise = run(String(message.text || '')); void runPromise; return;
  }
  if (message.type === 'new-chat') {
    if (state.running) throw new Error('Stop the task before starting a new chat');
    state.messages = []; state.steps = []; hand.notices = [];
    await chrome.storage.session.set({transcript:[]});
    await chrome.storage.session.remove('debugTrace');
  }
  if (message.type === 'settings') {
    if (state.running) throw new Error('Stop the task before changing settings');
    const s = message.settings;
    state.settings = {base:engineBase(s.base),apiKey:String(s.apiKey || ''),model:String(s.model || ''),vision:s.vision === true,stepCap:Math.max(1,Math.min(100,Math.floor(Number(s.stepCap) || 60)))};
    hand.settings = state.settings;
    await chrome.storage.local.set({settings:state.settings});
    await models();
  }
  if (message.type === 'connect') await models();
  if (message.type === 'share') {
    if (state.running) throw new Error('Stop the task before sharing another tab');
    const [tab] = await chrome.tabs.query({active:true,lastFocusedWindow:true});
    if (!tab) throw new Error('No active tab');
    if (!tab.url?.startsWith('chrome-error://')) safeURL(tab.url,state.settings.base);
    hand.tabs.set(tab.id,{attached:false,created:false,groupId:tab.groupId,refs:new Map(),epoch:0,binding:`rapidStop_${crypto.randomUUID().replaceAll('-','')}`});
    hand.active = tab.id; refreshTabs();
    emit({kind:'notice',text:`Shared tab: ${tab.title}`});
  }
  if (message.type === 'release') { if (state.running) throw new Error('Stop first'); await hand.takeover(message.id,'released by user'); refreshTabs(); }
  publish();
}
chrome.runtime.onConnect.addListener(port => {
  if (port.name !== 'rapid-panel' || port.sender?.url !== chrome.runtime.getURL('panel.html')) return;
  ports.add(port); void ready.then(publish);
  port.onDisconnect.addListener(() => ports.delete(port));
  port.onMessage.addListener(m => { void handle(m).catch(e => {
    state.status = m.type === 'connect' || m.type === 'settings' ? `Rapid engine not reachable at ${state.settings.base} — start it with rapid-mlx serve <model>. ${e.message}` : e.message;
    publish();
  }); });
});
chrome.runtime.onMessage.addListener((m,sender) => {
  if (m.type === 'stop' && sender.tab && hand.tabs.has(sender.tab.id)) stop();
});
chrome.debugger.onEvent.addListener((source,method,p) => { void hand.event(source.tabId,method,p).catch(e => emit({kind:'notice',text:e.message})); });
chrome.debugger.onDetach.addListener((source,reason) => {
  const t = hand.tabs.get(source.tabId);
  if (!t?.attached) return; // Expected detach during cleanup.
  t.attached = false; hand.invalidate(source.tabId);
  if (reason === 'canceled_by_user' && state.running) {
    for (const finish of [...pending.values()]) finish(false);
    if (!state.paused) pausePromise = new Promise(resolve => { resumeGate = resolve; });
    state.paused = true; hand.paused = true;
    state.status = 'Debugger detached. Keep working or stop?';
    hand.notices.push(`User detached debugger from tab ${source.tabId}; execution paused.`); publish();
  } else { void hand.takeover(source.tabId,`detached (${reason})`).then(refreshTabs); }
});
chrome.tabs.onRemoved.addListener(id => { void hand.takeover(id,'closed').then(refreshTabs); });
chrome.tabs.onUpdated.addListener((id,change,tab) => {
  const t = hand.tabs.get(id); if (!t) return;
  if (change.url || change.status === 'loading') hand.invalidate(id);
  if (!t.grouping && tab.groupId !== t.groupId && t.groupId !== undefined) void hand.takeover(id,'moved out of its group').then(refreshTabs);
});
chrome.action.onClicked.addListener(tab => { void chrome.sidePanel.open({windowId:tab.windowId}); });
chrome.commands.onCommand.addListener((command,tab) => {
  if (command === 'open-panel' && tab?.windowId) void chrome.sidePanel.open({windowId:tab.windowId});
});
