import { errorType, runSnapshot, restoreRun } from './src/ui-state.js';
import { BrowserHand, ChromeAdapter } from './src/cdp.js';
import { runAgent } from './src/agent.js';
import { LOCAL_ENDPOINTS, testConnection, redact } from './src/endpoint.js';
import { clip } from './src/transcript.js';
import { t, localize } from './src/i18n.js';
import { engineBase, safeURL } from './src/safety.js';

const defaults = {base:'http://127.0.0.1:8000',apiKey:'',model:'',vision:false,stepCap:60,endpointMode:'local',reasoningEffort:'off',theme:'dark'};
const state = {epoch:crypto.randomUUID(),sequence:0,run:null,error:null,messages:[],steps:[],approvals:[],running:false,paused:false,status:'Ready',connected:false,settings:{...defaults},models:[],tabs:[],chats:[],resultTab:null,pauseReason:null};
const ports = new Set(), pending = new Map();
let snapshotWrite = Promise.resolve();
let controller = null, runPromise = null, resumeGate = null, pausePromise = null, pauseCleanup = null;
const hand = new BrowserHand(new ChromeAdapter(), {settings:state.settings,approve,emit,stop,challenge,takeover:pauseForTakeover});
const ready = (async () => {
  const {settings,conversation,chats,runState} = await chrome.storage.local.get(['settings','conversation','chats','runState']);
  state.chats = (chats || []).slice(-8);
  if (settings) { try { engineBase(settings.base); state.settings = {...defaults,...settings,endpointMode:settings.endpointMode || (LOCAL_ENDPOINTS.includes(settings.base) ? 'local' : 'custom')}; } catch {} }
  hand.settings = state.settings;
  const saved = await chrome.storage.session.get(['transcript','owned']);
  state.messages = (runState?.messages || conversation || saved.transcript || []).slice(-60);
  Object.assign(state,restoreRun(runState));
  state.epoch = crypto.randomUUID();
  state.sequence = runState?.sequence || 0;
  for (const tab of saved.owned || []) {
    try {
      if (runState?.running) { try { await chrome.debugger.detach?.({tabId:tab.id}); } catch {} }
      const current = await chrome.tabs.get(tab.id);
      safeURL(current.url,state.settings.base);
      if (tab.created && current.groupId !== tab.groupId) continue;
      hand.tabs.set(tab.id,{...tab,attached:false,refs:new Map(),epoch:0});
      hand.active = tab.id;
    } catch {}
  }
  refreshTabs();
  if (settings) {
    try { await models(undefined,true,true); if (state.run?.interrupted) { state.status = 'Stopped'; publish(); } }
    catch (e) { state.error = {type:errorType(e.message)}; state.status = localize(e.message); publish(); }
  } else { state.status = state.run?.interrupted ? 'Stopped' : t('stepConnect'); publish(); }
})();
function publish() {
  state.sequence++;
  const snapshot = JSON.parse(JSON.stringify(runSnapshot(state)));
  snapshotWrite = snapshotWrite.then(() => chrome.storage.local.set({runState:snapshot})).catch(() => { state.historyWarning = t('historySaveFailed'); });
  for (const port of ports) { try { port.postMessage({...state,chats:state.chats.map(({id,title}) => ({id,title})),messages:state.messages.slice(-60).map(m => ({...m,content:clip(m.content,12000)})),steps:state.steps.map(s => ({...s,result:clip(s.result)})),status:localize(state.status)}); } catch { ports.delete(port); } }
}
function refreshTabs() {
  state.tabs = [...hand.tabs].map(([id,t]) => ({id,attached:t.attached})); publish();
  void chrome.storage.session.set({owned:[...hand.tabs].map(([id,t]) => ({id,created:t.created,groupId:t.groupId,binding:t.binding}))});
}
function persistConversation() {
  return chrome.storage.local.set({conversation:state.messages.slice(-60).map(m => ({...m,content:clip(m.content,12000)})),chats:state.chats.slice(-8)}).then(() => { state.historyWarning = ''; },() => { state.historyWarning = t('historySaveFailed'); publish(); });
}
function archiveConversation() {
  if (!state.messages.some(m => m.role === 'user')) return;
  state.chats.push({id:crypto.randomUUID(),title:clip(state.messages.find(m => m.role === 'user').content,80),messages:state.messages.slice(-60).map(m => ({...m,content:clip(m.content,4000)}))});
  state.chats = state.chats.slice(-8);
}
async function pauseForTakeover() {
  if (!state.running || state.paused) return;
  pausePromise = new Promise(resolve => { resumeGate = resolve; });
  state.paused = true; state.pauseReason = 'takeover'; hand.paused = true;
  state.status = t('takeoverTitle');
  for (const finish of [...pending.values()]) finish(false);
  pauseCleanup = hand.finish();
  publish(); await pauseCleanup; refreshTabs();
}
function emit(event) {
  if (event.kind === 'step') {
    const existing = state.steps.findIndex(s => s.id === event.id);
    const tabId = event.args?.tabId ?? hand.active;
    const targetLabel = hand.tabs.get(tabId)?.refs.get(event.args?.ref)?.name;
    const detail = {url:state.steps[existing]?.url,targetLabel:state.steps[existing]?.targetLabel,...event,...(targetLabel ? {targetLabel} : {}),result:clip(event.result)};
    if (existing < 0) state.steps.push(detail); else state.steps[existing] = detail;
  } else if (event.kind === 'answer' || event.kind === 'notice' || event.kind === 'progress') {
    state.messages.push({id:crypto.randomUUID(),role:'assistant',kind:event.kind,content:event.text,...(event.browserSummary ? {browserSummary:event.browserSummary} : {}),at:new Date().toISOString()});
    if (state.messages.length > 60) state.messages.splice(0,state.messages.length - 60);
    void chrome.storage.session.set({transcript:state.messages.slice(-60)});
    void persistConversation();
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
  state.handover = info;
  state.paused = true; state.pauseReason = 'challenge'; hand.paused = true;
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
    await pauseCleanup; pauseCleanup = null;
    controller.signal.throwIfAborted();
    hand.paused = false;
    for (const [id,t] of hand.tabs) if (!t.attached) { await hand.own(id); await hand.showOverlay(id,true); }
    state.paused = false; state.pauseReason = null; state.handover = null; state.status = 'Working'; refreshTabs();
  }
  if (messages && hand.notices.length) messages.push({role:'user',content:`Browser ownership notice (from extension): ${hand.notices.splice(0).join('\n')}`});
}
async function models(signal, probe = true, discover = false) {
  state.connected = false; state.status = t('connecting'); publish();
  const candidates = discover && state.settings.endpointMode === 'local' ? [...new Set([state.settings.base,...LOCAL_ENDPOINTS])] : [state.settings.base];
  let failure;
  for (const base of candidates) {
    try {
      const result = await testConnection({...state.settings,base},{signal,probe,onModels:info => {
        state.models = info.models; Object.assign(state.settings,{base:info.base,model:info.model}); publish();
      }});
      Object.assign(state.settings,{base:result.base,model:result.model});
      state.models = result.models; state.connected = true;
      state.error = null;
      if (!state.run?.interrupted) state.status = result.warning ? t(result.warning) : t('connected');
      await chrome.storage.local.set({settings:state.settings}); publish(); return;
    } catch (e) { failure = e; if (e.message !== 'unreachable' && e.message !== 'permissionMissing') break; }
  }
  throw failure;
}
async function run(task, tabId) {
  if (state.running) throw new Error('A task is already running');
  if (!task.trim()) return;
  state.run = {id:crypto.randomUUID(),startedAt:Date.now(),endedAt:null}; state.error = null; state.handover = null;
  hand.paused = false; state.running = true; state.resultTab = null; state.pauseReason = null; state.steps = []; state.status = 'Working';
  controller = new AbortController();
  const signal = controller.signal;
  const history = state.messages.filter(m => !['progress','notice'].includes(m.kind)).slice(-60);
  let currentTab;
  let browserStarted = false, grantDenied = false;
  const visited = new Set();
  const debugTrace = {version:1,id:crypto.randomUUID(),task,startedAt:new Date().toISOString(),events:[]};
  const started = Date.now();
  const trace = async event => {
    debugTrace.events.push(redact({atMs:Date.now()-started,...event},state.settings.apiKey));
    await chrome.storage.session.set({debugTrace:redact(debugTrace,state.settings.apiKey)});
  };
  state.messages.push({id:crypto.randomUUID(),role:'user',content:task,at:new Date().toISOString()}); void persistConversation(); publish();
  // A Chrome API call keeps a waiting approval / local inference run alive.
  const keepAlive = setInterval(() => { void chrome.runtime.getPlatformInfo(); },20000);
  try {
    currentTab = tabId != null ? await chrome.tabs.get(tabId) : (await chrome.tabs.query?.({active:true,lastFocusedWindow:true}))?.[0];
    await trace({kind:'start'});
    if (!state.settings.model) await models(signal);
    signal.throwIfAborted();
    await runAgent({settings:{...state.settings},task,history,currentTab,signal,execute:async (...a) => {
      if (!browserStarted && currentTab) hand.active = currentTab.id;
      if (currentTab && (a[1]?.tabId ?? hand.active) === currentTab.id && !hand.tabs.has(currentTab.id) && a[0] !== 'open_tab' && a[0] !== 'list_tabs') {
        if (grantDenied) throw new Error('denied: tab grant already denied for this turn');
        safeURL(currentTab.url,state.settings.base);
        const allowed = await approve({kind:'grant',action:'Let Rapid control this tab',site:`${currentTab.title || ''}\n${currentTab.url}`,tabId:currentTab.id});
        signal.throwIfAborted();
        if (!allowed) { grantDenied = true; throw new Error('denied: Let Rapid control this tab'); }
        hand.tabs.set(currentTab.id,{attached:false,created:false,groupId:currentTab.groupId,refs:new Map(),epoch:0,binding:`rapidStop_${crypto.randomUUID().replaceAll('-','')}`});
        if (browserStarted) { await hand.own(currentTab.id); await hand.showOverlay(currentTab.id,true); }
      }
      if (!browserStarted) {
        browserStarted = true; await hand.begin(signal,task,history); refreshTabs();
      }
      const id = a[1]?.tabId ?? hand.active;
      const step = state.steps.findLast(step => step.status === 'running');
      if (step && id != null) { try { step.url = (await chrome.tabs.get(id)).url; publish(); } catch {} }
      let result;
      try { result = await hand.execute(...a); }
      catch (e) {
        if (/blocked_sensitive_field|needs_human_permission/.test(e.message)) {
          const kind = /permission/.test(e.message) ? 'permission' : 'login';
          await challenge({kind,message:t(kind === 'permission' ? 'handoverPermission' : 'handoverSensitive'),reason:t(kind === 'permission' ? 'permissionHandover' : 'sensitiveHandover')});
          signal.throwIfAborted();
          // Never retry a sensitive action after handover; require a fresh read.
          throw new Error('needs_human: user returned; read_page for a fresh observation');
        }
        throw e;
      }
      const text = result.observation?.text || result.text || '';
      for (const match of text.matchAll(/^URL:\s*(https?:\/\/\S+)/gm)) visited.add(match[1]);
      if (step?.url) visited.add(step.url);
      return result;
    },emit:event => {
      if (event.kind === 'answer' && browserStarted) {
        // Store only a small summary with the final reply, never tool transcripts.
        const summary = `Sites: ${[...visited].join(', ').slice(0,1200) || 'none recorded'}. Result: ${event.text.slice(0,2500)}`;
        emit({...event,browserSummary:summary});
      } else emit(event);
    },checkpoint,trace});
    await trace({kind:'answer',text:state.messages.at(-1)?.content});
    state.status = 'Ready';
    const final = state.messages.at(-1)?.content || '';
    const type = errorType(final);
    if (type === 'stepCap' || (type === 'noTools' && /completion is unconfirmed|model did not inspect/i.test(final))) state.error = {type};
  } catch (e) {
    await trace({kind:'error',error:e.message,aborted:signal.aborted});
    state.status = signal.aborted ? 'Stopped' : 'Error';
    state.error = signal.aborted ? null : {type:errorType(e.message),message:redact(e.message,state.settings.apiKey)};
    emit({kind:'notice',text:signal.aborted ? 'Stopped. No further actions will run.' : e.message});
  } finally {
    clearInterval(keepAlive);
    state.resultTab = browserStarted ? hand.active : null;
    if (browserStarted) await hand.finish();
    await persistConversation();
    await chrome.storage.session.set({transcript:state.messages.slice(-60)});
    try { await trace({kind:'end',status:state.status,steps:state.steps.length,ms:Date.now()-started}); }
    finally { controller = null; state.running = false; state.paused = false; state.handover = null; state.run.endedAt = Date.now();
      state.steps = state.steps.map(step => step.status === 'running' ? {...step,status:signal.aborted ? 'stopped' : 'error'} : step);
      refreshTabs(); await snapshotWrite; }
  }
}
async function handle(message) {
  if (message.type === 'stop') { stop(); return; }
  await ready;
  if (message.type === 'approve') { pending.get(message.id)?.(message.allow === true); return; }
  if (message.type === 'handover') {
    if (!state.running || state.paused || !['login','payment','permission'].includes(message.kind)) return;
    await pauseForTakeover();
    state.handover = {kind:message.kind,message:t(({login:'handoverLogin',payment:'handoverPayment',permission:'handoverPermission'})[message.kind])};
    publish(); return;
  }
  if (message.type === 'take-over') { await pauseForTakeover(); return; }
  if (message.type === 'result-tab') { if (state.resultTab) { const tab = await chrome.tabs.update(state.resultTab,{active:true}); await chrome.windows.update(tab.windowId,{focused:true}); } return; }
  if (message.type === 'restore-chat') {
    if (state.running) throw new Error('Stop first');
    const chat = state.chats.find(chat => chat.id === message.id);
    if (chat) { archiveConversation(); state.run = null; state.error = null; state.messages = chat.messages.slice(-60); state.steps = []; state.resultTab = null; state.status = 'Ready'; await persistConversation(); publish(); }
    return;
  }
  if (message.type === 'resume') { if (message.allow) resumeGate?.(true); else stop(); return; }
  if (message.type === 'send') {
    if (state.running) throw new Error('A task is already running');
    runPromise = run(String(message.text || ''),message.tabId); void runPromise; return;
  }
  if (message.type === 'new-chat') {
    if (state.running) throw new Error('Stop the task before starting a new chat');
    archiveConversation();
    state.run = null; state.error = null; state.messages = []; state.steps = []; state.resultTab = null; state.status = 'Ready'; hand.notices = [];
    await persistConversation();
    await chrome.storage.session.set({transcript:[]});
    await chrome.storage.session.remove('debugTrace');
  }
  if (message.type === 'settings') {
    if (state.running) throw new Error('Stop the task before changing settings');
    const s = message.settings;
    const base = engineBase(s.base);
    if (s.endpointMode !== 'custom' && !LOCAL_ENDPOINTS.includes(base)) throw new Error('invalidEndpoint');
    state.settings = {theme:s.theme === 'light' ? 'light' : 'dark',endpointMode:s.endpointMode === 'custom' ? 'custom' : 'local',reasoningEffort:['low','high'].includes(s.reasoningEffort) ? s.reasoningEffort : 'off',base,apiKey:String(s.apiKey || ''),model:String(s.model || ''),vision:s.vision === true,stepCap:Math.max(1,Math.min(100,Math.floor(Number(s.stepCap) || 60)))};
    hand.settings = state.settings;
    await chrome.storage.local.set({settings:state.settings});
    await models(undefined,true,true);
  }
  if (message.type === 'connect') {
    if (state.running) throw new Error('Stop first');
    await models(undefined,true,true);
  }
  if (message.type === 'open-tab') {
    if (state.running) throw new Error('Stop first');
    await chrome.tabs.create({url:'about:blank'});
    state.status = t('openTabHelp'); publish();
  }
  if (message.type === 'share') {
    if (state.running) throw new Error('Stop the task before sharing another tab');
    const [tab] = await chrome.tabs.query({active:true,lastFocusedWindow:true});
    if (!tab) throw new Error('No active tab');
    if (!tab.url?.startsWith('chrome-error://')) safeURL(tab.url,state.settings.base);
    hand.tabs.set(tab.id,{attached:false,created:false,groupId:tab.groupId,refs:new Map(),epoch:0,binding:`rapidStop_${crypto.randomUUID().replaceAll('-','')}`});
    hand.active = tab.id; refreshTabs();
    // The tab chip is the grant receipt; keep it out of the conversation.
  }
  if (message.type === 'revoke') {
    if (state.running) { stop(); await runPromise; }
    await hand.takeover(message.id,'released by user'); refreshTabs(); return;
  }
  if (message.type === 'release') { if (state.running) throw new Error('Stop first'); await hand.takeover(message.id,'released by user'); refreshTabs(); }
  publish();
}
chrome.runtime.onConnect.addListener(port => {
  if (port.name !== 'rapid-panel' || port.sender?.url !== chrome.runtime.getURL('panel.html')) return;
  ports.add(port); void ready.then(publish);
  port.onDisconnect.addListener(() => ports.delete(port));
  port.onMessage.addListener(m => { void handle(m).catch(e => {
    if (m.type === 'connect' || m.type === 'settings') state.connected = false;
    state.error = {type:errorType(e.message),message:redact(e.message,state.settings.apiKey)};
    state.status = localize(redact(e.message,state.settings.apiKey));
    publish();
  }); });
});
chrome.runtime.onMessage.addListener((m,sender,respond) => {
  if (m.type === 'overlay-heartbeat') { respond(!!(sender.tab && hand.tabs.get(sender.tab.id)?.attached && state.running && !state.paused)); return; }
  if (!sender.tab || !hand.tabs.has(sender.tab.id)) return;
  if (m.type === 'stop') stop();
  if (m.type === 'take-over') void pauseForTakeover();
});
chrome.debugger.onEvent.addListener((source,method,p) => { void hand.event(source.tabId,method,p).catch(e => emit({kind:'notice',text:e.message})); });
chrome.debugger.onDetach.addListener((source,reason) => {
  const t = hand.tabs.get(source.tabId);
  if (!t?.attached) return; // Expected detach during cleanup.
  t.attached = false; hand.invalidate(source.tabId);
  if (reason === 'canceled_by_user' && state.running) {
    for (const finish of [...pending.values()]) finish(false);
    if (!state.paused) pausePromise = new Promise(resolve => { resumeGate = resolve; });
    state.paused = true; state.pauseReason = 'takeover'; hand.paused = true;
    pauseCleanup = hand.finish();
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
