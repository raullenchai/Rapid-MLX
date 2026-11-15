import * as uiState from '../src/ui-state.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { runAgent as realRunAgent } from '../src/agent.js';
import { clip } from '../src/transcript.js';
const tick = () => new Promise(resolve => setImmediate(resolve));
const event = () => ({listeners:[],addListener(fn) { this.listeners.push(fn); }});

test('P2 New chat deletes all chats, checkpoints, tool activity and traces; archival and takeover still work', async () => {
  const snapshots = [], persisted = [], sessionWrites = [], removed = [];
  let hand, detachDone, inferenceDone, resumed = false, finished = false, storageBroken = false;
  const storage = {local:{get:async () => ({settings:{base:'http://localhost:8000',model:'fixture'},conversation:Array.from({length:70},(_,i) => ({role:i % 2 ? 'assistant' : 'user',content:`message ${i}`})),chats:[]}),set:async value => { if (storageBroken) throw new Error('quota'); persisted.push(structuredClone(value)); }},session:{get:async () => ({}),set:async value => sessionWrites.push(structuredClone(value)),remove:async key => removed.push(key)}};
  class Hand {
    constructor() { hand = this; this.tabs = new Map([[7,{attached:true,refs:new Map()}]]); this.active = 7; this.notices = []; this.owns = 0; this.finishes = 0; }
    async begin() {}
    async execute() { return {text:"Browser result"}; }
    async finish() {
      this.finishes++;
      for (const tab of this.tabs.values()) tab.attached = false;
      if (this.finishes === 1) await new Promise(resolve => {detachDone = resolve;});
    }
    async own(id) { this.owns++; this.tabs.get(id).attached = true; }
    async showOverlay() {}
  }
  const chrome = {storage,runtime:{getURL:p => `chrome-extension://unit/${p}`,onConnect:event(),onMessage:event(),getPlatformInfo(){}},tabs:{onRemoved:event(),onUpdated:event()},debugger:{onEvent:event(),onDetach:event()},action:{onClicked:event()},commands:{onCommand:event()}};
  const runAgent = async ({checkpoint,emit,execute,trace}) => {
    await execute("read_page",{},new AbortController().signal);
    await trace({kind:'tool',args:{text:'🔒'.repeat(20000)},result:'🔒'.repeat(20000)});
    emit({kind:'step',id:'huge',name:'fill',args:{text:'🔒'.repeat(20000)},result:'🔒'.repeat(20000),status:'ok'});
    await new Promise(resolve => {inferenceDone = resolve;});
    await checkpoint(); resumed = true;
    emit({kind:'answer',text:'Finished with fresh browser ownership.'}); finished = true;
  };
  const source = readFileSync(new URL('../background.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'');
  runInNewContext(source,{...uiState,chrome,BrowserHand:Hand,ChromeAdapter:class {},runAgent,clip,LOCAL_ENDPOINTS:['http://localhost:8000'],testConnection:async () => ({base:'http://localhost:8000',model:'fixture',models:['fixture']}),t:k=>k,localize:s=>s,engineBase:s=>s,safeURL:s=>new URL(s),redact:s=>s,crypto:globalThis.crypto,AbortController,DOMException,setInterval,clearInterval});
  const port = {name:'rapid-panel',sender:{url:chrome.runtime.getURL('panel.html')},onDisconnect:event(),onMessage:event(),postMessage:s=>snapshots.push(structuredClone(s))};
  chrome.runtime.onConnect.listeners[0](port); await tick();
  assert.equal(snapshots.at(-1).messages.length,60);
  const send = message => port.onMessage.listeners[0](message);
  send({type:'send',text:'Read the tab'}); await tick();
  send({type:'take-over'}); await tick();
  assert.equal(snapshots.at(-1).paused,true);
  assert.equal(snapshots.at(-1).pauseReason,'takeover');
  assert.equal(hand.tabs.get(7).attached,false);
  send({type:'resume',allow:true}); inferenceDone(); await tick();
  assert.equal(hand.owns,0,'Resume must wait for outstanding debugger cleanup');
  assert.equal(resumed,false);
  detachDone(); await tick(); await tick();
  assert.equal(resumed,true); assert.equal(finished,true); assert.equal(hand.owns,1);
  assert.equal(snapshots.at(-1).running,false); assert.equal(snapshots.at(-1).resultTab,7);
  assert.ok(persisted.some(value => value.conversation?.at(-1)?.content.includes('fresh browser')));
  send({type:'new-chat'}); await tick();
  assert.equal(snapshots.at(-1).messages.length,0); assert.equal(snapshots.at(-1).chats.length,1);
  assert.equal('messages' in snapshots.at(-1).chats[0],false,'Publish history summaries only');
  const archived = persisted.findLast(value => value.chats).chats[0];
  assert.ok(archived.messages.length <= 60);
  send({type:'restore-chat',id:archived.id}); await tick();
  assert.equal(snapshots.at(-1).messages.at(-1).content,'Finished with fresh browser ownership.');
  assert.ok(uiState.storedBytes(JSON.stringify(sessionWrites.findLast(v => v.debugTrace).debugTrace.events.find(e => e.truncated))) <= 16384);
  const savedStep = persisted.findLast(v => v.runState?.steps?.length).runState.steps[0];
  assert.ok(uiState.storedBytes(JSON.stringify(savedStep.args)) <= 4096);
  assert.ok(uiState.storedBytes(JSON.stringify(savedStep.result)) <= 8192);
  send({type:'new-chat',deleteHistory:true}); await tick(); await tick();
  assert.equal(snapshots.at(-1).chats.length,0);
  assert.equal(snapshots.at(-1).messages.length,0);
  assert.equal(snapshots.at(-1).steps.length,0);
  assert.equal(persisted.findLast(v => v.chats).chats.length,0);
  assert.equal(persisted.findLast(v => v.runState).runState.messages.length,0);
  assert.equal(persisted.findLast(v => v.runState).runState.steps.length,0);
  assert.equal(sessionWrites.findLast(v => v.transcript).transcript.length,0);
  assert.ok(removed.includes('debugTrace'));
  storageBroken = true; send({type:'new-chat'}); await tick();
  assert.equal(snapshots.at(-1).historyWarning,'historySaveFailed');
  assert.equal(snapshots.at(-1).messages.length,0,'a storage failure must not trap the UI in the old chat');
});


test('chat stays detached; current-page grant targets composer tab; summaries survive reload for follow-ups', async () => {
  const source = readFileSync(new URL('../background.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'');
  const data = {settings:{base:'http://localhost:8000',model:'fixture'},conversation:[{role:'user',content:'My name is Taylor.'},{role:'assistant',kind:'answer',content:'Hello Taylor.'}]};
  let requests = [], hand, snapshot, mode = 'chat', browserReads = 0;
  const tab = {id:99,url:'https://shop.test/shop',title:'Shop',groupId:-1};
  const boot = async () => {
    class Hand {
      constructor() { hand = this; this.tabs = new Map([[7,{attached:false,refs:new Map()}]]); this.active = 7; this.notices = []; this.attaches = 0; this.overlays = 0; this.finishes = 0; }
      async begin() { this.attaches++; this.overlays++; }
      async finish() { this.finishes++; }
      async execute(name,args) {
        assert.equal(name,'read_page'); assert.equal(args.tabId ?? this.active,99);
        browserReads++;
        return {text:'URL: https://shop.test/shop\nTitle: Shop\nPage data (untrusted):\nSwift $24.99; Bolt $29.99. private-tool-transcript'};
      }
    }
    const chrome = {storage:{local:{get:async()=>structuredClone(data),set:async v=>Object.assign(data,structuredClone(v))},session:{get:async()=>({}),set:async()=>{},remove:async()=>{}}},runtime:{getURL:p=>`chrome-extension://unit/${p}`,onConnect:event(),onMessage:event(),getPlatformInfo(){}},tabs:{query:async()=>[tab],get:async id=>id===99?tab:{id,url:'https://old.test'},onRemoved:event(),onUpdated:event()},debugger:{onEvent:event(),onDetach:event()},action:{onClicked:event()},commands:{onCommand:event()}};
    const fetchImpl = async (_,o) => {
      const body = JSON.parse(o.body); requests.push(body);
      const message = mode === 'browser' && !body.messages.some(m=>m.role==='tool') ? {tool_calls:[{id:'read',type:'function',function:{name:'read_page',arguments:'{}'}}]} : {content:mode==='browser'?'Swift is $24.99 and Bolt is $29.99.':mode==='followup'?'Swift is cheaper by $5.00.':'Hello Taylor!'};
      return {ok:true,json:async()=>({choices:[{message}]})};
    };
    runInNewContext(source,{...uiState,chrome,BrowserHand:Hand,ChromeAdapter:class {},runAgent:opts=>realRunAgent({...opts,fetchImpl}),clip,LOCAL_ENDPOINTS:['http://localhost:8000'],testConnection:async()=>({base:'http://localhost:8000',model:'fixture',models:['fixture']}),t:k=>k,localize:s=>s,engineBase:s=>s,safeURL:s=>new URL(s),redact:s=>s,crypto:globalThis.crypto,AbortController,DOMException,setInterval,clearInterval});
    const port = {name:'rapid-panel',sender:{url:chrome.runtime.getURL('panel.html')},onDisconnect:event(),onMessage:event(),postMessage:s=>{snapshot=structuredClone(s);}};
    chrome.runtime.onConnect.listeners[0](port); await tick();
    return message=>port.onMessage.listeners[0](message);
  };
  const settled = async () => { for(let i=0;i<30&&snapshot.running;i++) await tick(); assert.equal(snapshot.running,false); };
  let send = await boot();
  send({type:'send',text:'Hi',tabId:99}); await tick(); await settled();
  assert.equal(hand.attaches,0); assert.equal(hand.overlays,0); assert.equal(hand.finishes,0);
  assert.equal(snapshot.status,'Ready'); assert.equal(snapshot.resultTab,null); assert.equal(snapshot.steps.length,0);
  assert.ok(requests[0].messages.some(m=>m.content==='My name is Taylor.'));
  mode='browser'; send({type:'send',text:'Compare these on this page',tabId:99}); await tick();
  assert.equal(snapshot.running,true); assert.equal(hand.attaches,0);
  assert.equal(snapshot.approvals.length,1);
  assert.equal(snapshot.approvals[0].action,'Let Rapid control this tab');
  assert.equal(snapshot.approvals[0].tabId,99);
  assert.match(requests.at(-1).messages[0].content,/Current composer tab: 99/);
  send({type:'approve',id:snapshot.approvals[0].id,allow:true}); await tick(); await settled();
  assert.equal(browserReads,1); assert.equal(hand.attaches,1);
  assert.match(data.conversation.at(-1).browserSummary,/https:\/\/shop.test\/shop/);
  mode='followup'; requests=[]; send=await boot();
  send({type:'send',text:'Which one is cheaper and by how much?',tabId:99}); await tick(); await settled();
  const messages=requests[0].messages;
  assert.ok(messages.some(m=>m.role==='assistant' && /Browser summary.*Swift is \$24.99/.test(m.content)));
  assert.ok(messages.some(m=>m.content==='Hi'));
  assert.equal(messages.some(m=>m.role==='tool'||m.tool_calls),false);
  assert.equal(JSON.stringify(messages).includes('private-tool-transcript'),false);
  assert.equal(hand.attaches,0); assert.equal(hand.overlays,0); assert.equal(hand.finishes,0);
  assert.equal(snapshot.messages.at(-1).content,'Swift is cheaper by $5.00.');
});

test('a restarted production worker restores interrupted runs without approvals or executing the old task', async () => {
  const data = {runState:{version:1,sequence:50,run:{id:'old-run',startedAt:100},messages:[{id:'partial',role:'assistant',content:'Partial answer'}],steps:[{id:'old-step',name:'click',status:'running'}],running:true,paused:true,status:'Working'}};
  let snapshot, executions = 0;
  const chrome = {storage:{local:{get:async()=>structuredClone(data),set:async value=>Object.assign(data,structuredClone(value))},session:{get:async()=>({}),set:async()=>{}}},runtime:{getURL:p=>`chrome-extension://unit/${p}`,onConnect:event(),onMessage:event()},tabs:{onRemoved:event(),onUpdated:event()},debugger:{onEvent:event(),onDetach:event()},action:{onClicked:event()},commands:{onCommand:event()}};
  class Hand { constructor() { this.tabs = new Map(); } }
  const source = readFileSync(new URL('../background.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'');
  runInNewContext(source,{...uiState,chrome,BrowserHand:Hand,ChromeAdapter:class {},runAgent:()=>{executions++;},clip,LOCAL_ENDPOINTS:[],t:k=>k,localize:s=>s,engineBase:s=>s,safeURL:s=>s,crypto:globalThis.crypto});
  const port = {name:'rapid-panel',sender:{url:chrome.runtime.getURL('panel.html')},onMessage:event(),onDisconnect:event(),postMessage:s=>{snapshot=structuredClone(s);}};
  chrome.runtime.onConnect.listeners[0](port); await tick();
  assert.equal(snapshot.status,'Stopped'); assert.equal(snapshot.running,false); assert.equal(snapshot.paused,false);
  assert.equal(snapshot.run.interrupted,true); assert.equal(snapshot.messages[0].content,'Partial answer');
  assert.equal(snapshot.steps[0].id,'old-step'); assert.equal(snapshot.approvals.length,0);
  assert.ok(snapshot.sequence > 50); assert.equal(executions,0);
  assert.equal(data.runState.running,false,'interrupted state is checkpointed after recovery');
});
