import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { clip } from '../src/transcript.js';
const tick = () => new Promise(resolve => setImmediate(resolve));
const event = () => ({listeners:[],addListener(fn) { this.listeners.push(fn); }});

test('take-over yields all tabs, early Resume waits for detach, and bounded conversations restore locally', async () => {
  const snapshots = [], persisted = [];
  let hand, detachDone, inferenceDone, resumed = false, finished = false, storageBroken = false;
  const storage = {local:{get:async () => ({settings:{base:'http://localhost:8000',model:'fixture'},conversation:Array.from({length:70},(_,i) => ({role:i % 2 ? 'assistant' : 'user',content:`message ${i}`})),chats:[]}),set:async value => { if (storageBroken) throw new Error('quota'); persisted.push(structuredClone(value)); }},session:{get:async () => ({}),set:async () => {},remove:async () => {}}};
  class Hand {
    constructor() { hand = this; this.tabs = new Map([[7,{attached:true,refs:new Map()}]]); this.active = 7; this.notices = []; this.owns = 0; this.finishes = 0; }
    async begin() {}
    async finish() {
      this.finishes++;
      for (const tab of this.tabs.values()) tab.attached = false;
      if (this.finishes === 1) await new Promise(resolve => {detachDone = resolve;});
    }
    async own(id) { this.owns++; this.tabs.get(id).attached = true; }
    async showOverlay() {}
  }
  const chrome = {storage,runtime:{getURL:p => `chrome-extension://unit/${p}`,onConnect:event(),onMessage:event(),getPlatformInfo(){}},tabs:{onRemoved:event(),onUpdated:event()},debugger:{onEvent:event(),onDetach:event()},action:{onClicked:event()},commands:{onCommand:event()}};
  const runAgent = async ({checkpoint,emit}) => {
    await new Promise(resolve => {inferenceDone = resolve;});
    await checkpoint(); resumed = true;
    emit({kind:'answer',text:'Finished with fresh browser ownership.'}); finished = true;
  };
  const source = readFileSync(new URL('../background.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'');
  runInNewContext(source,{chrome,BrowserHand:Hand,ChromeAdapter:class {},runAgent,clip,LOCAL_ENDPOINTS:['http://localhost:8000'],testConnection:async () => ({base:'http://localhost:8000',model:'fixture',models:['fixture']}),t:k=>k,localize:s=>s,engineBase:s=>s,safeURL:s=>new URL(s),redact:s=>s,crypto:globalThis.crypto,AbortController,DOMException,setInterval,clearInterval});
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
  const archived = persisted.at(-1).chats[0];
  assert.ok(archived.messages.length <= 60);
  send({type:'restore-chat',id:archived.id}); await tick();
  assert.equal(snapshots.at(-1).messages.at(-1).content,'Finished with fresh browser ownership.');
  storageBroken = true; send({type:'new-chat'}); await tick();
  assert.equal(snapshots.at(-1).historyWarning,'historySaveFailed');
  assert.equal(snapshots.at(-1).messages.length,0,'a storage failure must not trap the UI in the old chat');
});
