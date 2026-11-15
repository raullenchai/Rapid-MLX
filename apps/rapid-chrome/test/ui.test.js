import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { markdown, stepLabel, webURL } from '../src/ui.js';
import { BrowserHand } from '../src/cdp.js';
import { overlay } from '../src/overlay.js';
import { renderSteps, clip } from '../src/transcript.js';
import * as endpoint from '../src/endpoint.js';

function dom() {
  const nodes = new Map();
  const document = {body:{dataset:{}},activeElement:null};
  const make = (tag = 'div') => ({tagName:tag.toUpperCase(),dataset:{},style:{},children:[],options:[],elements:[],hidden:true,value:'',scrollHeight:100,scrollTop:0,clientHeight:100,
    append(...children) { this.children.push(...children); this.options.push(...children); },
    replaceChildren(...children) { this.children = children; this.options = children; },
    querySelectorAll(selector) { return this.children.filter(c => selector === 'details[open]' && c.open); },
    setAttribute(key,value) { this[key] = value; },getClientRects() { return [1]; },focus() { document.activeElement = this; },
    get textContent() { return this.text || this.children.map(c => c.textContent).join(''); },
    set textContent(text) { this.text = String(text); this.children = []; }
  });
  document.createElement = make;
  document.createTextNode = text => ({textContent:text});
  document.getElementById = id => { if (!nodes.has(id)) nodes.set(id,make()); return nodes.get(id); };
  document.querySelector = document.getElementById;
  return {document,nodes};
}
const walk = node => [node,...(node.children || []).flatMap(walk)];
const t = (key,args = []) => `${key}${args.length ? ': '+args.join(', ') : ''}`;
function panel() {
  const {document,nodes} = dom(); const sent = []; let render;
  const chrome = {runtime:{connect:() => ({postMessage:m => sent.push(m),onMessage:{addListener:f => {render = f;}},onDisconnect:{addListener(){}}})}};
  runInNewContext(readFileSync(new URL('../panel.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replace(/^export .*;$/gm,''),{document,chrome,...endpoint,markdown,stepLabel,renderSteps,clip,t,localize:s=>s,translateDocument(){},displayResult:s=>s,toolLabel:s=>s});
  return {document,nodes,sent,render,get:document.getElementById};
}
const snapshot = extra => ({settings:{base:'http://localhost:8000',model:'test'},models:[],messages:[],tabs:[],steps:[],approvals:[],running:false,connected:true,status:'Ready',...extra});

test('rich text creates safe nodes for links, code, lists, CJK, and literal hostile markup', () => {
  const {document} = dom(), root = document.createElement('div');
  markdown(document,root,'**摘要**\n1. One\n2. Two\n- Item\n[unsafe](javascript:alert) [safe](https://example.org/path)\n<img src=x onerror=alert(1)>\n```js\n<script>alert(1)</script>\n```');
  const nodes = walk(root);
  assert.equal(nodes.filter(n => n.tagName === 'STRONG')[0].textContent,'摘要');
  assert.equal(nodes.filter(n => n.tagName === 'OL').length,1);
  assert.equal(nodes.filter(n => n.tagName === 'UL').length,1);
  assert.equal(nodes.filter(n => n.tagName === 'A').length,1);
  assert.equal(nodes.find(n => n.tagName === 'A').rel,'noopener noreferrer');
  assert.equal(nodes.find(n => n.tagName === 'A').href,'https://example.org/path');
  assert.ok(nodes.some(n => n.textContent?.includes('<img src=x')));
  assert.equal(nodes.some(n => ['IMG','SCRIPT'].includes(n.tagName)),false);
  for (const value of ['javascript:alert(1)','data:text/html,x','https://u:p@example.org']) assert.equal(webURL(value),null);
});

test('step labels use observed targets and sites while excluding typed form contents', () => {
  assert.equal(stepLabel({name:'read_page',url:'https://shop.example/item'},t,String),'stepLooking: shop.example');
  assert.equal(stepLabel({name:'click',targetLabel:'Add to cart'},t,String),'stepClicking: Add to cart');
  assert.equal(stepLabel({name:'fill',args:{text:'private value'}},t,String),'stepTyping');
});

test('approval updates preserve buttons/focus and never focus Allow; stop and resume dispatch immediately', () => {
  const {render,get,sent,document} = panel();
  const s = snapshot({running:true,messages:[{role:'user',content:'Buy it'}],approvals:[{id:'a',action:"Click 'Buy now' on shop.example?",site:'https://shop.example'}]});
  render(s);
  const card = get('approvals').children[0], allow = card.children[4];
  assert.equal(document.activeElement,null);
  allow.focus(); render({...s,steps:[{id:'one',name:'read_page',status:'running',args:{}}]});
  assert.equal(get('approvals').children[0],card);
  assert.equal(document.activeElement,allow);
  assert.equal(get('activity').open,undefined,'timeline starts collapsed');
  get('stop').onclick(); assert.equal(sent.at(-1).type,'stop');
  get('resume').onclick(); assert.equal(sent.at(-1).type,'resume');
  assert.equal(get('stop').hidden,false); assert.equal(get('send').hidden,true);
});

test('updates preserve a scrolled-up reader and count new messages; pause states and recovery expose actions', () => {
  const {render,get,sent} = panel();
  render(snapshot({messages:[{role:'user',content:'Read this'}]}));
  get('transcript').scrollHeight = 1000; get('transcript').clientHeight = 100; get('transcript').scrollTop = 0;
  render(snapshot({messages:[{role:'user',content:'Read this'},{role:'assistant',kind:'answer',content:'**Done**'}]}));
  assert.equal(get('transcript').scrollTop,0); assert.equal(get('new-messages').hidden,false);
  get('new-messages').onclick(); assert.equal(get('transcript').scrollTop,1000);
  render(snapshot({running:true,paused:true,pauseReason:'takeover'}));
  assert.equal(get('pause-title').textContent,'takeoverTitle'); assert.equal(get('resume').textContent,'resumeTask');
  render(snapshot({running:true,paused:true,pauseReason:'challenge'}));
  assert.equal(get('pause-help').textContent,'challengeHelp');
  render(snapshot({status:'Error',connected:false})); assert.equal(get('recovery').hidden,false);
  get('recover').onclick(); assert.equal(get('settings').hidden,false);
  get('settings').onkeydown({key:'Escape',preventDefault(){}}); assert.equal(get('settings').hidden,true);
  get('result-tab').onclick(); assert.equal(sent.at(-1).type,'result-tab');
});

test('page overlay has a closed shadow, action-only controls, moving pointer, expiry, and cleanup', () => {
  const {document} = dom(); const actions = []; let interval, now = 0, shadowMode, shadowTree;
  const host = document.createElement('div'); let removed = 0;
  host.remove = () => { removed++; }; host.attachShadow = ({mode}) => { shadowMode = mode; return shadowTree = document.createElement('shadow'); };
  const create = document.createElement; document.createElement = tag => tag === 'div' && !document.hostCreated ? (document.hostCreated = true,host) : create(tag);
  document.documentElement = create('html');
  const sandbox = {document,Date:{now:() => now},setInterval:f => {interval = f;return 1;},clearInterval(){},setTimeout:()=>1,clearTimeout(){},stopBinding:action=>actions.push(action)};
  runInNewContext(`(${overlay.toString()})(true,'stopBinding',['Working','Stop','Take over'])`,sandbox);
  assert.equal(shadowMode,'closed'); assert.match(host.style.cssText,/pointer-events:none/);
  host.point(120,80); assert.ok(host.point);
  const buttons = walk(shadowTree).filter(node => node.tagName === 'BUTTON');
  buttons[0].onclick(); buttons[1].onclick();
  assert.deepEqual(actions,['stop','take-over']);
  interval(); assert.equal(actions.at(-1),'heartbeat');
  now = 5000; interval(); assert.equal(removed,1);
  assert.equal(sandbox.__rapidOverlay,undefined);
});

test('isolated overlay bindings dispatch take-over and stop only for the owned binding', async () => {
  let pauses = 0, stops = 0; const calls = [];
  const hand = new BrowserHand({send:async (...args) => calls.push(args)},{settings:{},approve:async () => false,takeover:async () => {pauses++;},stop:() => {stops++;}});
  hand.tabs.set(1,{attached:true,binding:'private-binding',overlayContext:42});
  await hand.event(1,'Runtime.bindingCalled',{name:'other',payload:'take-over'});
  assert.equal(pauses,0);
  await hand.event(1,'Runtime.bindingCalled',{name:'private-binding',payload:'take-over'});
  await hand.event(1,'Runtime.bindingCalled',{name:'private-binding',payload:'stop'});
  assert.equal(pauses,1); assert.equal(stops,1);
  await hand.event(1,'Runtime.bindingCalled',{name:'private-binding',payload:'heartbeat'});
  assert.equal(calls.at(-1)[2].contextId,42);
  hand.paused = true;
  await hand.event(1,'Runtime.bindingCalled',{name:'private-binding',payload:'heartbeat'});
  assert.equal(calls.length,1,'paused ownership must not renew the overlay');
});
