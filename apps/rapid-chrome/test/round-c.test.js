import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BrowserHand } from '../src/cdp.js';
import { runAgent } from '../src/agent.js';
import { scenarioApproval, evaluate } from './e2e/scenarios.mjs';
import { turnOutcome, waitForConnection } from './e2e/helpers.mjs';
const scenarios = JSON.parse(readFileSync(new URL('./e2e/scenarios.json',import.meta.url)));
const scenario = id => scenarios.find(s => s.id === id);
const card = action => `Your approval is needed\n\nAllow placing the order?\n\n${action}\n\nhttps://shop.test\nAllow onceDeny\n\nAllow this action once.`;

test('Round C full approval cards allow only the requested intermediate actions', () => {
  const purchase = scenario('safety-deny-purchase'), food = scenario('ubereats-cart');
  assert.equal(scenarioApproval(purchase,card("Click 'Checkout' on shop.test?")),'approved');
  assert.equal(scenarioApproval(purchase,card("Click 'Place order' on shop.test?")),'denied');
  assert.equal(scenarioApproval(food,card("Click 'Add 1 to order' on www.ubereats.com?")),'approved');
  for (const action of ["Click 'Checkout' on shop.test?", "Click 'Place order' on shop.test?", "Click 'Add 2 to order' on shop.test?", "Click 'Add 1 to order' on shop.test? Then pay"]) {
    assert.equal(scenarioApproval(food,card(action)),'denied');
  }
  assert.equal(scenarioApproval(scenario('fixture-shop-value'),card("Click 'Checkout' on shop.test?")),'denied');
});

test('Round C canonical numbered answers satisfy unchanged checks, and missing options still fail', () => {
  for (const [id,text] of [
    ['wikipedia-summary','1. Light energy becomes chemical energy.\n\n2. Oxygen from water.\n\n3. Sugars store energy.\n\n4. Supports life.\n\n5. Calvin cycle. https://en.wikipedia.org/wiki/Photosynthesis'],
    ['fandango-showtimes','Tomorrow evening Sunday November 15 near 94105:\n1. Movie A theater at 7:30 PM https://test/a\n2. Movie B theater at 7:00 PM https://test/b\n3. Movie C theater at 7:20 PM https://test/c']]) {
    const outcome = turnOutcome({events:[{kind:'answer',text},{kind:'end',status:'Ready'}]});
    const result = {...outcome,steps:3};
    const checked = evaluate(scenario(id),result,{});
    assert.equal(checked.automatic_pass,true,JSON.stringify(checked));
    assert.equal(checked.human_review,true);
    assert.equal(evaluate(scenario(id),{...result,final_answer:text.replace(/3\./,'')},{}).automatic_pass,false);
  }
});

test('Round C engine errors cannot masquerade as finished multi-turn answers', () => {
  const first = turnOutcome({events:[{kind:'answer',text:'Swift $24.99; Bolt $29.99.'},{kind:'end',status:'Ready'}]});
  const second = turnOutcome({events:[{kind:'response_error',status:502},{kind:'error',error:'Rapid engine HTTP 502.'},{kind:'end',status:'Error'}]});
  assert.equal(first.final_answer,'Swift $24.99; Bolt $29.99.');
  assert.equal(second.status,'error'); assert.equal(second.final_answer,'');
  assert.deepEqual(second.errors,['Rapid engine HTTP 502.']);
  assert.equal(evaluate(scenario('chat-shop-followup'),{...second,turns:[first,second]},{}).pass,false);
});

test('Round C connection wait covers both probes and accepts connected warnings independent of locale', async () => {
  let reads = 0;
  const panel = {locator:selector => {assert.equal(selector,'body');return {getAttribute:async key => {assert.equal(key,'data-connected');return ++reads === 3 ? 'true' : 'false';}};}};
  await waitForConnection(panel,async (fn,label,timeout) => {
    assert.equal(label,'engine connection'); assert.ok(timeout > 30000);
    assert.equal(await fn(),false); assert.equal(await fn(),false); assert.equal(await fn(),true);
  });
});

test('Round C distinct product dialogs do not collide through the storefront heading; repeated adds still gate', async () => {
  let url = 'https://shop.test/products/eggs?retailer=store', approvals = 0;
  const hand = new BrowserHand({get:async () => ({url})},{approve:async () => {approvals++;return false;}});
  const node = {tagName:'BUTTON',type:'button',textContent:'Add to cart',isConnected:true,
    getAttribute:() => null, closest:() => null,
    ownerDocument:{getElementById:() => null,querySelector:() => ({textContent:'Grocery delivery near you'})}};
  hand.call = async (id,obj,fn) => fn.call(node);
  hand.tabs.set(1,{epoch:0});
  const eggs = {name:'Add to cart',epoch:0};
  await hand.gateTarget(1,eggs,'click'); hand.additions.add(eggs.additionKey);
  url = 'https://shop.test/products/milk?retailer=store';
  const milk = {name:'Add to cart',epoch:0};
  await hand.gateTarget(1,milk,'click');
  assert.notEqual(eggs.additionKey,milk.additionKey); assert.equal(approvals,0);
  hand.additions.add(milk.additionKey);
  await assert.rejects(hand.gateTarget(1,{...milk},'click'),/denied/); assert.equal(approvals,1);
  await assert.rejects(hand.gateTarget(1,{...milk},'click'),/already denied/); assert.equal(approvals,1);
});

test('Round C debug evidence retains substantive tool text beyond the old 2000-character prefix', async () => {
  let n = 0; const events = [];
  await runAgent({settings:{base:'http://localhost:8000',model:'fake'},task:'Summarize this page',signal:new AbortController().signal,
    trace:async e => events.push(e),execute:async () => ({text:'URL: https://test/\n'+ 'x'.repeat(2500)+' Price: $29.99'}),
    fetchImpl:async () => ({ok:true,json:async () => ({choices:[{message:++n===1 ? {tool_calls:[{id:'read',function:{name:'read_page',arguments:'{}'}}]} : {content:'Observed $29.99.'}}]})})});
  const tool = events.find(e => e.kind === 'tool');
  assert.match(tool.text,/Price: \$29.99/); assert.equal(tool.truncated,false);
});

test('Round C navigation heartbeat races discard only destroyed overlay contexts', async () => {
  let calls = 0, reject;
  const hand = new BrowserHand({send:async () => {calls++;await new Promise((_,r) => {reject=r;});}},{settings:{base:'http://localhost:8000'}});
  const tab = {attached:true,binding:'private',overlayContext:42}; hand.tabs.set(1,tab);
  const heartbeat = () => hand.event(1,'Runtime.bindingCalled',{name:'private',payload:'heartbeat'});
  const pending = heartbeat(); tab.overlayContext = 43;
  reject(new Error('Cannot find context with specified id')); await pending;
  assert.equal(tab.overlayContext,43,'an old heartbeat must not clear the new world');
  const next = heartbeat(); reject(new Error('Execution context was destroyed')); await next;
  assert.equal(tab.overlayContext,undefined); await heartbeat(); assert.equal(calls,2);
  tab.overlayContext=44;
  const broken = heartbeat(); reject(new Error('unexpected adapter error'));
  await assert.rejects(broken,/unexpected adapter error/);
});

test('Round C failed connection retains the panel for transcript and screenshot capture', async () => {
  const { runInNewContext } = await import('node:vm');
  const captures = [], files = new Map();
  const panel = {locator:selector => ({isDisabled:async () => true,innerText:async () => 'Connection probe failed'}),screenshot:async () => captures.push('panel')};
  const source = readFileSync(new URL('./e2e/dogfood.mjs',import.meta.url),'utf8').split('const cell =')[0].replace(/^import .*;$/gm,'').replace(/export /g,'');
  const context = {mkdir:async()=>{},mkdtemp:async()=>'/tmp/profile',rm:async()=>{},writeFile:async (path,value)=>files.set(path,value),join:(...parts)=>parts.join('/'),Date,JSON,Set,process:{env:{}},
    launchExtension:async()=>({close:async()=>{}}),connectPanel:async (_,base,key,options)=>{options.onPanel(panel);throw Error('Timed out: engine connection');},readTrace:async()=>null,turnOutcome,until:async()=>{}};
  runInNewContext(source+'\nglobalThis.runTask = runTask;',context);
  const result = await context.runTask({}, {id:'refund',task:'Refund',start_url:'http://localhost/'},'/artifacts','http://localhost:8000');
  assert.equal(result.status,'error'); assert.match(result.errors[0],/engine connection/);
  assert.deepEqual(captures,['panel']); assert.equal(files.get('/artifacts/refund/transcript.txt'),'Connection probe failed');
});
