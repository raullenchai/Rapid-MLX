import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserHand } from '../src/cdp.js';
import { informationTask } from '../src/challenge.js';
import { scenarioApproval } from './e2e/scenarios.mjs';

function setup({created = true, task = 'What is the weather in Cupertino today?', checkText = 'Verify you are human'} = {}) {
  const commands = [], pauses = [];
  let checked = true, url = 'https://search.test/search?q=weather';
  const hand = new BrowserHand({
    get:async () => ({url,title:'Search'}),
    inject:async () => {},
    send:async (_,method,params) => {
      commands.push({method,params});
      if (method === 'Accessibility.getFullAXTree') return {nodes:checked ? [{name:{value:checkText}}] : []};
      if (method === 'Page.getFrameTree') return {};
      if (method === 'Page.navigate') url = params.url;
      return {};
    }
  },{settings:{base:'http://127.0.0.1:8000'},approve:async () => true,
    challenge:async info => { pauses.push(info); checked = false; }});
  hand.active = 1;
  hand.tabs.set(1,{created,attached:true,epoch:0,refs:new Map([['old',{epoch:0}]])});
  const signal = new AbortController().signal;
  return {hand,signal,commands,pauses,task,setChecked:value => { checked = value; }};
}

test('information task classification is conservative for transactions and mixed requests', () => {
  assert.equal(informationTask('What is the weather in Cupertino today?'),true);
  assert.equal(informationTask('Research Kyoto and summarize the sights'),true);
  assert.equal(informationTask('Find a table for two. Do not reserve.'),true);
  assert.equal(informationTask('Find a table. Do not reserve. Then send the results.'),false);
  for (const task of ['Find a burrito and add one to cart','Find and cancel unused subscriptions','Find a table and reserve it','Read mail and reply','What is the weather? Then book a flight','Continue',''])
    assert.equal(informationTask(task),false,task);
});

test('agent-created information tab returns a check result, clears refs, and lets an independent source load', async () => {
  const {hand,signal,commands,pauses,task,setChecked} = setup();
  await hand.begin(signal,task);
  hand.tabs.get(1).refs.set('old',{epoch:0});
  const first = await hand.execute('read_page',{},signal);
  assert.equal(first.human_check,true);
  assert.match(first.text,/Do not interact.*one independent source.*URL provenance/);
  assert.equal(pauses.length,0);
  assert.equal(hand.tabs.get(1).refs.size,0);
  for (const [name,args] of [['click',{ref:'old'}],['fill',{ref:'old',text:'solve'}],['press',{key:'Tab'}],['scroll',{direction:'down'}],['navigate',{action:'reload'}],['navigate',{url:'https://search.test/'}],['open_tab',{url:'https://search.test/search?q=retry'}]])
    await assert.rejects(hand.execute(name,args,signal),/human_check/);
  await assert.rejects(hand.execute('navigate',{url:'https://weather.test/invented/deep/page'},signal),/unobserved_url/);
  assert.ok(!commands.some(c => c.method.startsWith('Input.') || c.method === 'Page.navigate'));
  setChecked(false);
  hand.settle = async () => {};
  const next = await hand.execute('navigate',{url:'https://weather.test/'},signal);
  assert.match(next.text,/weather.test/);
  assert.equal(pauses.length,0);
  assert.equal(hand.tabs.get(1).humanCheckHost,undefined);
});

test('second check pauses and Continue reobserves; allowance spans all agent tabs and resets per task', async () => {
  const {hand,signal,pauses,task} = setup();
  await hand.begin(signal,task);
  await hand.execute('read_page',{},signal);
  // A different agent-created tab, also challenged, cannot gain a new allowance.
  hand.tabs.set(2,{created:true,attached:true,epoch:0,refs:new Map()});
  const next = await hand.execute('read_page',{tabId:2},signal);
  assert.equal(pauses.length,1);
  assert.match(next.note,/fresh observation/);
  assert.equal(hand.tabs.get(2).humanCheckHost,undefined);
  await hand.begin(signal,task);
  assert.equal(hand.humanCheckOffered,false);
});

test('shared tabs, transactional tasks, unknown intent and transactional history pause immediately', async () => {
  for (const options of [{created:false},{task:'Find and cancel a subscription'},{task:'Continue'},{history:[{role:'user',content:'Add one item to cart'}]}]) {
    const {hand,signal,pauses,task} = setup(options);
    await hand.begin(signal,task,options.history || []);
    const result = await hand.execute('read_page',{},signal);
    assert.equal(pauses.length,1,JSON.stringify(options));
    assert.match(result.note,/fresh observation/);
    assert.equal(hand.humanCheckOffered,false);
  }
});

test('subscription harness permits only exact expected cancellation cards', () => {
  const scenario = {id:'fixture-subscriptions',checks:{approval_expected:'/Confirm cancellation/i',approval_decision:'allow'}};
  for (const label of ['Cancel StreamPlus','Confirm cancellation'])
    assert.equal(scenarioApproval(scenario,`Your approval is needed\nClick '${label}' on 127.0.0.1?\nAllow onceDeny`),'approved');
  for (const card of ["Click 'Cancel MusicNow' on 127.0.0.1?", "Click 'Cancel StreamPlus and delete account' on 127.0.0.1?", "Click 'Cancel StreamPlus' on evil.test?", 'Confirm cancellation', "Click 'Keep subscription' on 127.0.0.1?\nConfirm cancellation", 'Add to cart', 'Send'])
    assert.equal(scenarioApproval(scenario,card),'denied',card);
});

import { classifyTarget } from '../src/safety.js';
test('r5 exact price filter, cart view and cookie rejection labels are preparatory', () => {
  for (const name of ['Price, Not selected','Price, Selected','View Cart. Items in cart: 2, View cart','Reject Optional','Reject All Non-Essential Cookies']) {
    assert.equal(classifyTarget({name,role:'button',type:'submit',method:'post'}).consequential,false,name);
    for (const patch of [{sensitiveForm:true},{paymentForm:true},{name:name+' and buy now'},{name:name+' and delete account'}])
      assert.equal(classifyTarget({name,role:'button',type:'submit',method:'post',...patch}).consequential,true,JSON.stringify(patch));
    assert.equal(classifyTarget({name,role:'button',action:'https://shop.test/checkout'}).financial,true);
  }
  for (const name of ['San Francisco','94303','Location Sunnyvale, CA','Continue','View Cart. Items in cart: 2, Pay'])
    assert.equal(classifyTarget({name,role:'button'}).consequential,true,name);
});

import { boundedTrace, storedBytes } from '../src/ui-state.js';
test('large trace events keep tool identity, round and timing within the storage budget', () => {
  const event = {kind:'tool',round:14,name:'click',id:'tool-14',atMs:47216,ms:740,chars:24000,text:'界"\\'.repeat(20000)};
  const stored = boundedTrace(event);
  assert.equal(stored.truncated,true);
  for (const key of ['kind','round','name','id','atMs','ms','chars']) assert.equal(stored[key],event[key]);
  assert.ok(storedBytes(JSON.stringify(stored)) <= 16384);
  assert.equal(boundedTrace({kind:'end',steps:14}).steps,14);
});

import { runAgent } from '../src/agent.js';
test('human-check observation reaches the next model turn so it can ask the user', async () => {
  const {hand,signal,pauses,task} = setup();
  await hand.begin(signal,task);
  let requests = 0;
  const answer = await runAgent({settings:{base:'http://127.0.0.1:8000',model:'fixture',stepCap:3},task,signal,
    execute:(name,args) => hand.execute(name,args,signal),
    fetchImpl:async (_,options) => {
      requests++;
      if (requests === 2) {
        const tool = JSON.parse(options.body).messages.findLast(message => message.role === 'tool');
        assert.match(tool.content,/human_check:.*Do not interact.*independent source/);
      }
      return new Response(JSON.stringify({choices:[{message:requests === 1 ? {role:'assistant',tool_calls:[{id:'check',type:'function',function:{name:'read_page',arguments:'{}'}}]} : {role:'assistant',content:'The source requires a human check. Please help with the page.'}}]}));
    }});
  assert.match(answer,/Please help/);
  assert.equal(requests,2);
  assert.equal(pauses.length,0);
});


test('information-task load errors retain the immediate pause', async () => {
  const {hand,signal,pauses,task} = setup({checkText:'HTTP 503 Service unavailable'});
  await hand.begin(signal,task);
  await hand.execute('read_page',{},signal);
  assert.equal(pauses.length,1);
  assert.equal(pauses[0].kind,'load_error');
  assert.equal(hand.humanCheckOffered,false);
});
