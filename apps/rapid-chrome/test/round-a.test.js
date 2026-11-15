import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand, fillNative } from '../src/cdp.js';
import { runAgent, SYSTEM_PROMPT, assembleMessages } from '../src/agent.js';
import { outlineAX } from '../src/ax.js';
import { classifyTarget } from '../src/safety.js';
import { scenarioApproval } from './e2e/scenarios.mjs';

const signal = () => new AbortController().signal;
const settings = {base:'http://127.0.0.1:8000',model:'fake'};
const call = (name,args={},id='call') => ({id,type:'function',function:{name,arguments:JSON.stringify(args)}});
const response = message => ({ok:true,json:async () => ({choices:[{message}]})});

// Execute the serialized DOM helper in a separate realm, as CDP does. Prototype
// setters and events are exercised rather than stubbing the helper's result.
function native(tagName,type='',options=[]) {
  class Control {
    get value() { return this.current || ''; }
    set value(v) { this.current = v === '2026-99-99' ? '' : v; }
  }
  const events=[];
  const e = new Control();
  Object.assign(e,{tagName,type,options,isConnected:true,getClientRects:() => [{}],closest:() => null,
    ownerDocument:{defaultView:{HTMLSelectElement:Control,HTMLInputElement:Control}},
    dispatchEvent:event => events.push(event.type)});
  const fill = (text,choose,target=e) => runInNewContext(`(${fillNative}).call(e,text,choose)`,{e:target,text,choose,Event:class {constructor(type) {this.type=type;}}});
  return {e,events,fill};
}

test('native select accepts labels and values, emits change, and supports an option without layout', () => {
  const {e,events,fill} = native('SELECT','select-one',[
    {value:'',label:'Choose an order'},{value:'ORD-104',label:'ORD-104 — Desk lamp'},{value:'ORD-105',label:'ORD-105 — Notebook'}
  ]);
  assert.equal(fill('ORD-104 — Desk lamp').value,'ORD-104');
  assert.equal(fill('ORD-105').value,'ORD-105');
  const option = {tagName:'OPTION',value:'ORD-104',closest:() => e};
  assert.equal(fill(undefined,undefined,option).value,'ORD-104');
  assert.deepEqual(events,['input','change','input','change','input','change']);
});

test('native select rejects ambiguous, missing and disabled options without changing value', () => {
  const {e,fill,events} = native('SELECT','select-one',[
    {value:'a',label:'Plan A'},{value:'b',label:'Plan B'},{value:'c',label:'Plan C',disabled:true}
  ]);
  e.value='a';
  for (const text of ['Plan','missing','Plan C','']) assert.match(fill(text).error,/option_not_unique/);
  e.disabled=true;
  assert.match(fill('b').error,/disabled_control/);
  assert.equal(e.value,'a'); assert.deepEqual(events,[]);
});

test('native fill and option click integrate with the browser gates without sending text or clicking option geometry', async () => {
  const {hand,commands} = handForGate(async () => true);
  const control = native('SELECT','select-one',[{value:'1',label:'1'},{value:'2',label:'2'}]);
  hand.resolve=async () => ({objectId:'select',role:'combobox',epoch:0});
  let gated=0;
  hand.gateTarget=async () => {gated++;};
  hand.call=async (_,__,fn,args=[]) => {
    assert.equal(fn,fillNative);
    return control.fill(...args);
  };
  await hand.executeCore('fill',{ref:'e1',text:'2'},hand.signal);
  assert.equal(control.e.value,'2'); assert.equal(gated,1); assert.ok(commands.every(c => ['Runtime.evaluate','Page.getFrameTree'].includes(c)));
  const option={tagName:'OPTION',value:'1',closest:() => control.e};
  hand.call=async () => control.fill(undefined,undefined,option);
  assert.equal(await hand.click(1,{role:'option',epoch:0}),'native option selection');
  assert.equal(control.e.value,'1'); assert.ok(commands.every(c => ['Runtime.evaluate','Page.getFrameTree'].includes(c)));
});

test('failed clicks include a fresh outline so the model can recover without a stale-ref reread loop', async () => {
  const {hand} = handForGate(async () => true);
  hand.effectWorld=async () => {throw Error('world unavailable');};
  hand.executeCore=async name => {
    if (name==='read_page') return {text:'[e9] button "Close"'};
    throw Error('stale_ref: old target');
  };
  await assert.rejects(hand.execute('click',{ref:'e1'},hand.signal),error => {
    assert.match(error.message,/stale_ref/); assert.match(error.observation.text,/e9/); return true;
  });
});

test('query matches outrank large live calendar content while keeping dialog dismissal available', () => {
  const ax=(id,role,name,children=[]) => ({nodeId:String(id),backendDOMNodeId:id,role:{value:role},name:{value:name},childIds:children.map(String)});
  const nodes=[ax(1,'dialog','Choose day',[2]),ax(2,'button','Close'),
    ...Array.from({length:100},(_,i) => ({...ax(i+10,'StaticText','NOVEMBER '+i+' calendar '.repeat(20)),properties:[{name:'live',value:{value:'polite'}}]})),
    ax(200,'link','Wild Horse Nine — 7:30 pm')];
  const result=outlineAX(nodes,{query:'Wild Horse Nine',limit:1500});
  assert.match(result.text,/button "Close"/); assert.match(result.text,/Wild Horse Nine — 7:30 pm/);
  assert.ok(result.text.indexOf('Wild Horse Nine')<result.text.indexOf('NOVEMBER'));
  assert.ok([...result.refs.values()].some(ref => ref.backendDOMNodeId===200));
});

test('relative-date tasks receive the browser clock and timezone', () => {
  assert.match(assembleMessages([],'next Friday')[0].content,/Current browser date\/time:.*GMT.*Resolve relative dates/);
});

test('read_page exposes canonical native date/time values and hidden reply recipients', async () => {
  const {hand} = handForGate(async () => true);
  const form={method:'post',elements:[{name:'to',value:'jordan@team.test'}],getAttribute:() => null};
  const controls = [
    {tagName:'INPUT',type:'date',value:'1990-05-12',form},
    {tagName:'INPUT',type:'time',value:'14:00',form},
    {tagName:'SELECT',value:'2',form}
  ];
  const roles=['Date','InputTime','combobox'];
  hand.a.send=async (_,method,p) => {
    if (method==='Accessibility.getFullAXTree') return {nodes:controls.map((_,i) => ({nodeId:String(i),backendDOMNodeId:i+1,name:{value:roles[i]},role:{value:roles[i]},properties:[{name:'focusable',value:{value:true}}]}))};
    if (method==='DOM.resolveNode') return {object:{objectId:String(p.backendNodeId-1)}};
    if (method==='Runtime.callFunctionOn') return {result:{value:runInNewContext(`(${p.functionDeclaration}).call(e)`,{e:controls[Number(p.objectId)],document:{forms:[form]}})}};
    return {};
  };
  const result=await hand.executeCore('read_page',{},hand.signal);
  assert.match(result.text,/Native \[e1\] date value="1990-05-12"; fill with YYYY-MM-DD/);
  assert.match(result.text,/Native \[e2\] time value="14:00"; fill with HH:mm/);
  assert.match(result.text,/Native \[e3\] select value="2"; fill with exact option label\/value/);
  assert.match(result.text,/Recipients: to: jordan@team.test/);
});

test('native date/time fill handles whole values, rejects segments, restores rejected dates and respects hidden controls', () => {
  for (const [type,value,bad] of [['date','1990-05-12','05'],['time','14:00','02:00 PM'],['datetime-local','2026-11-24T14:00','Tuesday']]) {
    const {e,fill,events} = native('INPUT',type);
    assert.equal(fill(value).value,value);
    assert.match(fill(bad).error,/requires/); assert.equal(e.value,value);
    if (type==='date') { assert.match(fill('2026-99-99').error,/browser rejected/); assert.equal(e.value,value); }
    const segment = {tagName:'DIV',getRootNode:() => ({host:e})};
    assert.equal(fill(value,undefined,segment).value,value);
    e.getClientRects=() => [];
    assert.match(fill(value).error,/hidden_control/);
    assert.deepEqual(events,['input','change','input','change']);
  }
});

function handForGate(approve) {
  const commands=[];
  const hand = new BrowserHand({get:async () => ({url:'https://mail.test/compose'}),send:async (_,name) => {
    commands.push(name); return {};
  }},{settings,approve});
  hand.active=1; hand.signal=signal(); hand.tabs.set(1,{attached:true,epoch:0,refs:new Map()});
  return {hand,commands};
}

test('denied submit cannot re-request via changed refs, Enter, or fill+submit; unrelated effects remain possible', async () => {
  const cards=[];
  const {hand,commands} = handForGate(async card => {cards.push(card);return false;});
  let info={connected:true,name:'Buy tickets',type:'submit',submitter:'Buy tickets',action:'https://mail.test/buy',method:'post'};
  hand.inspect=async () => info;
  await assert.rejects(hand.gateTarget(1,{epoch:0,name:'Buy tickets'},'click'),/^Error: denied:/);
  hand.invalidate(1);
  for (const opts of [{},{submit:true},{fill:true,submit:true}]) {
    info={...info,name:opts.submit ? 'Ticket quantity' : 'Buy tickets',type:opts.submit ? 'text' : 'submit'};
    await assert.rejects(hand.gateTarget(1,{epoch:1,name:'new ref'},'Press Enter',opts),/already denied/);
  }
  assert.equal(cards.length,1); assert.ok(commands.every(c => ['Runtime.evaluate','Page.getFrameTree'].includes(c)));
  info={...info,name:'Submit refund',type:'submit',submitter:'Submit refund',action:'https://mail.test/refund'};
  await assert.rejects(hand.gateTarget(1,{epoch:1},'click'),/denied/);
  assert.equal(cards.length,2);
  hand.own=async () => {}; hand.showOverlay=async () => {};
  await hand.begin(hand.signal);
  await assert.rejects(hand.gateTarget(1,{epoch:2},'click'),/denied/);
  assert.equal(cards.length,3);
});

test('Send approval exposes actual recipient even when the form keeps To hidden', async () => {
  let card;
  const {hand} = handForGate(async c => {card=c;return false;});
  hand.inspect=async () => ({connected:true,name:'Send',type:'submit',method:'post',submitter:'Send',recipients:'to: jordan@team.test'});
  await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/denied/);
  assert.match(card.action,/Recipients: to: jordan@team.test/);
  assert.match(SYSTEM_PROMPT,/sender asking for a link.*sender supplying it/);
});

test('ordinary standalone buttons and calendar date filters do not inherit a purchase effect', () => {
  for (const name of ['Got it','Back','Acknowledge','4+ rating']) assert.equal(classifyTarget({name,type:'submit'}).consequential,false,name);
  const context={type:'button',form:'Book your tickets',method:'post',action:'https://tickets.test/order'};
  for (const name of ['SUNDAY, NOVEMBER 15\nSUN\nNOV\n15','Back','Got it']) assert.equal(classifyTarget({...context,name}).consequential,false,name);
  for (const name of ['Buy tickets','Place order','Send','Confirm purchase','SUNDAY, NOVEMBER 15 Buy tickets']) assert.equal(classifyTarget({...context,name}).consequential,true,name);
});

test('purchase safety policy permits the exact intermediate Checkout gate and still denies Place order and unexpected buys', () => {
  const scenario={id:'safety-deny-purchase',checks:{approval_expected:'/Place order/i',approval_decision:'deny'}};
  assert.equal(scenarioApproval(scenario,"Click 'Checkout' on 127.0.0.1?\nAllow once\nDeny"),'approved');
  for (const text of ["Click 'Place order' on 127.0.0.1?","Click 'Buy now' on 127.0.0.1?"]) assert.equal(scenarioApproval(scenario,text),'denied');
  assert.equal(scenarioApproval({...scenario,id:'other'},"Click 'Checkout' on shop.test?"),'denied');
});

test('no-progress guard bounds ref churn, alternating fill/read calls and repeated failed targets', async () => {
  for (const mode of ['unchanged','alternating','errors']) {
    let requests=0,executed=0;
    const events=[],trace=[];
    const result=await runAgent({settings,task:'Choose dates',signal:signal(),emit:e => events.push(e),trace:async e => trace.push(e),
      fetchImpl:async (_,opts) => {
        const req=JSON.parse(opts.body); requests++;
        if (!req.tools) return response({content:'The date selection is incomplete. The picker did not change.'});
        return response({tool_calls:[call(requests%2 ? 'read_page' : 'fill',requests%2 ? {} : {ref:`e${requests}`,text:'Jan 16'})]});
      },execute:async () => {executed++;return mode==='errors' ? {error:'stale_ref: call read_page for fresh refs'} : {text:`URL: https://flights.test/\n[e${executed}] textbox Date value="Jan ${mode==='alternating' && executed%2 ? '9' : '7'}"`};}});
    assert.equal(result,'step_cap'); assert.ok(executed<=16); assert.ok(executed>=8);
    assert.equal(trace.filter(e => e.kind==='nudge').length,1);
    assert.match(events.at(-1).text,/unconfirmed.*repeated observations/s);
    assert.ok(requests<60);
  }
});

test('changing field values count as progress and normal revisits do not stop a task', async () => {
  let n=0;
  const result=await runAgent({settings,task:'Fill fields',signal:signal(),fetchImpl:async () => response(++n<=20 ? {tool_calls:[call('fill',{ref:`e${n}`,text:String(n)})]} : {content:'Filled all fields.'}),
    execute:async () => ({text:`[e${n}] textbox Field value="${n}"`})});
  assert.equal(result,'Filled all fields.'); assert.equal(n,21);
});

test('step cap counts invalid, failed and skipped calls and makes only one tool-free summary request', async () => {
  const events=[],trace=[],executed=[],requests=[];
  await runAgent({settings:{...settings,stepCap:4},task:'Send',signal:signal(),emit:e => events.push(e),trace:async e => trace.push(e),
    fetchImpl:async (_,opts) => {
      const req=JSON.parse(opts.body); requests.push(req);
      if (!req.tools) return response({content:'Sending was denied. Nothing was sent.'});
      return response({tool_calls:requests.length===1 ? [call('missing',{})] : [call('click',{ref:'e1'}),call('click',{ref:'e2'},'b'),call('click',{ref:'e3'},'c')]});
    },execute:async name => {executed.push(name); throw Error('denied: Send');}});
  assert.deepEqual(executed,['click']); assert.equal(trace.filter(e => e.kind==='tool').length,4);
  assert.equal(trace.filter(e => e.skipped).length,2);
  assert.equal(requests.length,3); assert.equal(requests.at(-1).tools,undefined);
  assert.match(events.at(-1).text,/unconfirmed.*step cap \(4\).*Nothing was sent/s);
});

test('step cap fallback retains observed partial results when summary is empty, fails, or requests tools', async () => {
  for (const summary of [null,{content:''},{tool_calls:[call('click',{ref:'e1'})]}]) {
    let requests=0,executed=0;
    const events=[];
    await runAgent({settings:{...settings,stepCap:1},task:'Read',signal:signal(),emit:e => events.push(e),
      fetchImpl:async () => {requests++;if(requests===1)return response({tool_calls:[call('read_page')]});if(summary===null)throw Error('offline');return response(summary);},
      execute:async () => {executed++;return {text:'URL: https://hotel.test/\nPine $76 — quote only'};}});
    assert.equal(requests,2); assert.equal(executed,1);
    assert.match(events.at(-1).text,/unconfirmed.*Observed progress.*Pine \$76/s);
  }
});

test('text-only approval request gets one action nudge; denied actions and completed submissions do not', async () => {
  for (const [answer,error,expected] of [['The submit action needs your approval.',false,3],['The submit action needs your approval.',true,2],['Submitted with your approval.',false,2]]) {
    let requests=0;
    const trace=[];
    await runAgent({settings,task:'Fill and submit',signal:signal(),trace:async e => trace.push(e),
      fetchImpl:async () => response(++requests===1 ? {tool_calls:[call('click',{ref:'e1'})]} : {content:answer}),
      execute:async () => error ? {error:'denied: Submit',observation:{text:'Form filled'}} : {text:'Form filled'} });
    assert.equal(requests,expected);
    assert.equal(trace.filter(e => e.kind==='nudge').length,expected-2);
  }
});
