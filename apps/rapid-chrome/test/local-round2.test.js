import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand, mutationQuiet, fillNative, suggestionMatches } from '../src/cdp.js';
import { classifyTarget } from '../src/safety.js';
import { effectReport } from '../src/effects.js';
import { outlineAX } from '../src/ax.js';
import { EvidenceLedger, SemanticProgress, compactMessages, contextBudget, estimateTokens } from '../src/task-state.js';
import { runAgent } from '../src/agent.js';
const settings = {base:'http://127.0.0.1:8000',model:'qwen',endpointMode:'local'};
const signal = () => new AbortController().signal;
const button = {tag:'BUTTON',type:'button',role:'button',connected:true};
const browse = {...button,landmark:'nav Search restaurants',heading:'Find restaurants'};

test('r10 preparatory fixtures require real browse context; saved/account/payment/opaque writes and unknown stay gated', () => {
  const fixtures = [
    {...browse,name:'Location Sunnyvale, CA'},
    {...browse,name:'San Francisco',heading:'Choose a location',choiceContext:true},
    {...button,name:'San Francisco Bay Area',landmark:'main',heading:'Shop groceries by region',choiceContext:true},
    {...button,name:'current address: 94303. Or select a new address.',heading:'Shop groceries by delivery address'},
    {...button,name:'65w usb-c wall charger',heading:'Search suggestions',suggestionQuery:'65w usb-c wall charger'},
    {...button,name:'Go',type:'submit',method:'get',action:'https://amazon.test/s',formName:'Search'},
    {...button,name:'All filters (1)',heading:'Hotel search results'},
    {...button,name:'Main navigation menu',landmark:'nav navigation'},
    {...button,name:'Time',heading:'Search restaurants'},
    {...button,name:'Party size',heading:'Search restaurants'},
    {tag:'INPUT',type:'range',role:'slider',name:'Maximum price',heading:'Search filters'}
  ];
  for (const f of fixtures) {
    assert.equal(classifyTarget(f).consequential,false,JSON.stringify(f));
    for (const bad of [{heading:'Saved account addresses'},{action:'https://shop.test/profile/address',method:'post'},{paymentForm:true},{sensitiveForm:true}]) {
      assert.equal(classifyTarget({...f,...bad}).consequential,true,JSON.stringify({...f,...bad}));
    }
  }
  for (const name of ['San Francisco','Location Sunnyvale, CA','All filters (1)','Go somewhere','Unknown']) assert.equal(classifyTarget({...button,name}).consequential,true,name);
  for (const name of ['Save address','Confirm purchase','Reserve table','Book now','Delete account','Send']) assert.equal(classifyTarget({...browse,name}).consequential,true,name);
});

test('target context is captured by production inspection at classification and trace bounds omit form values', async () => {
  const attributes = {'aria-label':'Location Sunnyvale, CA'};
  const scope = {tagName:'NAV',getAttribute:n => n === 'role' ? 'navigation' : '',querySelector:()=>({textContent:'Search restaurants'})};
  const e = {tagName:'BUTTON',type:'button',id:'location',value:'',textContent:'Location Sunnyvale, CA',isConnected:true,
    getAttribute:n => attributes[n] || '',closest:selector => selector.includes('section') ? scope : null,
    ownerDocument:{location:{href:'https://resy.test/'},querySelectorAll:()=>[],getElementById:()=>null}};
  const hand = new BrowserHand({get:async()=>({url:'https://resy.test/'})},{settings,approve:async()=>assert.fail('browse approval')});
  hand.call=async(_id,_object,fn)=>runInNewContext(`(${fn}).call(e)`,{e});
  const info=await hand.gateTarget(1,{objectId:'target',role:'button'},'click');
  assert.equal(info.tag,'BUTTON');assert.equal(info.heading,'Search restaurants');assert.match(info.landmark,/navigation/);
  assert.equal(hand.lastTargetContext.name,'Location Sunnyvale, CA');assert.equal(hand.lastTargetContext.method,'');
  assert.ok(JSON.stringify(hand.lastTargetContext).length<2000);assert.equal(hand.lastTargetContext.values,undefined);
});

test('location denial persists across refs, label changes, fill and alternate controls without a second approval', async () => {
  let approvals=0;
  const hand = new BrowserHand({get:async()=>({url:'https://shop.test/'})},{settings,approve:async()=>{approvals++;return false;}});
  hand.inspect=async()=>({...button,name:'Change location',heading:'Saved account addresses'});
  await assert.rejects(hand.gateTarget(1,{},'click'),/^Error: denied:/);
  hand.inspect=async()=>({...browse,name:'ZIP',id:'postal'});
  await assert.rejects(hand.gateTarget(1,{},'fill',{fill:true}),/already denied/);
  hand.inspect=async()=>({...browse,name:'Select location'});
  await assert.rejects(hand.gateTarget(1,{},'click'),/already denied/);
  assert.equal(approvals,1);
});

async function simulateSettle({loadingUntil=0,switchAt=0,pending=false}) {
  let time=0,next=0,disconnects=0;const timers=new Map();
  const input={value:'Sotto Mare',matches:()=>true,getAttribute:()=> 'true'};
  const option={getClientRects:()=>[{}],get innerText(){return time<switchAt ? 'María Corona' : 'Sotto Mare';},get textContent(){return this.innerText;}};
  const doc={activeElement:input,readyState:'complete',querySelectorAll:selector=>selector.startsWith('[aria-busy') ? time<loadingUntil ? [option] : [] : pending ? [] : [option]};
  const promise=runInNewContext(`(${mutationQuiet})('Sotto Mare')`,{document:doc,getComputedStyle:()=>({visibility:'visible'}),Date:{now:()=>time},Promise,
    MutationObserver:class{observe(){}disconnect(){disconnects++;}},setTimeout:(fn,ms)=>{timers.set(++next,{fn,at:time+ms});}});
  while(timers.size && time<3000){const [id,event]=[...timers].sort((a,b)=>a[1].at-b[1].at)[0];timers.delete(id);time=event.at;event.fn();}
  return {result:await promise,time,disconnects};
}
test('Resy delayed suggestions and Seoul loading settle bounded; stale suggestion cannot be clicked', async () => {
  const settled=await simulateSettle({loadingUntil:700,switchAt:900});
  assert.equal(settled.result.ready,true);assert.ok(settled.time>=1200);assert.equal(settled.disconnects,1);
  const timeout=await simulateSettle({pending:true});assert.equal(timeout.result.timed_out,true);assert.ok(timeout.time<=2600);
  assert.equal(suggestionMatches('Sotto Mare San Francisco','Sotto Mare'),true);assert.equal(suggestionMatches('María Corona','Sotto Mare'),false);
  const hand = new BrowserHand({}, {settings,approve:async()=>assert.fail('stale option approval')});
  hand.inspect=async()=>({...button,name:'María Corona',suggestionQuery:'Sotto Mare'});
  await assert.rejects(hand.gateTarget(1,{role:'option'},'click'),/suggestions_pending/);
});

test('Fandango/Instacart frame loss refreshes main frame and refs once, observations only', async () => {
  const hand=new BrowserHand({}, {settings,approve:async()=>true});hand.active=1;hand.tabs.set(1,{refs:new Map([['e1',{}]]),epoch:0,attached:true});
  let reads=0,trees=0;hand.send=async(_id,method)=>{assert.equal(method,'Page.getFrameTree');trees++;return {frameTree:{frame:{id:'new-main'}}};};hand.settle=async()=>{};
  hand.executeCore=async name=>{assert.equal(name,'read_page');if(++reads===1)throw Error('No frame for given id found');return {text:'[e2] button Current'};};
  const r=await hand.freshRead({},signal());assert.match(r.text,/e2/);assert.equal(reads,2);assert.equal(trees,1);assert.equal(hand.tabs.get(1).mainFrame,'new-main');assert.equal(hand.tabs.get(1).refs.size,0);
  hand.executeCore=async()=>{throw Error('No frame for given id found');};reads=0;
  await assert.rejects(hand.freshRead({},signal()),/No frame/);assert.equal(trees,2);
});

test('ledger retains Seoul strict price/date/source facts, mutations, blockers and denied goals across complete-round compaction', () => {
  const ledger=new EvidenceLedger('Seoul Dec 1–3, three hotels under $150, rating >=4.3','2026-11-15');
  const text='URL: https://hotels.test/seoul\nheading "Dec 1–3"\nProduct: LOTTE City Guro | $104 | 4.3 stars href="https://hotels.test/lotte"\nProduct: Vista | $150 | 4.5 stars';
  ledger.observe('read_page',{}, {text});ledger.observe('click',{ref:'e8'}, {effect:{state:'changed',counters:[{after:'Cart 1'}]}});
  ledger.observe('click',{}, {error:'denied: location change'});ledger.observe('read_page',{}, {error:'Access denied'});
  const messages=[{role:'system',content:'Rules'},{role:'user',content:'task'}];
  for(let i=0;i<10;i++)messages.push({role:'assistant',tool_calls:[{id:`a${i}`},{id:`b${i}`}]},{role:'tool',tool_call_id:`a${i}`,content:'a'},{role:'tool',tool_call_id:`b${i}`,content:'b'});
  for(const [profile,rounds] of [['compact',3],['standard',6]]){
    const r=compactMessages(messages,ledger.task,ledger,contextBudget(profile).rounds);
    assert.equal(r.filter(m=>m.tool_calls).length,rounds);
    assert.match(r[2].content,/under \$150 excludes \$150/);assert.match(r[2].content,/LOTTE City Guro/);assert.match(r[2].content,/denied: location/);assert.match(r[2].content,/Access denied/);
    for(let i=0;i<r.length;i++)if(r[i].role==='tool')assert.ok(r.slice(0,i).some(a=>a.tool_calls?.some(c=>c.id===r[i].tool_call_id)));
  }
  const fact=JSON.parse(ledger.render()).candidateFacts.find(f=>f.observed.includes('LOTTE'));
  assert.equal(fact.url,'https://hotels.test/seoul');assert.equal(fact.observed,text.split('\n')[2]);assert.ok(fact.observedAt);
  assert.equal(contextBudget('compact').threshold,16000);assert.equal(contextBudget('standard').threshold,48000);
  assert.equal(contextBudget('compact',20000).threshold,13000);assert.equal(contextBudget('standard',20000).threshold,15000);
});

test('proactive compact inference refreshes refs, retains task/denials and never replays a mutation', async () => {
  const requests=[],executed=[],trace=[];let n=0;
  const answer=await runAgent({settings,task:'Inspect Seoul under $150 Dec 1–3',signal:signal(),trace:async e=>trace.push(e),
    fetchImpl:async(_url,opts)=>{const req=JSON.parse(opts.body);requests.push(req);return {ok:true,json:async()=>({choices:[{message:++n<=7 ? {tool_calls:[{id:`c${n}`,function:{name:n===1?'click':'read_page',arguments:JSON.stringify(n===1?{ref:'e1'}:{query:`hotel${n}`})}}]} : {content:'Verified partial Seoul evidence.'}}]})};},
    execute:async(name,args)=>{executed.push({name,args});return name==='click' ? {error:'denied: change location'} : {text:`URL: https://hotels.test/seoul\nProduct: hotel ${n} $104 4.3 ${'supported text '.repeat(1200)}\n[e999] button Current`};}});
  assert.equal(answer,'Verified partial Seoul evidence.');assert.ok(trace.some(e=>e.kind==='compaction'));assert.equal(executed.filter(c=>c.name==='click').length,1);
  assert.ok(executed.some(c=>c.name==='read_page' && !c.args.query));
  const compact=requests.find(r=>r.messages.some(m=>m.content?.includes('Compacted evidence ledger')));assert.ok(compact);
  assert.match(JSON.stringify(requests.at(-1)),/denied: change location/);assert.match(JSON.stringify(requests.at(-1)),/e999/);assert.match(JSON.stringify(requests.at(-1)),/under \$150 excludes \$150/);
  assert.ok(estimateTokens(compact)<=16000,`compacted request estimated ${estimateTokens(compact)} tokens`);
});

test('semantic Uber/cart and Instacart/modal retries count across intervening reads; Resy new queries without candidates stop', () => {
  const sem=new SemanticProgress();
  for(let i=1;i<=4;i++){
    sem.record('read_page',{}, {text:`URL: https://uber.test/store\n[e${i}] button "Cart"`});
    const result=sem.record('click',{ref:`e${i}`},{url:'https://uber.test/store',effect:{state:'unchanged',verified:[{key:'cart',before:'1',after:'1',status:'unchanged'}]}});
    if(i===1)assert.equal(result,null);else assert.equal(result.count,i);
    if(i===4)assert.equal(result.stop,true);
  }
  const queries=new SemanticProgress();
  queries.record('read_page',{query:'Cotogna'},{text:'URL: https://resy.test/\nStaticText "No availability"'});
  let result;
  for(const query of ['Sotto Mare','La Folage','Italian','See all'])result=queries.record('read_page',{query},{text:'URL: https://resy.test/\nStaticText "No availability"'});
  assert.equal(result.stop,true);
  assert.equal(queries.record('read_page',{query:'Other'},{text:'URL: https://resy.test/\nProduct: New candidate 7 PM Nov 16'}),null);
  const denied=new SemanticProgress();for(let i=1;i<=4;i++)result=denied.record('click',{ref:`e${i}`},{url:'https://resy.test/',target_context:{name:i%2?'Location':'Region'},error:'denied: location'});
  assert.equal(result.stop,true);
});

function tree() {
  const nodes=[];const node=(role,name,children=[])=>{const n={nodeId:String(nodes.length+1),backendDOMNodeId:nodes.length+1,role:{value:role},name:{value:name},childIds:children.map(c=>c.nodeId)};nodes.push(n);return n;};
  return {nodes,node};
}
test('Seoul query removes marker collection and chrome while Kyoto multiword fallback retains contextual prose', () => {
  const {nodes,node}=tree();
  const list=node('list','Hotel results',[node('article','',[node('heading','LOTTE City Guro'),node('StaticText','$104, rating 4.3'),node('link','Details')])]);
  node('RootWebArea','',[node('navigation','',[node('link','LOTTE promotion')]),node('main','',[node('region','Map',[node('group','LOTTE markers',[node('StaticText','Map marker unrelated $150')])]),list])]);
  const r=outlineAX(nodes,{query:'LOTTE'});assert.match(r.text,/LOTTE City Guro/);assert.match(r.text,/104/);assert.doesNotMatch(r.text,/marker|promotion/);
  const kyoto=tree();kyoto.node('RootWebArea','',[kyoto.node('main','',[kyoto.node('paragraph','',[kyoto.node('StaticText','Tenryū-ji temple'),kyoto.node('StaticText','A Zen temple in Arashiyama with a garden.')])])]);
  const k=outlineAX(kyoto.nodes,{query:'Arashiyama attractions'});assert.equal(k.query_matched,true);assert.match(k.text,/any-term/);assert.match(k.text,/Tenryū-ji/);assert.match(k.text,/Zen temple/);
});

test('Seoul native range sets value via native setter/events and verifies unchanged sliders/cart counters', () => {
  let value='1000';const events=[];
  class Input{} Object.defineProperty(Input.prototype,'value',{get(){return value;},set(v){value=String(v);}});
  const e=new Input();Object.assign(e,{tagName:'INPUT',type:'range',min:'0',max:'1000',step:'1',isConnected:true,getClientRects:()=>[{}],closest:()=>null,dispatchEvent:e=>events.push(e.type),ownerDocument:{defaultView:{HTMLInputElement:Input}}});
  const result=runInNewContext(`(${fillNative}).call(e,'149')`,{e,Event:class{constructor(type){this.type=type;}}});
  assert.equal(result.value,'149');assert.deepEqual(events,['input','change']);
  const before={url:'https://hotels.test/',title:'Seoul',controls:[{key:'max',text:'1000'}],counters:[{key:'cart',text:'Cart 1'}],regions:[]};
  const unchanged=effectReport(before,[before]);assert.equal(unchanged.state,'unchanged');assert.equal(unchanged.verified.length,2);assert.ok(unchanged.verified.every(v=>v.status==='unchanged'));
  const changed=effectReport(before,[{...before,controls:[{key:'max',text:'149'}]}]);assert.equal(changed.state,'changed');assert.equal(changed.verified.find(v=>v.key==='max').after,'149');
  assert.equal(effectReport(before,[]).state,'unverified');
});

test('input disclosure changes allow typing, while approved snapshots, form values, target and endpoints remain strictly bound', async () => {
  const hand=new BrowserHand({}, {settings,approve:async()=>true});hand.tabs.set(1,{epoch:0});hand.current=async()=>({});
  const original={name:'Search',expanded:'false',form:'Search',action:'https://shop.test/s',method:'get',values:[{name:'q',value:''}]};
  const changed={...original,expanded:'true',form:'Search charger suggestions'};
  hand.inspect=async()=>changed;
  await hand.verifyApproval(1,{epoch:0,editingInput:true,inspectedInfo:original});
  await assert.rejects(hand.verifyApproval(1,{epoch:0,editingInput:true,approvedInfo:original}),/approval_snapshot_changed/);
  for(const mutation of [{action:'https://shop.test/pay'},{name:'Buy now'},{values:[{name:'q',value:'other'}]}]){
    hand.inspect=async()=>({...changed,...mutation});
    await assert.rejects(hand.verifyApproval(1,{epoch:0,editingInput:true,inspectedInfo:original}),/approval_snapshot_changed/);
  }
});

test('production agent reports second semantic stall in tool result and stops fourth across intervening observations', async () => {
  let round=0,actions=0;const requests=[],events=[];
  await runAgent({settings,task:'Inspect cart subtotal',signal:signal(),trace:async e=>events.push(e),
    fetchImpl:async(_url,opts)=>{const r=JSON.parse(opts.body);requests.push(r);return {ok:true,json:async()=>({choices:[{message:!r.tools ? {content:'Cart remains at one item. Subtotal unverified.'} : {tool_calls:[{id:`c${++round}`,function:{name:round%2?'read_page':'click',arguments:JSON.stringify(round%2?{}:{ref:`e${round}`})}}]}}]})};},
    execute:async name=>{
      if(name==='click')actions++;
      return {url:'https://uber.test/store',text:`URL: https://uber.test/store\n[e${round+1}] button "Cart"`,...(name==='click'?{target_context:{role:'button',name:'Cart'},effect:{state:'unchanged',verified:[{key:'cart',before:'1',after:'1',status:'unchanged'}]}}:{})};
    }});
  assert.equal(actions,4);assert.equal(requests.at(-1).tools,undefined);
  const tools=events.filter(e=>e.kind==='tool' && e.name==='click').map(e=>JSON.parse(e.text));
  assert.equal(tools[1].no_progress.count,2);assert.match(tools[1].no_progress.message,/Change strategy/);assert.equal(tools[3].no_progress.stop,true);
  assert.ok(events.some(e=>e.kind==='summary'));
});

test('unchanged blocker observations stop after four, while changed cart values allow a new strategy', () => {
  const sem=new SemanticProgress();let r;
  for(let i=1;i<=4;i++)r=sem.record('read_page',{}, {text:`URL: https://opentable.test/\nStaticText "Access Denied"`});
  assert.equal(r.stop,true);
  const cart=new SemanticProgress();
  const result=value=>({url:'https://uber.test/',target_context:{role:'button',name:'Cart'},effect:{state:'unchanged',verified:[{key:'cart',before:value,after:value,status:'unchanged'}]}});
  cart.record('click',{ref:'e1'},result('1'));assert.equal(cart.record('click',{ref:'e2'},result('1')).count,2);
  assert.equal(cart.record('click',{ref:'e3'},result('2')),null);
});

test('metadata frame recovery after navigation observes once and never dispatches navigation again', async () => {
  const hand=new BrowserHand({}, {settings,approve:async()=>true});hand.tabs.set(1,{epoch:0,refs:new Map(),attached:true});hand.active=1;
  let current=0,navigations=0,trees=0;
  hand.current=async()=>{if(++current===2)throw Error('No frame for given id found');return {id:1,tab:{url:'https://site.test/search',title:'Search'},t:hand.tabs.get(1)};};
  hand.site=async url=>url;hand.send=async(_id,method)=>{if(method==='Page.navigate')navigations++;else if(method==='Page.getFrameTree')trees++;else assert.fail(method);return {frameTree:{frame:{id:'new'}}};};
  await hand.executeCore('navigate',{url:'https://site.test/search?q=restaurant'},signal());
  assert.equal(navigations,1);assert.equal(trees,1);assert.equal(current,3);
});

test('new result cards after repeated Show more are progress even if watched cart/dialog/controls stay unchanged', () => {
  const sem=new SemanticProgress();
  for(let i=0;i<8;i++){
    const r=sem.record('click',{ref:`e${i}`},{url:'https://hotels.test/',target_context:{role:'button',name:'Show more'},effect:{state:'unchanged',verified:[]},text:`URL: https://hotels.test/\nProduct: New qualifying hotel ${i} $104 rating 4.3`});
    assert.equal(r,null);
  }
  // A no-results status alongside a newly observed candidate is also progress.
  for(let i=0;i<8;i++)assert.equal(sem.record('read_page',{query:'hotels'},{text:`URL: https://hotels.test/\nStaticText "No results in old area"\nProduct: Other hotel ${i} $117 rating 4.5`}),null);
});
