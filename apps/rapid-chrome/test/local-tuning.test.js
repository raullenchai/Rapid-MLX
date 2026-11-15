import test from 'node:test';
import assert from 'node:assert/strict';
import { runAgent, assembleMessages, SYSTEM_PROMPT } from '../src/agent.js';
import { requestOptions, pageProfile, endpointError } from '../src/endpoint.js';
import { outlineAX } from '../src/ax.js';
import { BrowserHand } from '../src/cdp.js';
import { detectChallenge } from '../src/challenge.js';
import { TOOLS } from '../src/tools.js';
const settings = {base:'http://127.0.0.1:8000',endpointMode:'local',model:'qwen'};
const call = (name,args={}) => ({id:'c',function:{name,arguments:JSON.stringify(args)}});
const ok = message => ({ok:true,json:async()=>({choices:[{message}]})});
async function run(task,replies,results=[{text:'Form ready; [e19] button Submit'}],extra={}) {
  const requests=[],events=[],executed=[];
  const answer=await runAgent({settings,task,currentTab:{id:9,shared:true},signal:new AbortController().signal,...extra,
    trace:async e=>events.push(e),fetchImpl:async (_,o)=>{requests.push(JSON.parse(o.body));assert.ok(requests.length<=6);const r=replies.shift();return r?.ok !== undefined ? r : ok(r || {content:'Done'});},
    execute:async (name,args)=>{executed.push({name,args});const r=results.shift();if(r instanceof Error)throw r;return r || {text:'Saved'};}});
  return {answer,requests,events,executed};
}
test('P1 precedes chat and stays within original prompt budget; grant protects all composer metadata',()=>{
  assert.ok(SYSTEM_PROMPT.length<=4959);
  assert.ok(SYSTEM_PROMPT.indexOf('first call read_page')<SYSTEM_PROMPT.indexOf('greetings'));
  const tab={id:987654,title:'private-title',url:'https://mail.test/?token=private-token'};
  const before=JSON.stringify(assembleMessages([],'hello',tab));
  assert.doesNotMatch(before,/987654|private-title|private-token/);
  const after=JSON.stringify(assembleMessages([],'this email',{...tab,shared:true}));
  assert.match(after,/987654/);assert.match(after,/private-title/);assert.match(after,/private-token/);
  assert.match(before,/Browser date \(ISO\): \d{4}-\d{2}-\d{2}/);
});
test('mail paste refusal recovers once, hello and grounded follow-up stay zero tool',async()=>{
  const r=await run('Summarize this email',[{content:'Please paste the content'},{tool_calls:[call('read_page')]},{content:'The form is ready'}]);
  assert.deepEqual(r.executed.map(c=>c.name),['read_page']);assert.equal(r.events.filter(e=>e.kind==='nudge').length,1);
  for(const task of ['hello','What was that price?'])assert.equal((await run(task,[{content:'Please paste the content'}])).requests.length,1);
});
test('query miss from charger trace is bounded, actionable, typed; matching products keep card evidence',()=>{
  const nodes=Array.from({length:200},(_,i)=>({nodeId:''+i,backendDOMNodeId:i+1,role:{value:i===0?'searchbox':'StaticText'},name:{value:i===0?'Search Amazon':'Anker charger $29.99 4.7 stars '+i}}));
  const result=outlineAX(nodes,{query:'65W USB-C wall adapter'});
  assert.equal(result.query_matched,false);assert.ok(result.text.length<=2000);assert.equal(result.truncated,true);assert.ok(result.omitted_entries>0);assert.match(result.text,/Search Amazon/);assert.match(result.text,/did not search the website/);
  assert.equal(outlineAX(nodes,{query:'Anker'}).query_matched,true);
  assert.equal(pageProfile(settings),'compact');assert.equal(pageProfile({...settings,pageDetail:'standard'}),'standard');assert.equal(pageProfile({endpointMode:'custom'}),'standard');
});
test('school prose approval recovery is single, gated by requested action, denial and human handover',async()=>{
  for(const content of ['I request approval. Shall I submit?', 'May I send?', '请求批准，可以提交吗']){
    const r=await run('Fill and submit this form',[{tool_calls:[call('fill',{ref:'e1',text:'JFK',choose:'JFK'})]},{content},{tool_calls:[call('click',{ref:'e19'})]},{content:'Saved'}]);
    assert.deepEqual(r.executed.map(c=>c.name),['fill','click']);assert.equal(r.events.filter(e=>e.kind==='nudge').length,1);
  }
  for(const result of [{error:'denied: submission'},{error:'needs_human: check'},{human_check:true,text:'Blocked'}]){
    const r=await run('Submit this form',[{tool_calls:[call('read_page')]},{content:'I request approval. Shall I submit?'}],[result]);assert.equal(r.requests.length,2);
  }
  assert.equal((await run('Should I submit this form?',[{tool_calls:[call('read_page')]},{content:'Shall I submit?'}])).requests.length,2);
});
test('missing fill text has actionable schema repair and runs no action',async()=>{
  const r=await run('Fill JFK',[{tool_calls:[call('fill',{ref:'e19',choose:'JFK'})]},{content:'Unable'}]);
  assert.equal(r.executed.length,0);assert.match(r.requests[1].messages.find(m=>m.role==='tool').content,/No browser action ran.*text.*JFK/);
  assert.match(TOOLS.find(t=>t.function.name==='fill').function.description,/text are always required/);
});
test('frame loss after open preserves opened tab identity without duplicate creation',async()=>{
  const hand=new BrowserHand({}, {settings,approve:async()=>true});hand.tabs.set(42,{});
  let opens=0;hand.executeCore=async()=>{opens++;return {tabId:42,text:'Opened tab'};};hand.settle=async()=>{};hand.freshRead=async()=>{throw new Error('No frame for given id found');};hand.meta=async()=>({url:'https://target.test/',tabId:42});hand.check=()=>{};
  const result=await hand.execute('open_tab',{url:'https://target.test/'},new AbortController().signal);
  assert.equal(result.tabId,42);assert.equal(result.observation_failed,true);assert.match(result.recovery,/do not open it again/);assert.equal(opens,1);
});
test('approval snapshot payload categorizes form mismatch and preserves blocking',async()=>{
  const hand=new BrowserHand({}, {settings,approve:async()=>true});hand.tabs.set(1,{epoch:0});hand.current=async()=>({});hand.inspect=async()=>({values:[{value:'changed'}]});
  await assert.rejects(hand.verifyApproval(1,{epoch:0,approvedInfo:{values:[{value:'approved'}]}}),e=>e.payload?.type==='approval_snapshot_changed' && e.payload.mismatch==='form' && e.payload.action_executed===false && e.payload.approval_required);
});
test('embedded challenge checkbox and Continue are blocked, normal consent is permitted',async()=>{
  assert.equal(detectChallenge({frames:[{visible:true,url:'https://geo.captcha-delivery.com/captcha/'}]})?.kind,'human_check');
  const hand=new BrowserHand({}, {settings,approve:async()=>true});
  for(const name of ['checkbox','Continue']){hand.inspect=async()=>({connected:true,challengeControl:true,name});await assert.rejects(hand.gateTarget(1,{},'click'),/human_check/);}
  assert.equal(detectChallenge({nodes:[{role:{value:'checkbox'},name:{value:'I consent to the school trip'}}]}),null);
});
test('local params, remote kwargs isolation and final budgets',()=>{
  assert.deepEqual(requestOptions(settings),{chat_template_kwargs:{enable_thinking:false},temperature:0.2,max_tokens:1024});
  assert.equal(requestOptions(settings,true).max_tokens,2048);assert.equal(requestOptions({...settings,temperature:0.7}).temperature,0.7);
  assert.deepEqual(requestOptions({base:'https://custom.test',endpointMode:'custom',reasoningEffort:'high'}),{reasoning_effort:'high'});
});
test('HTTP diagnostics bounded and redacted; known overflow/502 retry model only once, unknown/template do not retry',async()=>{
  const diagnostic=await endpointError(new Response(JSON.stringify({error:{code:'bad_template',type:'template_error',message:'Bearer private-key',detail:'api_key=secret-value'}}),{status:400,headers:{'x-request-id':'request-42'}}));
  assert.equal(diagnostic.code,'bad_template');assert.equal(diagnostic.requestId,'request-42');assert.doesNotMatch(JSON.stringify(diagnostic),/private-key|secret-value/);
  assert.ok((await endpointError(new Response('x'.repeat(50000),{status:400}))).body.length<=8192);
  for(const [status,text] of [[400,'maximum context length exceeded'],[502,'gateway']]){
    const r=await run('hello',[new Response(text,{status}),{content:'Recovered'}]);assert.equal(r.executed.length,0);assert.equal(r.requests.length,2);assert.equal(r.events[0].params.temperature,0.2);
    await assert.rejects(run('hello',[new Response(text,{status}),new Response(text,{status})]),/HTTP/);
  }
  await assert.rejects(run('hello',[new Response('template rejected',{status:400})]),/unknown_http_error/);
});
test('length retry never executes partial tool JSON and increases local budget once',async()=>{
  const truncated={ok:true,json:async()=>({choices:[{finish_reason:'length',message:{tool_calls:[call('fill',{ref:'e1'})]}}]})};
  const r=await run('hello',[truncated,{content:'Complete'}]);assert.equal(r.executed.length,0);assert.equal(r.requests[1].max_tokens,2048);
});

test('native/custom dates and autocomplete observations expose format, type, validity and unverified commit state',async()=>{
  const {runInNewContext}=await import('node:vm');
  const controls=[
    {tagName:'INPUT',type:'date',value:'2026-11-20'},
    {tagName:'INPUT',type:'text',name:'Departure date',value:'Fri, Nov 20',placeholder:'ddd, MMM D'},
    {tagName:'INPUT',type:'text',name:'Airport',value:'JFK',role:'combobox',expanded:'true'}
  ].map(c=>({...c,validity:{valid:true},closest:()=>null,getAttribute(name){return ({role:this.role,'aria-expanded':this.expanded,placeholder:this.placeholder})[name] || null;}}));
  const hand=new BrowserHand({}, {settings,approve:async()=>true});
  const state={refs:new Map(),epoch:0};hand.tabs.set(1,state);hand.current=async()=>({id:1,tab:{url:'https://flights.test/',title:'Flights'},t:state});hand.check=()=>{};
  hand.send=async(_,method)=>method==='Accessibility.getFullAXTree' ? {nodes:controls.map((c,i)=>({nodeId:''+i,backendDOMNodeId:i+1,role:{value:i===0?'Date':i===1?'textbox':'combobox'},name:{value:c.name || 'Departure'}}))} : {};
  hand.resolve=async(_,ref)=>({objectId:ref});
  hand.call=async(_,ref,fn)=>runInNewContext(`(${fn}).call(control)`,{control:controls[Number(ref.slice(1))-1],document:{forms:[],activeElement:controls[1]}});
  const result=await hand.executeCore('read_page',{},new AbortController().signal);
  assert.match(result.text,/2026-11-20.*YYYY-MM-DD.*input_type=date.*valid=true/);
  assert.match(result.text,/Fri, Nov 20.*displayed format.*placeholder="ddd, MMM D".*focused; commit unverified/);
  assert.match(result.text,/JFK.*suggestions\/picker open/);
});

test('compact default read is 6K, standard and focused read retain 12K budgets',async()=>{
  for(const [pageDetail,query,budget] of [['compact',undefined,6000],['standard',undefined,12000],['compact','price',12000]]){
    const hand=new BrowserHand({}, {settings:{...settings,pageDetail},approve:async()=>true});const state={refs:new Map(),epoch:0};
    hand.tabs.set(1,state);hand.current=async()=>({id:1,tab:{url:'https://shop.test/',title:'Shop'},t:state});hand.check=()=>{};
    hand.send=async(_,method)=>method==='Accessibility.getFullAXTree' ? {nodes:Array.from({length:500},(_,i)=>({nodeId:''+i,role:{value:'StaticText'},name:{value:'Product price $29.99 rating 4.7 stars '+i}}))} : {};
    const result=await hand.executeCore('read_page',{query},new AbortController().signal);
    assert.ok(result.text.length<=budget);assert.ok(result.text.length>budget-500);assert.equal(result.truncated,true);
  }
});

test('502 after mutation retries only inference; final synthesis uses 2048; evidence policy covers used offers and destination claims',async()=>{
  const r=await run('Add this item',[{tool_calls:[call('click',{ref:'e1'})]},new Response('gateway',{status:502}),{content:'Added'}]);
  assert.equal(r.executed.length,1);assert.deepEqual(r.requests[1],r.requests[2]);
  const final=await run('Fill this form',[{tool_calls:[call('fill',{ref:'e1',text:'Alex'})]},{content:'Partial result'}],undefined,{settings:{...settings,stepCap:1}});
  assert.equal(final.requests.at(-1).max_tokens,2048);assert.equal(final.requests.at(-1).tools,undefined);
  assert.match(SYSTEM_PROMPT,/used\/refurbished request report the observed offer's condition/);
  assert.match(SYSTEM_PROMPT,/Only claim destination-specific delivery when verified/);
  assert.match(SYSTEM_PROMPT,/Calendar price hints are not confirmed/);
});

test('overflow recovery keeps complete tool batches and original task without browser replay',async()=>{
  const r=await run('Fill and submit this form',[
    {tool_calls:[call('fill',{ref:'e1',text:'one'})]},
    {tool_calls:[call('fill',{ref:'e2',text:'two'})]},
    {tool_calls:[call('fill',{ref:'e3',text:'three'})]},
    new Response(JSON.stringify({error:{code:'context_length_exceeded',message:'Maximum context length exceeded'}}),{status:400}),
    {content:'Fields filled; submission unconfirmed'}
  ]);
  assert.equal(r.executed.length,3);
  const recovered=r.requests.at(-1).messages;
  assert.equal(recovered[1].content,'Fill and submit this form');
  for(let i=0;i<recovered.length;i++)if(recovered[i].role==='tool')assert.ok(recovered[i-1].tool_calls?.some(c=>c.id===recovered[i].tool_call_id));
});
test('scoped negation keeps the requested action; whole-task negation stays read-only',async()=>{
  const replies=()=>[{tool_calls:[call('read_page')]},{content:'I request approval to Send this reply to Alex'},{tool_calls:[call('click',{ref:'e19'})]},{content:'Sent'}];
  for(const task of ['Reply to Alex with the link. Ask me to approve Send. Do not send to anyone else.','回复 Alex 并附上链接，不要发给其他人']){
    const r=await run(task,replies());assert.equal(r.events.filter(e=>e.kind==='nudge').length,1,task);
  }
  for(const task of ['Draft a reply to Alex but do not send it.','Find the deck link. Do not reply or send anything.']){
    const r=await run(task,[{tool_calls:[call('read_page')]},{content:'I request approval to Send this reply'}]);assert.equal(r.requests.length,2,task);
  }
});
