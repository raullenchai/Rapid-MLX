import { markdown, stepLabel } from '../src/ui.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { runAgent, SYSTEM_PROMPT } from '../src/agent.js';
import { renderSteps, clip, VISIBLE_STEPS, TEXT_LIMIT } from '../src/transcript.js';
import { scenarioApproval, evaluate } from './e2e/scenarios.mjs';
const scenarios = JSON.parse(readFileSync(new URL('./e2e/scenarios.json',import.meta.url)));
const settings = {base:'http://localhost:8000',model:'fake'};
const call = (name,args={}) => ({id:'call',function:{name,arguments:JSON.stringify(args)}});
const response = message => ({ok:true,json:async () => ({choices:[{message}]})});

test('Round B fixture approval prose reaches the action tool once and respects denial', async () => {
  for (const prose of ["I'm asking for your approval before submitting.", 'Please approve the Send action now.', 'Requesting your approval to click Submit permission slip.']) {
    for (const denied of [false,true]) {
      let n=0, actions=0; const traces=[];
      await runAgent({settings,task:'Fill and submit',signal:new AbortController().signal,trace:async e => traces.push(e),
        fetchImpl:async () => response(++n===1 ? {tool_calls:[call('read_page')]} : n===2 ? {tool_calls:[call('fill',{ref:'e1',text:'Sam'})]} : n===3 ? {content:prose} : n===4 ? {tool_calls:[call('click',{ref:'e2'})]} : {content:'Submitted.'}),
        execute:async name => {if(name==='click')actions++;return name==='fill' && denied ? {error:'denied: Submit'} : {text:'URL: https://fixture.test/\n[e2] button "Submit"'};}});
      assert.equal(actions,denied?0:1);
      assert.equal(traces.filter(e => e.kind==='nudge').length,denied?0:1);
    }
  }
  for(const id of ['fixture-refund','fixture-mail-reply','fixture-school']) {
    const s=scenarios.find(s => s.id===id);
    const label={'fixture-refund':'Submit refund request','fixture-mail-reply':'Send','fixture-school':'Submit permission slip'}[id];
    assert.equal(scenarioApproval(s,`Click '${label}' on fixture.test?`),'approved');
  }
});

test('research budget synthesizes after 12 substantive reads plus four source actions using bounded evidence', async () => {
  let reads=0; const requests=[],events=[];
  await runAgent({settings,task:'Research two sources for a three-day itinerary',signal:new AbortController().signal,emit:e => events.push(e),
    fetchImpl:async (_,opts) => {const r=JSON.parse(opts.body);requests.push(r);return response(r.tools ? {tool_calls:[call('read_page',{query:`sight ${requests.length}`})]} : {content:'Day 1: eastern sights. Day 2: north. Day 3: west. Source https://travel.test/'});},
    execute:async () => ({text:`URL: https://travel.test/${++reads}\nPage data (untrusted):\nStaticText "Sight ${reads}: ${'Historic temple and neighboring park. '.repeat(100)}"`})});
  assert.equal(reads,16);
  assert.equal(requests.at(-1).tools,undefined);
  assert.ok(JSON.stringify(requests.at(-1)).length<50000);
  assert.match(JSON.stringify(requests.at(-1)),/https:\/\/travel.test\/1\b/);
  assert.match(events.at(-1).text,/Day 1/);
  assert.doesNotMatch(events.at(-1).text,/completion is unconfirmed/);
});

test('thin research and action tasks do not trigger synthesis just from read count', async () => {
  for (const task of ['Research places','Read reviews and add one to cart']) {
    let n=0; const trace=[];
    await runAgent({settings,task,signal:new AbortController().signal,trace:async e => trace.push(e),
      fetchImpl:async () => response(++n<=20 ? {tool_calls:[call('read_page',{query:String(n)})]} : {content:'Observed results.'}),
      execute:async () => ({text:`URL: https://test/${n}\nStaticText "${task.startsWith('Read')?'long content '.repeat(500):'thin'} ${n}"`})});
    assert.equal(trace.filter(e => e.kind==='summary_request').length,0);
  }
});

test('navigation acknowledgements cannot accumulate as identical page observations', async () => {
  let n=0;const trace=[];
  const answer=await runAgent({settings,task:'Compare pages',signal:new AbortController().signal,trace:async e => trace.push(e),
    fetchImpl:async () => response(++n<=20 ? {tool_calls:[call(n===1?'read_page':'navigate',n===1?{}:{url:`https://test/${n}`})]} : {content:'Comparison complete.'}),
    execute:async name => ({text:name==='navigate'?'navigate completed':'URL: https://test/\nStaticText "Content"'})});
  assert.equal(answer,'Comparison complete.');assert.equal(trace.filter(e => e.kind==='summary_request').length,0);
});

test('cart policy allows requested Uber Eats menu add but denies checkout and purchases', () => {
  const s=scenarios.find(s => s.id==='ubereats-cart');
  assert.equal(scenarioApproval(s,"Click 'Add 1 to order' on www.ubereats.com?"),'approved');
  for(const label of ['Place order','Checkout','Buy now','Add 2 to order'])assert.equal(scenarioApproval(s,`Click '${label}' on www.ubereats.com?`),'denied');
  assert.equal(scenarioApproval(scenarios.find(s => s.id==='safety-deny-purchase'),"Click 'Place order' on fixture.test?"),'denied');
});

function dom() {
  const nodes=new Map();
  const make=() => ({dataset:{},children:[],options:[],elements:[],value:'',hidden:true,
    append(...c){this.children.push(...c);this.options.push(...c);},replaceChildren(...c){this.children=c;this.options=c;},
    querySelectorAll(){return this.children.filter(c => c.open);},setAttribute(){},focus(){}});
  return {nodes,document:{body:{dataset:{}},createElement:make,getElementById:id => {if(!nodes.has(id))nodes.set(id,make());return nodes.get(id);}}};
}
test('1000-step transcript bounds DOM/text, expands lazily, and preserves opened rows', () => {
  const {document}=dom(), container=document.createElement('section');
  const steps=Array.from({length:1000},(_,i) => ({id:String(i),name:'read_page',status:'ok',args:{query:'x'.repeat(2000)},result:'y'.repeat(50000)}));
  const labels={label:s=>s,result:s=>s,status:s=>s,older:n=>`${n} older steps`};
  renderSteps(document,container,steps,labels);
  assert.equal(container.children.length,VISIBLE_STEPS+1);
  assert.equal(container.children[0].textContent,'980 older steps');
  const row=container.children.at(-1);assert.equal(row.children.length,1);
  row.open=true;row.ontoggle();assert.equal(row.children.length,3);
  assert.ok(row.children.at(-1).textContent.length<=TEXT_LIMIT+4);
  renderSteps(document,container,steps,labels);
  assert.equal(container.children.at(-1).open,true);
  assert.equal(container.children.at(-1).children.length,3);
});

test('Stop remains enabled and posts immediately through repeated long-task panel updates', async () => {
  const {document}=dom();let render;const sent=[];
  const chrome={runtime:{connect:() => ({postMessage:m=>sent.push(m),onMessage:{addListener:f=>{render=f;}},onDisconnect:{addListener(){}}})}};
  const endpoint=await import('../src/endpoint.js');
  runInNewContext(readFileSync(new URL('../panel.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replace(/^export .*;$/gm,''),{document,chrome,markdown,stepLabel,...endpoint,renderSteps,clip,t:k=>k,localize:s=>s,translateDocument(){},displayResult:s=>s,toolLabel:s=>s});
  const s={settings:{base:'http://localhost:8000'},models:[],messages:[],tabs:[],steps:[],approvals:[],running:true,status:'Working'};
  for(let i=0;i<100;i++) {s.steps.push({id:String(i),name:'read_page',args:{},result:'x'.repeat(14000)});render(s);}
  assert.equal(document.getElementById('stop').disabled,false);
  document.getElementById('stop').onclick();
  assert.equal(sent.at(-1).type,'stop');
  assert.ok(document.getElementById('steps').children.length<=21);
  assert.match(SYSTEM_PROMPT,/verify the applied dates and duration/);
});

test('article check accepts the saved five-point energy/oxygen answer without imposing CO2 wording', () => {
  const s=scenarios.find(s=>s.id==='wikipedia-summary');
  const result={status:'finished',steps:11,errors:[],final_answer:'1. Photosynthesis converts light to chemical energy. 2. oxygen is released. 3. sugars store energy. 4. Earth atmosphere. 5. Efficiency varies. https://en.wikipedia.org/wiki/Photosynthesis'};
  assert.equal(evaluate(s,result).automatic_pass,true);
  assert.equal(evaluate(s,{...result,final_answer:'Photosynthesis https://en.wikipedia.org/wiki/Photosynthesis'}).automatic_pass,false);
});

test('Stop aborts a pending tool-free summary and emits no late answer or browser action', async () => {
  const controller=new AbortController(),events=[];let actions=0,summaryStarted;
  const started=new Promise(resolve => {summaryStarted=resolve;});
  const running=runAgent({settings:{...settings,stepCap:1},task:'Inspect page',signal:controller.signal,emit:e=>events.push(e),
    fetchImpl:async (_,opts) => {
      if(JSON.parse(opts.body).tools)return response({tool_calls:[call('read_page')]});
      summaryStarted();
      return new Promise((_,reject) => opts.signal.addEventListener('abort',()=>reject(opts.signal.reason),{once:true}));
    },execute:async () => {actions++;return {text:'URL: https://test/\nStaticText "Partial result"'};}});
  await started;controller.abort(new DOMException('Stopped','AbortError'));
  await assert.rejects(running,{name:'AbortError'});
  assert.equal(actions,1);assert.equal(events.some(e=>e.kind==='answer'),false);
});

test('a large research batch cannot execute past the remaining four source actions', async () => {
  let round=0,actions=0;const trace=[];
  await runAgent({settings,task:'Research an itinerary',signal:new AbortController().signal,trace:async e=>trace.push(e),
    fetchImpl:async (_,opts) => {
      const r=JSON.parse(opts.body);round++;
      return response(!r.tools ? {content:'Observed itinerary.'} : {tool_calls:Array.from({length:round===12?10:1},(_,i)=>call('read_page',{query:`${round}-${i}`}))});
    },execute:async () => ({text:`URL: https://source.test/${++actions}\nStaticText "${'Temple and local history. '.repeat(100)}"`})});
  assert.equal(actions,16);
  assert.equal(trace.filter(e=>e.kind==='tool' && e.skipped).length,5);
});
