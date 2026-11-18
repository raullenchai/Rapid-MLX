// State/markup checks in a VM. These deliberately do not claim browser layout coverage.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const elements=new Map();
function element(id='') {
  const node={id,textContent:'',innerHTML:'',value:'',hidden:false,disabled:false,readOnly:false,style:{},dataset:{},attributes:{},clientWidth:1000,clientHeight:340,scrollTop:0,scrollHeight:100,
    append(child){child.parentElement=this;},focus(){document.activeElement=this;},reset(){},querySelector(){return null;},querySelectorAll(){return [];},setAttribute(k,v){this.attributes[k]=v;},removeAttribute(k){delete this.attributes[k];}};
  if(id==='curve' || id==='result-curve'){node.clientWidth=450;node.clientHeight=180;}
  return node;
}
function get(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id);}
const document={getElementById:get,activeElement:null,title:''};
const requests=[], streams=[];
const task={title:'A reusable study',editable:'experiment.py',metric:'score',metric_label:'Score',unit:'%',direction:'max',noise:.02,seeds:[0,1],goal:.99,curve:{x:'step',x_label:'Steps',series:[{key:'train',label:'Train'},{key:'score',label:'Test'}]}};
let detail={project:{slug:'study',title:'A reusable study',model:'q35'},program:'# Goal',task,runs:[],discussion:[],ideas:[],models:['q35','q27'],default_model:'q35',rounds:10};
let testRun={id:'r1',model:'q35',active:false,paused:false};
let testEvents=[],testFiles=[],testProjects=[];
const context=vm.createContext({document,console,URLSearchParams,URL,TextDecoder,TextEncoder,setTimeout,clearTimeout,
  location:{hash:'#/',search:''},window:{addEventListener(){}},
  fetch:async(path,options)=>{
    requests.push({path,options});
    const value=path==='/api/templates' ? {templates:[{slug:'generic',...task}]} : path==='/api/projects' ? {projects:testProjects} : path.endsWith('/program') ? JSON.parse(options.body) : path.endsWith('/diff/1') ? {parent:0,diff:'--- old\n+++ new\n-VALUE = 1\n+VALUE = 2\n'} : path.includes('/files/') ? null : /\/runs\/[^/]+$/.test(path) ? {run:testRun,events:testEvents,files:testFiles} : structuredClone(detail);
    return {ok:true,json:async()=>value,text:async()=>'<script>not HTML</script>\nVALUE = 2'};
  },
  EventSource:class{constructor(path){this.path=path;streams.push(this);}close(){this.closed=true;}addEventListener(name,fn){this[name]=fn;}}
});
const run=source=>vm.runInContext(source,context);
vm.runInContext(fs.readFileSync(new URL('../web/app.js',import.meta.url),'utf8'),context);
await new Promise(resolve=>setImmediate(resolve));
function event(data){context.input=data;run('apply(input,false)');}
function setup(config=task){context.testTask=structuredClone(config);run(`state.data={project:{title:'Study'},task:testTask,runs:[],ideas:[],discussion:[]};state.slug='study';state.run={id:'r1',model:'q35',active:false};state.rounds=new Map();state.best=state.selected=0;state.follow=true;state.files=[];state.stage='experiments';state.replay=false;artifactKey='';`);}
setup();
event({event:'baseline',metric:'score',result:{score:.1}});
event({event:'idea',round:1,parent:0,source:'you',hypothesis:'<img src=x onerror=alert(1)>',change:'Improve score',why:'Test generalisation',prediction:'Higher score'});
event({event:'curve',round:1,seed:0,step:20,train:1,score:.5});
event({event:'curve',round:1,seed:1,step:20,train:1,score:.7});
event({event:'result',round:1,value:.6,best_before:.1,verdict:'supported'});
event({event:'idea',round:2,parent:1,source:'model',hypothesis:'Regresses',change:'Try another change'});
event({event:'result',round:2,value:.2,best_before:.6,verdict:'refuted'});
run('renderRun()');
assert.match(get('progress').innerHTML,/data-running-best/);
assert.match(get('progress').innerHTML,/rotate\(-25\)/);
assert.match(get('progress').innerHTML,/fill="#d9a19f"/);
assert.match(get('progress').innerHTML,/data-noise-band/);
assert.deepEqual([...get('progress').innerHTML.matchAll(/data-y-tick="[^"]+"[^>]*>([^<]+)</g)].map(m=>m[1]),['0%','25%','50%','75%','100%']);
assert.doesNotMatch(get('progress').innerHTML,/>-\d+(?:\.\d+)?%|>1[1-9]\d(?:\.\d+)?%/);
assert.match(get('progress-subtitle').textContent,/higher is better · 2 experiments/);
assert.match(get('best-caption').textContent,/60.0% · 1 kept/);
assert.match(get('ledger').innerHTML,/>you</);
assert.match(get('ledger').innerHTML,/-40.0%/);
assert.match(get('learned').innerHTML,/Supported changes/);
assert.match(get('learned').innerHTML,/Refuted changes/);
run('state.selected=1;state.follow=false;renderRun()');
assert.match(get('detail-text').innerHTML,/&lt;img/);
assert.doesNotMatch(get('detail-text').innerHTML,/<img/);
assert.match(get('verdict').textContent,/10.0% → 60.0% · supported/);
assert.match(get('curve').innerHTML,/polyline/);
assert.match(get('detail-text').innerHTML,/Prediction/);
assert.equal(run('meanSeries(getRound(1).curves,"score")[0][1]'),.6);
assert.equal(run('progressData().at(-1).runningBest'),.6);
assert.equal(run('progressData().filter(p=>p.kept).length'),2);

// Crashes remain inspectable even when there is no numeric result.
event({event:'idea',round:3,hypothesis:'A failed attempt'});
event({event:'run_error',round:3,error:'Process exited'});
run('renderRun()');
assert.match(get('progress').innerHTML,/data-crashed="true"/);
assert.match(get('progress').innerHTML,/data-round="3"/);
assert.match(get('progress').innerHTML,/No measurement; marker at bottom edge/);
assert.doesNotMatch(get('progress').innerHTML,/NaN|Infinity/);
context.pointEvent={target:{closest:()=>({dataset:{round:'3'}})}};
run('selectProgress(pointEvent)');
assert.equal(run('state.selected'),3);
assert.equal(run('state.follow'),false);
assert.match(get('round-error').textContent,/Process exited/);

// Long changes retain full tooltips and keyboard-accessible expansion.
const longChange='Initialize the embedding matrix with values that encode the modular structure, then pre-train the embeddings before the main training loop. '.repeat(4);
context.longChange=longChange;
run('getRound(1).change=longChange;state.selected=1;renderRun()');
assert.equal(get('change').title,longChange);
assert.equal(get('expand-change').hidden,false);
get('expand-change').onclick();
assert.equal(get('change').className,'expanded');
assert.equal(get('expand-change').attributes['aria-expanded'],'true');
get('expand-detail').onclick();
assert.equal(get('detail-title').className,'expanded');
assert.match(get('ledger').innerHTML,/class="ledger-change"/);
assert.ok(get('ledger').innerHTML.includes(`title="${longChange}"`));
run('state.selected=0;renderRun()');
assert.equal(get('change').className,'');
// Both representative SVG sizes keep shortened labels within the drawing area.
for(const [width,height] of [[1462,415],[1040,315]]) {
  get('progress').clientWidth=width;get('progress').clientHeight=height;
  run('getRound(1).value=.99;state.selected=1;drawProgress()');
  const html=get('progress').innerHTML;
  assert.ok(html.includes(`<title>${longChange}</title>`));
  for(const label of html.matchAll(/data-kept-label="true" transform="translate\(([\d.]+) ([\d.]+)\) rotate\((-?[\d.]+)\)" text-anchor="(start|end)"[^>]*><title>.*?<\/title>([^<]*)<\/text>/g)) {
    const [,a,b,rotation,anchor,text]=label;
    assert.ok(text.length<=28);
    const length=text.length*7*(anchor==='end'?-1:1),angle=Number(rotation)*Math.PI/180;
    const endX=Number(a)+length*Math.cos(angle),endY=Number(b)+length*Math.sin(angle);
    assert.ok(endX>=0 && endX<=width && endY>=12 && endY<=height-48,`Label stays inside SVG: ${label[0]}`);
  }
}
get('progress').clientWidth=1000;get('progress').clientHeight=340;

// Smaller-is-better units come entirely from the template. No accuracy or training assumptions.
const minTask={title:'Sort lists',editable:'sort.py',metric:'runtime_ms',metric_label:'Runtime',direction:'min',unit:'ms',noise:1,seeds:[0,1]};
setup(minTask);
event({event:'baseline',metric:'runtime_ms',result:{runtime_ms:100}});
event({event:'idea',round:1,source:'model',hypothesis:'Sort faster',change:'Use built-in sort'});
event({event:'run_end',round:1,seed:0,result:{runtime_ms:79}});
event({event:'run_end',round:1,seed:1,result:{runtime_ms:81}});
event({event:'result',round:1,value:80,best_before:100,verdict:'supported'});
event({event:'idea',round:2,source:'you',hypothesis:'Marginal change',change:'Try key function'});
event({event:'result',round:2,value:79.8,best_before:80,verdict:'no effect'});
run('renderRun()');
assert.equal(run('progressData().at(-1).runningBest'),80);
assert.match(get('best-caption').textContent,/80 ms/);
assert.match(get('progress-subtitle').textContent,/Runtime · lower is better/);
assert.match(get('progress').innerHTML,/fill="#b3bdc5"/);
assert.ok(run('progressScale([79.8,80,100],state.data.task).low')>0);
assert.deepEqual(JSON.parse(run('JSON.stringify(progressScale([79.8,80,100],state.data.task).ticks)')),[70,80,90,100,110]);
assert.equal(run('progressScale([5,5],state.data.task).step'),.2);
assert.equal(run('progressScale([.000012,.000014],state.data.task).step'),.000001);
assert.equal(run('progressTick(.000012,state.data.task,.000001)'),'0.000012 ms');
assert.match(get('ledger').innerHTML,/-20 ms/);
assert.doesNotMatch(get('progress').innerHTML,/Goal/);
run('state.selected=1;renderRun()');
assert.match(get('curve').innerHTML,/Per-seed measurements/);
assert.match(get('curve').innerHTML,/Seed 0: 79 ms/);
assert.equal((get('seeds').innerHTML.match(/class="seed-chip"/g) || []).length,2);
assert.match(get('seeds').innerHTML,/Seed 0 <span>complete<\/span>/);
assert.match(get('seeds').innerHTML,/Seed 1 <span>complete<\/span>/);
assert.match(get('verdict').textContent,/100 ms → 80 ms · supported/);
assert.match(get('code-label').textContent,/round1_sort.py/);
run('state.stage="results";renderRun()');
await new Promise(resolve=>setImmediate(resolve));
assert.equal(get('code').textContent,'<script>not HTML</script>\nVALUE = 2');
assert.match(get('diff').innerHTML,/class="add"/);
assert.equal(run('parseRoute("#/p/sorting-study/results?run=r1&round=2").stage'),'results');
assert.equal(run('parseRoute("#/p/sorting-study/results?run=r1&round=2").round'),2);
assert.equal(run('parseRoute("#/p/a_b").invalid'),true);
assert.equal(run('parseRoute("#/p/study/experiments?replay=1&speed=nan").speed'),8);
// The default is based on creation time and measured results, independent of IDs or last updates.
context.runOptions=[{id:'z-alphabetically-first',created:100,updated:999,experiments:3},{id:'new-empty',created:400,experiments:0},{id:'a-newest-result',created:300,updated:301,best:.8},{id:'middle',created:200,experiments:1}];
assert.equal(run('defaultRun(runOptions).id'),'a-newest-result');
run('runOptions[0].active=true');
assert.equal(run('defaultRun(runOptions).id'),'z-alphabetically-first');
assert.equal(run('defaultRun([])'),undefined);
// The live pre-polish server can omit created on legacy runs; first event wins over updated.
testEvents=[{event:'baseline',ts:50},{event:'done',ts:900}];
context.legacyRuns=[{id:'legacy',updated:900,experiments:1}];
await run('resolveRunTimes(legacyRuns)');
assert.equal(run('legacyRuns[0].created'),50);
testEvents=[];
run('state.replay=true;state.run.active=true;controls()');
for(const id of ['start','send','queue-message','pause','resume','stop','save-program'])assert.equal(get(id).disabled,true,id);
assert.equal(get('program').readOnly,true);
assert.equal(get('edit-program').disabled,true);
assert.equal(get('replay').textContent,'Replay · 8× speed');

// Markdown-lite escapes raw HTML in prose, lists, and code.
context.markdownInput='Paragraph **strong**\n\n- <img src=x>\n- text\n\n```python\n<script>alert(1)</script>\n```';
const safe=run('markdown(markdownInput)');
assert.match(safe,/<strong>strong<\/strong>/);assert.match(safe,/<ul><li>&lt;img/);assert.match(safe,/<pre><code>&lt;script/);assert.doesNotMatch(safe,/<script>/);

// Empty states remain actionable for a new min-direction project.
setup(minTask);run('state.run=null;renderRun()');
assert.match(get('progress').innerHTML,/Start experiments/);
assert.match(get('ledger').innerHTML,/No results yet/);
assert.match(get('result-location').textContent,/projects\/study/);
assert.doesNotMatch(get('progress').innerHTML,/NaN|Infinity/);

// Route through all stages and switch projects/runs without restarting the app.
context.location.hash='#/p/study';await run('route()');
assert.equal(get('discuss').hidden,false);
assert.equal(get('chat-panel').parentElement.id,'discussion-host');
detail.discussion=[{role:'user',content:'Persisted question'},{role:'assistant',content:'Persisted **answer**',model:'q35'}];
testRun={id:'r1',model:'q27',active:false};testEvents=[{event:'baseline',metric:'score',result:{score:.1}},{event:'done',best:.1}];
context.location.hash='#/p/study/experiments?run=r1';await run('route()');
assert.equal(get('experiments').hidden,false);
assert.equal(get('chat-panel').parentElement.id,'rail-discussion');
assert.match(get('conversation').innerHTML,/Persisted question/);
assert.match(get('conversation').innerHTML,/<strong>answer<\/strong>/);
assert.equal(get('conversation').scrollTop,get('conversation').scrollHeight);
assert.equal(get('start').disabled,false);
context.location.hash='#/p/study/results?run=r1';await run('route()');
assert.equal(get('results').hidden,false);assert.equal(get('chat-panel').hidden,true);
context.location.hash='#/p/study/experiments?run=r1&replay=1&speed=8';await run('route()');
assert.match(streams.at(-1).path,/\/api\/projects\/study\/runs\/r1\/events\?replay=1&speed=8/);
assert.equal(get('start').disabled,true);

// A second template can use the entire app, including a completely empty project.
detail={project:{slug:'sort-speed',title:'Sort speed',model:'q27'},program:'# Reduce runtime\n\nMeasure **carefully**.',task:minTask,runs:[],discussion:[],ideas:[],models:['q35','q27'],default_model:'q35',rounds:5};
context.location.hash='#/p/sort-speed';await run('route()');
assert.match(get('task-summary').innerHTML,/Runtime/);
assert.match(get('task-summary').innerHTML,/lower is better/);
assert.equal(get('program').value,detail.program);
assert.equal(get('program').hidden,true);
assert.equal(get('program-view').hidden,false);
assert.match(get('program-view').innerHTML,/<strong>carefully<\/strong>/);
get('edit-program').onclick();
assert.equal(get('program').hidden,false);
assert.equal(get('edit-program').textContent,'Preview');
get('program').value='Updated **goal**';get('program').oninput();
get('edit-program').onclick();
assert.match(get('program-view').innerHTML,/<strong>goal<\/strong>/);
assert.equal(run('state.programDirty'),true);
get('edit-program').onclick();await get('save-program').onclick();
assert.equal(get('program').hidden,true);
assert.equal(run('state.programDirty'),false);
assert.equal(run('state.data.program'),'Updated **goal**');
assert.equal(get('model').value,'q27');
assert.equal(get('discuss').hidden,false);
context.location.hash='#/p/sort-speed/experiments';await run('route()');
assert.match(get('progress').innerHTML,/Start experiments/);
assert.match(get('progress-subtitle').textContent,/lower is better/);
context.location.hash='#/p/sort-speed/results';await run('route()');
assert.equal(get('results').hidden,false);
assert.match(get('ledger').innerHTML,/No results yet/);
assert.equal(get('downloads').innerHTML,'');

// Refresh the shared persisted discussion when returning from another stage/tab.
detail.discussion=[{role:'user',content:'First saved message'},{role:'assistant',content:'Latest saved reply',model:'q27'}];
context.location.hash='#/p/sort-speed/discuss';await run('route()');
assert.equal(get('chat-panel').parentElement.id,'discussion-host');
assert.ok(get('conversation').innerHTML.indexOf('First saved message')<get('conversation').innerHTML.indexOf('Latest saved reply'));
assert.equal(get('conversation').scrollTop,get('conversation').scrollHeight);

// Compact timestamps, native whole-row links, and metric direction semantics on home.
const now=Date.now();
context.now=now;
assert.equal(run('activityTime((now-12*60000)/1000,now)'),'12 min ago');
assert.equal(run('activityTime(now/1000,now)'),'Just now');
assert.equal(run('activityTime((now-2*3600000)/1000,now)'),'2 hr ago');
assert.doesNotMatch(run('activityTime((now-3*86400000)/1000,now)'),/ago|\/|:\d\d:\d\d/);
testProjects=[{slug:'sort-speed',title:'Sort speed',template:'sort-speed',task:minTask,updated:(now-12*60000)/1000,experiments:2,best:80},{slug:'study',title:'Score study',template:'generic',task,updated:(now-20*60000)/1000,experiments:2,best:.8}];
await run('loadHome()');
assert.match(get('project-list').innerHTML,/12 min ago/);
assert.match(get('project-list').innerHTML,/href="#\/p\/sort-speed"/);
assert.match(get('project-list').innerHTML,/aria-label="Lower is better"/);
assert.match(get('project-list').innerHTML,/aria-label="Higher is better"/);
const css=fs.readFileSync(new URL('../web/style.css',import.meta.url),'utf8');
assert.match(css,/\.project-row a:after\s*\{[^}]*position:absolute;[^}]*inset:0;/);
assert.match(css,/#change,#detail-title,\.ledger-change\s*\{[^}]*-webkit-line-clamp:2;/);
assert.match(css,/#detail-title\s*\{[^}]*-webkit-line-clamp:3;/);
const index=fs.readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
assert.match(index,/<link rel="icon" href="data:image\/svg\+xml,/);

// Default route selection also respects creation time, refreshed active runs, and explicit choice.
detail={...detail,project:{slug:'default-study',title:'Default run study'},runs:[{id:'z-old',created:10,updated:100,experiments:1},{id:'a-new',created:20,experiments:1},{id:'empty',created:30,experiments:0}]};
testRun={id:'a-new',model:'q35',active:false};testEvents=[{event:'baseline',metric:'runtime_ms',result:{runtime_ms:80}}];
context.location.hash='#/p/default-study/experiments';await run('route()');
assert.equal(run('state.run.id'),'a-new');
assert.ok(requests.some(r=>r.path==='/api/projects/default-study/runs/a-new'));
detail.runs[0].active=true;testRun={id:'z-old',model:'q35',active:true};
context.location.hash='#/p/default-study/discuss';await run('route()');
assert.equal(run('state.run.id'),'z-old');
assert.match(streams.at(-1).path,/\/runs\/z-old\/events$/);
testRun={id:'a-new',model:'q35',active:false};
context.location.hash='#/p/default-study/experiments?run=a-new';await run('route()');
assert.equal(run('state.run.id'),'a-new');

// Both real datasets must produce finite progress/curve coordinates and the exact recorded best.
for(const name of ['20261118-113651-acde31','recorded-q35-curves']) {
  const file=new URL(`../projects/grokking/runs/${name}/events.jsonl`,import.meta.url);
  if(!fs.existsSync(file))continue; // These local artifacts are intentionally gitignored.
  const config=JSON.parse(fs.readFileSync(new URL('../templates/grokking/task.json',import.meta.url)));
  setup(config);
  for(const line of fs.readFileSync(file,'utf8').trim().split('\n'))event(JSON.parse(line));
  run('renderRun()');
  assert.doesNotMatch(get('progress').innerHTML,/NaN|Infinity|undefined/);
  assert.doesNotMatch(get('curve').innerHTML,/NaN|Infinity|undefined/);
  const done=JSON.parse(fs.readFileSync(file,'utf8').trim().split('\n').at(-1));
  if(done.event==='done')assert.equal(run('progressData().at(-1).runningBest'),done.best);
  console.log(`Real stream checked: ${name}, ${run('ledgerRows().length')} ledger rows`);
}
clearTimeout(run('noticeTimer'));
console.log('Web checks passed: routing/default run, nice axes, verdict markers/noise band, label bounds, title expansion, seed chips, persisted discussion, goal editing, project metadata, favicon, ledger, artifacts, replay, empty states, real streams.');
