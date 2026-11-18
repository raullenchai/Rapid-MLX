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
const protocolResponses=new Map();
const context=vm.createContext({document,console,URLSearchParams,URL,TextDecoder,TextEncoder,setTimeout,clearTimeout,
  location:{hash:'#/',search:''},window:{addEventListener(){}},
  fetch:async(path,options)=>{
    requests.push({path,options});
    const value=path==='/api/templates' ? {templates:[{slug:'generic',...task}]} : path==='/api/projects' ? {projects:testProjects} : path.endsWith('/program') ? JSON.parse(options.body) : path.endsWith('/diff/1') ? {parent:0,diff:'--- old\n+++ new\n-VALUE = 1\n+VALUE = 2\n'} : path.includes('/files/') ? null : /\/runs\/[^/]+$/.test(path) ? {run:testRun,events:testEvents,files:testFiles} : structuredClone(detail);
    if(protocolResponses.has(path)){const payload=protocolResponses.get(path);return {ok:true,json:async()=>payload,text:async()=>typeof payload==='string'?payload:JSON.stringify(payload)};}
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
assert.match(get('progress').innerHTML,/experiment 1 of 2/);
assert.match(get('progress').innerHTML,/data-noise-band/);
assert.deepEqual([...get('progress').innerHTML.matchAll(/data-y-tick="[^"]+"[^>]*>([^<]+)</g)].map(m=>m[1]),['0%','25%','50%','75%','100%']);
assert.doesNotMatch(get('progress').innerHTML,/>-\d+(?:\.\d+)?%|>1[1-9]\d(?:\.\d+)?%/);
assert.match(get('progress-subtitle').textContent,/higher is better · 2 experiments/);
assert.match(get('best-caption').textContent,/60.0% · 1 kept/);
assert.match(get('queue').innerHTML,/you/);
assert.match(get('ledger').innerHTML,/-40.0%/);
assert.match(get('learned').innerHTML,/Supported hypotheses/);
assert.doesNotMatch(get('learned').innerHTML,/Refuted hypotheses/);
assert.match(get('ledger').innerHTML,/>testing</);
run('state.selected=1;state.follow=false;renderRun()');
assert.match(get('detail-text').innerHTML,/&lt;img/);
assert.doesNotMatch(get('detail-text').innerHTML,/<img/);
assert.match(get('verdict').textContent,/10.0% → 60.0% · improved/);
assert.match(get('curve').innerHTML,/<(?:polyline|circle) data-series=/);
assert.match(get('detail-text').innerHTML,/Prediction/);
assert.equal(run('meanSeries(getRound(1).curves,"score")[0][1]'),.6);
assert.equal(run('progressData().at(-1).runningBest'),.6);
assert.equal(run('progressData().filter(p=>p.kept).length'),2);

// Generic configured x keys, sparse measured series, neutral comparisons, and reference/outliers.
const drugTask={title:'Drug levels',editable:'pk.py',seeds:[0],metric:'rmse',metric_label:'Held-out error',unit:'mg/L',direction:'min',noise:.05,domain:'Medicine',curve:{x:'hours',x_label:'Hours after dose',series:[{key:'model',label:'Model'},{key:'measured',label:'Measured',style:'points'},{key:'other',label:'Other'}]}};
setup(drugTask);
event({event:'baseline',metric:'rmse',result:{rmse:6.283}});
event({event:'idea',round:1,hypothesis:'Fit absorption'});
for(const p of [{hours:0,model:0},{hours:.5,model:6},{hours:1,model:9},{hours:.25,measured:4},{hours:.75,measured:8},{hours:1,other:7}])event({event:'curve',round:1,seed:0,ts:123,...p});
event({event:'result',round:1,value:.864,best_before:6.283,verdict:'supported'});
event({event:'idea',round:2,hypothesis:'An outlier'});
event({event:'result',round:2,value:140.582,best_before:.864,verdict:'refuted'});
run('state.selected=1;renderRun()');
assert.match(get('curve').innerHTML,/data-series="model"/);
assert.match(get('curve').innerHTML,/data-series="measured"/);
assert.equal((get('curve').innerHTML.match(/data-series="measured"/g)||[]).length,2);
assert.doesNotMatch(get('curve').innerHTML,/<polyline[^>]*data-series="measured"/);
assert.match(get('curve').innerHTML,/data-series="model"[^>]*stroke="#176345"/);
assert.match(get('curve').innerHTML,/Hours after dose/);
assert.match(get('curve').innerHTML,/data-series="other"/);
assert.deepEqual(JSON.parse(run('JSON.stringify(meanSeries(getRound(1).curves,"model"))')),[[0,0],[.5,6],[1,9]]);
assert.deepEqual(JSON.parse(run('JSON.stringify(meanSeries(getRound(1).curves,"measured"))')),[[.25,4],[.75,8]]);
assert.doesNotMatch(get('curve').innerHTML,/<(?:polyline|circle) data-series="(?:measured|other)"[^>]*(?:stroke|fill)="#176345"/);
assert.match(get('metric-heading').textContent,/Best held-out error \(mg\/L\)/);
assert.match(get('verdict').innerHTML,/class="metric-value">0.864 mg\/L<\/span>/);
assert.doesNotMatch(get('progress').innerHTML,/data-y-tick="(?:50|100|150|200)"/);
assert.match(get('progress').innerHTML,/data-clamped="high"/);
assert.match(get('progress').innerHTML,/140.582 mg\/L/);
assert.doesNotMatch(get('curve').innerHTML,/NaN|Infinity|undefined/);
const clusterTask={...drugTask,metric:'energy',metric_label:'Cluster energy',unit:'',noise:.3,reference:{value:-44.326801,label:'Known global minimum'},curve:{x:'evals',x_label:'Energy evaluations',series:[{key:'energy',label:'Best energy'}]}};
setup(clusterTask);
event({event:'baseline',metric:'energy',result:{energy:-41.2}});
event({event:'idea',round:1,hypothesis:'Local search'});event({event:'result',round:1,value:-41.5,verdict:'supported'});
event({event:'idea',round:2,hypothesis:'Multiple starts'});event({event:'result',round:2,value:-44.3268,verdict:'supported'});
event({event:'idea',round:3,hypothesis:'Too much work'});event({event:'run_start',round:3,seed:0});
event({event:'result',round:3,verdict:'crashed',error:'Traceback (most recent call last):\n  File "private.py", line 1\nRuntimeError: evaluation budget exceeded\n'});
run('state.selected=3;renderRun()');
assert.match(get('progress').innerHTML,/data-reference="true"/);
assert.match(get('result-reference').textContent,/Known global minimum.*−44.3268.*reached in round 2/);
assert.match(get('seeds').innerHTML,/Seed 0 <span>crashed<\/span>/);
assert.equal(get('round-error').textContent,'RuntimeError: evaluation budget exceeded · on seed 0');
assert.match(get('round-traceback').textContent,/Traceback/);
assert.match(get('detail-text').innerHTML,/<summary>Show error<\/summary>/);
run('$("round-error-details").open=true;renderCurrent()');assert.equal(get('round-error-details').open,true);
run('state.selected=2;renderCurrent()');assert.equal(get('round-error-details').hidden,true);assert.equal(get('round-error-details').open,false);
run('state.selected=3;renderCurrent()');
assert.equal(get('round-error-details').hidden,false);
// Both reference and goal contribute to the domain; rejected lower outliers keep their values.
setup({...clusterTask,metric:'score',direction:'max',noise:0,goal:30,reference:{value:40,label:'Reference'}});
event({event:'baseline',metric:'score',result:{score:10}});
event({event:'idea',round:1,hypothesis:'Improve'});event({event:'result',round:1,value:20,verdict:'supported'});
event({event:'idea',round:2,hypothesis:'Too low'});event({event:'result',round:2,value:-200,verdict:'refuted'});
run('renderRun()');
assert.match(get('progress').innerHTML,/data-clamped="low"/);
assert.match(get('progress').innerHTML,/-200; outside plotted range/);
assert.match(get('progress').innerHTML,/data-y-tick="40"/);
assert.match(get('progress').innerHTML,/data-goal="true"/);
assert.match(get('progress').innerHTML,/data-reference="true"/);
assert.doesNotMatch(get('result-reference').textContent,/reached/);
setup();
event({event:'baseline',metric:'score',result:{score:.1}});
event({event:'idea',round:1,hypothesis:'Improve score'});event({event:'result',round:1,value:.6,verdict:'supported'});
event({event:'idea',round:2,hypothesis:'Regresses'});event({event:'result',round:2,value:.2,verdict:'refuted'});

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
assert.match(get('progress').innerHTML,/experiment 1 of 2/);
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
assert.match(get('verdict').textContent,/100 ms → 80 ms · improved/);
assert.match(get('code-label').textContent,/round1_sort.py/);
run('state.stage="results";renderRun()');
await new Promise(resolve=>setImmediate(resolve));
assert.match(get('code').innerHTML.replace(/<[^>]+>/g,''),/&lt;script&gt;not HTML&lt;\/script&gt;/);
assert.doesNotMatch(get('code').innerHTML,/<script>/);
assert.match(get('code').innerHTML,/class="line-number"/);
assert.match(get('code').innerHTML,/syntax-number/);
assert.equal(run('artifactText.code'),'<script>not HTML</script>\nVALUE = 2');
assert.equal(get('copy-code').disabled,false);
assert.match(get('diff').innerHTML,/class="add"/);
assert.equal(get('diff-stats').textContent,'+1 −1');
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

// Project summaries agree with verdicts and metric direction, across chronological runs.
context.historyRuns=[{created:2,events:[{event:'baseline',result:{score:.2}},{event:'result',value:.99,verdict:'no effect'},{event:'result',value:.9,verdict:'supported'}]},{created:1,events:[{event:'baseline',result:{score:.1}},{event:'result',value:.8,verdict:'supported'},{event:'result',value:.4,verdict:'refuted'},{event:'result',verdict:'crashed'}]}];
assert.deepEqual(Array.from(run('projectHistory(historyRuns,{metric:"score",direction:"max"})')),[.1,.8,.8,.8,.8,.8,.9]);
context.historyRuns=[{created:1,events:[{event:'baseline',result:{runtime_ms:100}},{event:'result',value:80,verdict:'supported'},{event:'result',value:79.9,verdict:'no effect'},{event:'result',value:90,verdict:'refuted'}]}];
assert.deepEqual(Array.from(run('projectHistory(historyRuns,{metric:"runtime_ms",direction:"min"})')),[100,80,80,80]);
const minSpark=run('sparkline([100,80,80],{direction:"min",unit:"ms"})');
assert.match(minSpark,/lower is better/);
assert.match(minSpark,/M5,7 H55 V31 H105 V31/); // The lower metric sits lower on its natural axis.
assert.match(run('sparkline([.1,.8,.8],{direction:"max",unit:"%"})'),/M5,31 H55 V7 H105 V7/);
assert.doesNotMatch(run('sparkline([5,5,5],{})'),/NaN|Infinity/);
assert.match(run('sparkline([],{})'),/Awaiting baseline/);
assert.equal(run(String.raw`goalDescription("\n# Research goal\nOther content")`),'Other content');
assert.equal(run('deltaClass(-1,{direction:"min"})'),'supported');
assert.equal(run('deltaClass(1,{direction:"min"})'),'refuted');
assert.equal(run('deltaClass(1,{direction:"max"})'),'supported');
assert.equal(run('deltaClass(-1,{direction:"max"})'),'refuted');
assert.equal(run('deltaClass(0,{})'),'');
assert.equal(run('deltaClass(NaN,{})'),'');
// Home renders new server summaries without any run-detail requests.
testProjects=[{slug:'fast',title:'Fast sort',template:'sort-speed',description:'<img src=x onerror=alert(1)>',status:'Finished',progress:[100,80],task:minTask,best:80,experiments:1,updated:now/1000}];
const beforeSummary=requests.length;await run('loadHome()');
assert.equal(requests.length-beforeSummary,2);
assert.match(get('project-list').innerHTML,/Finished/);
assert.match(get('project-list').innerHTML,/80 ms/);
assert.match(get('project-list').innerHTML,/class="sparkline"/);
assert.match(get('project-list').innerHTML,/&lt;img/);
assert.doesNotMatch(get('project-list').innerHTML,/<img/);
assert.match(get('template-cards').innerHTML,/type="radio"/);
assert.match(get('template-cards').innerHTML,/Higher is better/);
testProjects=[];await run('loadHome()');
assert.match(get('project-list').innerHTML,/Create your first project/);
get('first-project').onclick();assert.equal(get('new-form').hidden,false);
get('cancel-project').onclick();assert.equal(get('new-form').hidden,true);

// Domain chips filter without losing the activity ordering or unknown-domain projects.
const homeProject=(slug,domain,updated)=>({slug,title:slug,template:'generic',task:{...minTask,domain},description:'A research goal',status:'Finished',progress:[10,8],experiments:2,best:8,updated});
context.domainProjects=[homeProject('older','Physics',100),homeProject('newer','Medicine',200),homeProject('unclassified','',300)];
run('state.projects=domainProjects;state.domain="";renderProjects()');
assert.match(get('domain-filters').innerHTML,/All domains/);
assert.match(get('domain-filters').innerHTML,/data-domain="Medicine"/);
assert.ok(get('project-list').innerHTML.indexOf('unclassified')<get('project-list').innerHTML.indexOf('newer'));
assert.match(get('project-list').innerHTML,/class="domain-chip">Physics/);
context.domainClick={target:{closest:()=>({dataset:{domain:'Medicine'}})}};
get('domain-filters').onclick(context.domainClick);
assert.match(get('project-list').innerHTML,/newer/);assert.doesNotMatch(get('project-list').innerHTML,/older|unclassified/);
assert.match(get('domain-filters').innerHTML,/data-domain="Medicine" aria-pressed="true"/);
run('state.projects=[];renderProjects()');assert.equal(get('domain-filters').hidden,true);
assert.equal(run('format(1,{unit:""})'),'1');assert.equal(run('format(1,{unit:"mg/L"})'),'1 mg/L');
// Stale active summaries are rebuilt from completed results in every run, before done arrives.
detail={...detail,task:minTask,runs:[{id:'old',created:10},{id:'live',created:20,active:true}]};
testRun={id:'live',model:'q35',active:true};
testEvents=[{event:'baseline',metric:'runtime_ms',result:{runtime_ms:100},ts:100},{event:'result',value:80,verdict:'supported',ts:110},{event:'result',verdict:'crashed',ts:120},{event:'run_start',round:3,seed:0,ts:130}];
context.activeProject={...homeProject('active','Computer science',50),status:'Running',experiments:0};
await run('projectSummary(activeProject)');
assert.equal(run('activeProject.experiments'),4);
assert.equal(run('activeProject.best'),80);
assert.equal(run('activeProject.updated'),130);
// Empty discussion always begins at the heading, including after moving to the rail.
setup(minTask);run('state.data.discussion=[];renderConversation()');assert.equal(get('conversation').scrollTop,0);

// Generic starters fill the composer; they never send a request or queue an idea.
setup(minTask);run('state.run=null;renderConversation()');
assert.equal((get('conversation').innerHTML.match(/data-prompt=/g)||[]).length,3);
assert.match(get('conversation').innerHTML,/Why does the baseline fail on runtime/);
const requestCount=requests.length;
context.promptEvent={target:{closest:selector=>selector==='[data-prompt]'?{dataset:{prompt:'0'}}:null}};
await get('conversation').onclick(context.promptEvent);
assert.equal(get('message').value,'Why does the baseline fail on runtime?');
assert.equal(document.activeElement.id,'message');assert.equal(requests.length,requestCount);
run('state.replay=true');get('message').value='Keep draft';
await get('conversation').onclick(context.promptEvent);assert.equal(get('message').value,'Keep draft');
context.unsafeMetric='<script>alert(1)</script>';
run('state.replay=false;state.data.task.metric_label=unsafeMetric;renderConversation()');
assert.match(get('conversation').innerHTML,/&lt;script&gt;/);assert.doesNotMatch(get('conversation').innerHTML,/<script>/);
run('state.data.discussion=[{role:"user",content:"Question"}];state.chatDraft={role:"assistant",model:"q27",content:"Answer",pending:true};renderConversation()');
assert.match(get('conversation').innerHTML,/chat user/);
assert.match(get('conversation').innerHTML,/chat assistant streaming/);
assert.match(get('conversation').innerHTML,/27B · q27/);
assert.match(get('conversation').innerHTML,/streaming-caret/);
assert.doesNotMatch(get('conversation').innerHTML,/data-queue/);
run('state.chatDraft=null;state.data.discussion.push({role:"assistant",content:"A testable idea"});renderConversation()');
assert.match(get('conversation').innerHTML,/queue-action/);

// Tokenization retains exact source and never allows code to become markup.
context.sourceCode=`# <img src=x onerror=alert(1)>\nimport math\nx = "<script>&'\\"</script>"\nvalue = 1.25e-3\ntext = """first\n# still a string\nlast"""\nif value: return False`;
assert.equal(run('tokenizeCode(sourceCode).map(t=>t.text).join("")'),context.sourceCode);
const tinted=run('codeMarkup(sourceCode)');
assert.doesNotMatch(tinted,/<img|<script|<\/script>/);
assert.match(tinted,/&lt;img/);assert.match(tinted,/&lt;script/);
for(const kind of ['comment','keyword','number','string'])assert.match(tinted,new RegExp(`syntax-${kind}`));
assert.equal((tinted.match(/class="code-line"/g)||[]).length,context.sourceCode.split('\n').length);
assert.match(run(String.raw`codeMarkup("unterminated = \"<img src=x>\n# safe")`),/&lt;img/);
const diffInfo=run(String.raw`diffMarkup("--- before\n+++ after\n@@ -1 +1 @@\n-<script>\n+<img>\n context")`);
assert.equal(diffInfo.added,1);assert.equal(diffInfo.removed,1);
assert.match(diffInfo.html,/diff-meta/);assert.doesNotMatch(diffInfo.html,/<img|<script/);
const trickyDiff=run(String.raw`diffMarkup("--- old\n+++ new\n@@ -1 +1 @@\n--- comment\n+++ comment")`);
assert.equal(trickyDiff.added,1);assert.equal(trickyDiff.removed,1);
context.navigator={clipboard:{writeText:async text=>{context.copied=text;}}};
run('artifactText.code=sourceCode');await get('copy-code').onclick();assert.equal(context.copied,context.sourceCode);
// Live status and pulse follow the active round/seed, independently of inspection.
setup(task);event({event:'baseline',metric:'score',result:{score:.1}});
event({event:'idea',round:3,hypothesis:'Try this'});event({event:'run_start',round:3,seed:1});
run('state.run.active=true;state.run.rounds=10;state.selected=0;renderRun()');
assert.equal(get('run-state').textContent,'Running · experiment 3 of 10 · '+run('hypothesisFor(getRound(3)).label')+' · seed 1');
assert.match(get('progress').innerHTML,/live-point/);
assert.match(get('start-summary').textContent,/Baseline \+ 5 experiments · seeds 0, 1/);
run('state.run.paused=true;renderRun()');assert.doesNotMatch(get('progress').innerHTML,/live-point/);
assert.match(css,/@media\(prefers-reduced-motion:reduce\)/);
assert.match(css,/th\s*\{[^}]*position:sticky/);
assert.match(index,/class="brand-mark"/);

// Default route selection also respects creation time, refreshed active runs, and explicit choice.
detail={...detail,project:{slug:'default-study',title:'Default run study'},task:{...minTask,domain:'Physics'},runs:[{id:'z-old',created:10,updated:100,experiments:1},{id:'a-new',created:20,experiments:1},{id:'empty',created:30,experiments:0}]};
testRun={id:'a-new',model:'q35',active:false};testEvents=[{event:'baseline',metric:'runtime_ms',result:{runtime_ms:80}}];
context.location.hash='#/p/default-study/experiments';await run('route()');
assert.equal(run('state.run.id'),'a-new');
assert.equal(get('project-domain').textContent,'Physics');assert.equal(get('project-domain').hidden,false);
assert.ok(requests.some(r=>r.path==='/api/projects/default-study/runs/a-new'));
detail.runs[0].active=true;testRun={id:'z-old',model:'q35',active:true};
context.location.hash='#/p/default-study/discuss';await run('route()');
assert.equal(run('state.run.id'),'z-old');
assert.match(streams.at(-1).path,/\/runs\/z-old\/events\?after=0$/);
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
// All six domain runs exercise configured x keys, sparse measurements, crash summaries, and finite charts.
const projectsDir=new URL('../projects/',import.meta.url);
let domainRuns=0;
for(const entry of (fs.existsSync(projectsDir)?fs.readdirSync(projectsDir,{withFileTypes:true}):[]).filter(e=>e.isDirectory())) {
  const projectDir=new URL(`${entry.name}/`,projectsDir), runsDir=new URL('runs/',projectDir);
  if(!fs.existsSync(runsDir))continue;
  for(const id of fs.readdirSync(runsDir).filter(id=>id.startsWith('20261118-133652-'))) {
    const config=JSON.parse(fs.readFileSync(new URL('task.json',projectDir),'utf8'));
    setup(config);
    const events=fs.readFileSync(new URL(`${id}/events.jsonl`,runsDir),'utf8').trim().split('\n').map(JSON.parse);
    for(const e of events)event(e);
    run('renderRun()');
    assert.doesNotMatch(get('progress').innerHTML,/NaN|Infinity|undefined/);
    for(const round of run('ledgerRows().map(r=>r.round)')) {
      context.round=round;run('state.selected=round;renderCurrent()');
      assert.doesNotMatch(get('curve').innerHTML,/NaN|Infinity|undefined/);
      const hasCurve=run('getRound(round).curves.some(p=>Number.isFinite(p[state.data.task.curve.x]) && state.data.task.curve.series.some(s=>Number.isFinite(p[s.key])))');
      if(hasCurve)assert.match(get('curve').innerHTML,/data-series=/,`${entry.name} round ${round} renders real curves`);
      if(run('getRound(round).verdict')==='crashed')assert.doesNotMatch(get('seeds').innerHTML,/>running</);
    }
    const done=events.at(-1);if(done.event==='done')assert.equal(run('progressData().at(-1).runningBest'),done.best);
    if(config.domain==='Medicine')assert.match(get('progress').innerHTML,/data-clamped="high"/);
    if(config.reference)assert.match(get('result-reference').textContent,/reached in round 2/);
    domainRuns++;console.log(`Real domain stream checked: ${entry.name}, ${run('ledgerRows().length')} ledger rows`);
  }
}
if(domainRuns)console.log(`Local domain streams checked: ${domainRuns}`);
clearTimeout(run('noticeTimer'));
console.log('Web checks passed: configured sparse curves, robust domains/carets, references, domain filters, active counts, compact errors/seed state, empty rail scroll; sparklines, starter prompts, tokenizer safety, copy/diff counts, live status, directional deltas, template cards; routing/default run, nice axes, verdict markers/noise band, label bounds, title expansion, seed chips, persisted discussion, goal editing, project metadata, favicon, ledger, artifacts, replay, empty states, real streams.');

// First-class hypotheses group many experiments; only evidence determines their status.
setup({...minTask,experiments_per_hypothesis:2});
run(`state.data.hypotheses=[{id:'h1',statement:'Stronger decay improves generalisation',why:'Less overfitting',prediction:'Lower error',source:'you',experiments:[]},{id:'h2',statement:'Smoothing lowers variance',source:'discussion',experiments:[]}];`);
event({event:'baseline',metric:'runtime_ms',result:{runtime_ms:100}});
event({event:'idea',round:1,hypothesis_id:'h1',hypothesis:'Stronger decay improves generalisation',change:'WEIGHT_DECAY 0.01 -> 1.0'});
event({event:'result',round:1,hypothesis_id:'h1',value:100,best_before:100,verdict:'no effect'});
run('renderRun()');
assert.equal(run('hypothesisGroups()[0].status'),'testing');
assert.equal(run('hypothesisGroups()[1].status'),'untested');
assert.match(get('queue').innerHTML,/H1.*testing/s);
assert.match(get('queue').innerHTML,/1 experiment\b(?!s)/);
assert.match(get('ledger').innerHTML,/data-hypothesis="h1"/);
assert.match(get('ledger').innerHTML,/code-change/);
assert.match(get('ledger').innerHTML,/title="WEIGHT_DECAY 0.01 → 1.0"/);
assert.match(get('experiment-groups').innerHTML,/data-experiment="1"/);
assert.match(get('hypothesis-rule').textContent,/After 2 experiments/);
const h1Color=run('hypothesisColor(hypothesisGroups()[0])');
event({event:'idea',round:2,hypothesis_id:'h1',hypothesis:'Stronger decay improves generalisation',change:'WEIGHT_DECAY 0.01 -> 2.0'});
event({event:'result',round:2,hypothesis_id:'h1',value:80,best_before:100,verdict:'improved'});
run('renderRun()');
assert.equal(run('hypothesisGroups().length'),2);
assert.equal(run('hypothesisGroups()[0].status'),'supported');
assert.equal(run('hypothesisGroups()[0].experiments.length'),2);
assert.match(get('progress').innerHTML,/H1 · Stronger decay improves generalisation · experiment 2 of 2/);
assert.equal((get('progress').innerHTML.match(new RegExp(`fill="${h1Color}"`,'g')) || []).length>=2,true);
assert.match(get('learned').innerHTML,/Supported hypotheses/);
assert.doesNotMatch(get('learned').innerHTML,/Supported changes/);
assert.doesNotMatch(get('verdict').textContent,/supported|refuted/);
context.groupEvent={target:{closest:()=>({dataset:{experiment:'1'}})}};
get('experiment-groups').onclick(context.groupEvent);
assert.equal(run('state.selected'),1);assert.equal(run('state.follow'),false);
for(const [verdicts,status] of [[['worse','no effect'],'refuted'],[['crashed','crashed'],'inconclusive'],[['crashed','no effect'],'inconclusive'],[['crashed'],'testing'],[[],'untested']]) {
  context.statusInput=verdicts.map(verdict=>({verdict}));
  assert.equal(run('hypothesisStatus(statusInput)'),status);
}
// A server snapshot is replaced by the selected stream, so replay cannot leak future results.
run(`state.data.hypotheses[0].experiments=[{run:'r1',round:1,verdict:'improved'},{run:'older',round:4,verdict:'no effect'}];state.replay=true;state.rounds=new Map();`);
assert.equal(run('hypothesisGroups()[0].status'),'testing');
assert.equal(run('hypothesisGroups()[0].experiments.length'),1);
// Identical legacy claim text dedupes across rounds without changing recorded objects.
setup(minTask);
event({event:'idea',round:1,hypothesis:'Same claim'});event({event:'result',round:1,verdict:'refuted',value:110});
event({event:'idea',round:2,hypothesis:'Same claim'});event({event:'result',round:2,verdict:'no effect',value:100});
run('renderRun()');
assert.equal(run('hypothesisGroups().length'),1);assert.equal(run('hypothesisGroups()[0].status'),'refuted');
assert.match(get('learned').innerHTML,/Refuted hypotheses/);
// Error retries with the same exception collapse; seed scope is backed by observed seed events.
setup({...clusterTask,seeds:[0,1,2]});
event({event:'idea',round:1,hypothesis:'More starts'});
for(const seed of [0,1,2]){event({event:'run_start',round:1,seed});event({event:'run_error',round:1,seed,error:`Traceback at seed ${seed}\nRuntimeError: evaluation budget exceeded`});}
event({event:'result',round:1,verdict:'crashed',error:'RuntimeError: evaluation budget exceeded'});
run('state.selected=1;renderRun()');
assert.equal((get('detail-text').innerHTML.match(/<summary>Show error<\/summary>/g)||[]).length,1);
assert.match(get('detail-text').innerHTML,/on all seeds/);
assert.match(get('round-error').textContent,/on all seeds/);
// Both round SVGs reserve sign/label width and use rounded 1–2–5 ticks.
for(const p of [{evals:0,energy:-39.7445},{evals:16579,energy:-44.3268}])event({event:'curve',round:1,seed:0,...p});
run('renderRun()');
assert.match(get('curve').innerHTML,/>-4[0246]</);
assert.doesNotMatch(get('curve').innerHTML,/>39\.7445|>16,57(?:<|\.)/);
assert.match(get('curve').innerHTML,/data-curve-x-tick="15000"/);
for(const id of ['curve','result-curve']) {
  const html=get(id).innerHTML;
  for(const m of html.matchAll(/data-curve-y-tick="[^"]+" x="([\d.]+)"[^>]*>([^<]+)</g))assert.ok(Number(m[1])-m[2].length*7>=0);
  for(const m of html.matchAll(/data-curve-x-tick="[^"]+" x="([\d.]+)"[^>]*>([^<]+)</g))assert.ok(Number(m[1])+m[2].length*3.5<=get(id).clientWidth);
}
assert.match(index,/Add hypothesis/);assert.match(index,/id="hypothesis-form"/);
assert.doesNotMatch(index,/Queued ideas|Queue this idea|Queue as idea|>Refuted<\/span>/);
console.log('Hypothesis checks passed: grouped evidence, statuses, live association, nested inspection, replay isolation, legacy dedupe, error dedupe/seed scope, signed nice curve axes.');
clearTimeout(run('noticeTimer'));
// Hypothesis mutations send claim/rationale fields; replay remains read only.
setup(minTask);
const priorFetch=context.fetch;
context.fetch=async(path,options)=>{
  requests.push({path,options});
  const data=JSON.parse(options.body);
  return {ok:true,json:async()=>({hypothesis:{id:'h9'},hypotheses:[{id:'h9',...data,experiments:[]}]})};
};
await run('addHypothesis("Regularization lowers error","you","Less overfitting","Lower error")');
assert.equal(requests.at(-1).path,'/api/projects/study/hypotheses');
assert.deepEqual(JSON.parse(requests.at(-1).options.body),{statement:'Regularization lowers error',source:'you',why:'Less overfitting',prediction:'Lower error'});
assert.equal(run('hypothesisGroups()[0].status'),'untested');
assert.match(get('queue').innerHTML,/Regularization lowers error/);
const beforeReadOnly=requests.length;
run('state.replay=true;controls()');
await run('addHypothesis("Another claim")');
assert.equal(requests.length,beforeReadOnly);assert.equal(get('add-hypothesis').disabled,true);
assert.equal(get('hypothesis-statement').readOnly,true);
context.fetch=priorFetch;
// A repair-time error is visible, but does not conclude the hypothesis prematurely.
setup({...minTask,experiments_per_hypothesis:1});
event({event:'idea',round:1,hypothesis_id:'h1',hypothesis:'A claim'});
event({event:'run_start',round:1,seed:0});event({event:'run_error',round:1,error:'RuntimeError: transient'});
assert.equal(run('hypothesisGroups()[0].status'),'testing');
event({event:'result',round:1,verdict:'crashed'});
assert.equal(run('hypothesisGroups()[0].status'),'inconclusive');
clearTimeout(run('noticeTimer'));
assert.equal(run('legacyHypothesisId("Stronger decay α 💡")'),'legacy-04a44bf9e6cd4edc');
// Home carries the two scientific counts together.
testProjects=[{slug:'counted',title:'Counted study',template:'sort-speed',description:'Goal',status:'Finished',progress:[10,8],task:minTask,best:8,hypotheses:2,experiments:4,updated:Date.now()/1000}];
await run('loadHome()');assert.match(get('project-list').innerHTML,/2 hypotheses · 4 experiments/);
clearTimeout(run('noticeTimer'));

// Protocol renderers are project independent and all unsafe content is escaped.
setup();
run('state.protocol=1;state.artifacts=[];state.watermark=0');
const envelope=(id,kind,role='attachment',mime='text/plain')=>({protocol:1,id,revision:1,kind,role,title:'Artifact '+id,path:'artifacts/'+id+'.txt',mime,bytes:10,sha256:'a'.repeat(64),producer:'harness',owner:{scope:'experiment',run:'r1',experiment:1},refs:[],provenance:{},created:1});
const artifacts=[envelope('source','code'),envelope('notes','report'),envelope('table','data','curve','text/csv'),envelope('plot','chart','curve','application/json'),envelope('image','image','structure','image/svg+xml'),envelope('logs','log'),envelope('attachment','file'),envelope('future','molecule')];
artifacts[3].refs=['table'];context.protocolArtifacts=artifacts;run('state.artifacts=protocolArtifacts');
for(const a of artifacts) {
  context.inputArtifact=a;
  const url=run('artifactURL(inputArtifact)');
  protocolResponses.set(url,a.kind==='report'?'**Report** <script>alert(1)</script> $x^2$':a.kind==='chart'?JSON.stringify({$schema:'rl-chart/1',mark:'line',data:{artifact:'table'},encoding:{x:{field:'x',title:'Time'},y:{field:'y',title:'Score'}}}):a.kind==='data'?'x,y\n1,2\n2,3\n':'<script>raw text</script>\nx = 2');
  if(a.kind==='data')protocolResponses.set(url+'&rows=500',{rows:[{x:1,y:2},{x:2,y:3}],columns:{x:{type:'number',unit:'s'}},total:1000});
  const html=await run('artifactMarkup(inputArtifact)');
  assert.match(html,/Download/);
  if(a.kind==='code' || a.kind==='log' || a.kind==='report'){assert.doesNotMatch(html,/<script>/);assert.match(html,/&lt;script&gt;/);}
  if(a.kind==='image'){assert.match(html,/<img/);assert.doesNotMatch(html,/<svg/);}
  if(a.kind==='chart')assert.match(html,/data-mark="line"/);
  if(a.kind==='data'){assert.match(html,/2 of 1000 rows/);assert.match(html,/x \(s\)/);assert.match(html,/data-sort/);}
  if(a.kind==='log'){assert.match(html,/data-log-search/);assert.match(html,/Last 200 lines/);assert.doesNotMatch(html,/markdown-body/);}
  if(a.kind==='molecule'){assert.match(html,/Unrecognized artifact/);assert.doesNotMatch(html,/<svg|<img|<pre/);}
}
assert.equal(run('safeArtifact({kind:"report",title:"bad"}).kind'),'file');
assert.match(run('markdown("[safe](https://example.org) [local](/docs) [bad](javascript:alert) <img>")'),/href="https:\/\/example.org"/);
assert.doesNotMatch(run('markdown("[bad](javascript:alert) [bad](//evil.test) [bad](data:text/html,bad)")'),/href=/);
context.chart={...JSON.parse(protocolResponses.get(run('artifactURL(protocolArtifacts[3])'))),mark:'bar',band:{low:'lo',high:'hi'},reference:{value:2,label:'<unsafe>'}};
context.chartRows=[{x:1,y:2,lo:1,hi:3},{x:2,y:3,lo:2,hi:4}];
assert.match(run('chartMarkup(chart,chartRows)'),/data-mark="bar"/);
assert.match(run('chartMarkup(chart,chartRows)'),/data-band="true"/);
assert.match(run('chartMarkup(chart,chartRows)'),/&lt;unsafe&gt;/);
context.chart.mark='point';context.chart.encoding.x.scale={type:'log'};
assert.match(run('chartMarkup(chart,chartRows)'),/data-mark="point"/);
context.chart.data={url:'https://evil.test'};
assert.throws(()=>run('chartMarkup(chart,chartRows)'),/Unsupported chart/);
context.csvInput='x,y\n"one,two","quoted ""value"""\n';
assert.deepEqual(JSON.parse(run("JSON.stringify(parseCSV(csvInput))")),[{x:"one,two",y:'quoted "value"'}]);
event({seq:1,event:'curve',round:1,seed:0,step:1,score:2});
event({seq:1,event:'curve',round:1,seed:0,step:1,score:2});
assert.equal(run('getRound(1).curves.length'),1);
console.log('Protocol checks passed: every renderer, safe file fallback, CSV quoting/units/preview, declarative marks/bands/log scales/reference, Markdown URLs, sequenced event deduplication.');
clearTimeout(run('noticeTimer'));

run('state.artifacts.push(null,{kind:"report",owner:"bad owner"})');
assert.equal(run('latestArtifacts().at(-1).owner.scope'),'run');
assert.equal(run('latestArtifacts().at(-1).kind'),'file');

// Recording polish: shared Results DOM leads with the outcome, then evidence.
const resultsDOM=index.slice(index.indexOf('<section id="results"'));
const outcomeOrder=['id="result-best"','<h2>Hypotheses</h2>','id="ledger"','id="result-detail"','id="run-report"','id="run-artifacts"'];
for(let i=1;i<outcomeOrder.length;i++)assert.ok(resultsDOM.indexOf(outcomeOrder[i-1])<resultsDOM.indexOf(outcomeOrder[i]),outcomeOrder[i]);
assert.match(index,/id="run-actions"/);
assert.match(index,/id="run-control-note"/);
setup();run('controls()');
assert.equal(get('run-state').textContent,'Run finished');
for(const id of ['run-actions','run-model','run-control-note'])assert.equal(get(id).hidden,true,id);
run('state.run.active=true;state.run.paused=false;controls()');
assert.equal(get('run-actions').hidden,false);
assert.equal(get('pause').disabled,false);assert.equal(get('resume').disabled,true);assert.equal(get('stop').disabled,false);
run('state.run.paused=true;controls()');
assert.equal(get('pause').disabled,true);assert.equal(get('resume').disabled,false);

// Saved completion and replay never announce a fresh completion.
for(const protocol of [undefined,1]) {
  setup();context.testProtocol=protocol;run('state.protocol=testProtocol;notice("")');
  event({event:'done'});assert.equal(get('notice').textContent,'');
  run('state.run.active=true;state.replay=true;apply({event:"done"})');
  assert.equal(get('notice').textContent,'');
  run('state.replay=false;state.run.active=true');
  const originalTimeout=context.setTimeout;let dismiss;
  context.setTimeout=(callback,delay)=>{if(delay===6000)dismiss=callback;return originalTimeout(callback,delay);};
  run('apply({event:"done"},false)');
  assert.equal(get('notice').textContent,'');
  run('state.run.active=true;apply({event:"done"})');
  assert.match(get('notice').textContent,/Run complete/);assert.equal(typeof dismiss,'function');
  clearTimeout(run('noticeTimer'));dismiss();assert.equal(get('notice').textContent,'');
  context.setTimeout=originalTimeout;
}
clearTimeout(run('noticeTimer'));
console.log('Recording polish checks passed: Results order, active controls, session-only completion and six-second dismissal.');

// Only the harness run log is disclosed; experiment output stays expanded.
context.recordingLog={...artifacts.find(a=>a.kind==='log'),producer:'harness',owner:{scope:'run',run:'r1'}};
protocolResponses.set(run('artifactURL(recordingLog)'),'first line\nsecond line\n');
run('artifactCache.clear()');
const collapsedLog=await run('artifactMarkup(recordingLog)');
assert.match(collapsedLog,/<details class="loop-log"><summary>Loop log · 2 lines<\/summary>/);
assert.doesNotMatch(collapsedLog,/<details[^>]* open/);
assert.match(collapsedLog,/class="log-controls"/);
run('recordingLog.owner={scope:"experiment",run:"r1",experiment:1};artifactCache.clear()');
assert.doesNotMatch(await run('artifactMarkup(recordingLog)'),/<details/);
assert.match(css,/\.log-tail\s*\{[^}]*flex-direction:row;/);
assert.match(css,/\.hypothesis-rail\s*\{[^}]*max-height:360px;[^}]*overflow:auto;/);
