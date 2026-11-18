// Browser-independent state checks. This does not replace a rendered Chrome check.
import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const elements = new Map();
function element() {
  return {textContent:'',innerHTML:'',value:'',disabled:false,hidden:true,dataset:{},
    scrollTop:0,scrollHeight:0,clientHeight:0,append(){},setAttribute(){}};
}
const get = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
const context = vm.createContext({
  document:{getElementById:get,querySelector:()=>get('scroll'),createElement:element},
  location:{search:'?run=test&replay=1&speed=8',href:'http://localhost/?run=test&replay=1&speed=8',toString(){return this.href}},
  history:{replaceState(){}}, URL, URLSearchParams, console,
  fetch:async()=>({ok:true,json:async()=>({runs:[{id:'test',model:'q35',task:'grokking'}],model:'q35',tasks:['grokking']})}),
  EventSource:class {close(){} addEventListener(){}},
});
vm.runInContext(fs.readFileSync(new URL('../web/app.js', import.meta.url), 'utf8'), context);
await new Promise(resolve=>setImmediate(resolve));
function event(data) { context.input = data; vm.runInContext('apply(input)', context); }
event({event:'baseline',metric:'test_acc',task:'Grokking',result:{test_acc:0}});
event({event:'idea',round:1,parent:0,source:'human',hypothesis:'<img src=x onerror=alert(1)>',change:'weight decay',why:'Generalise'});
event({event:'curve',round:1,seed:0,step:200,train_acc:1,test_acc:.108});
event({event:'result',round:1,value:.108,best_before:0,verdict:'supported'});
assert.match(get('verdict').textContent, /0.0% → 10.8% · supported/);
assert.match(get('cards').innerHTML, /&lt;img/);
assert.match(get('cards').innerHTML, />you</);
assert.match(get('chart').innerHTML, /polyline/);
assert.match(get('map').innerHTML, /stroke="#181818"/);
assert.equal(get('test').disabled, true);
assert.equal(get('start').disabled, true);
assert.equal(get('replay').textContent, 'replay · 8× speed');
event({event:'idea',round:2,parent:1,hypothesis:'Another idea',change:'learning rate'});
event({event:'run_error',round:2,error:'Training failed'});
event({event:'fix',round:2,diff:'-bad\n+fixed'});
event({event:'run_end',round:2,seed:0});
assert.match(get('diff-label').textContent, /crashed → fixed by the model → ran/);
assert.match(get('diff').innerHTML, /class="add"/);
event({event:'result',round:2,verdict:'crashed'});
assert.equal(get('verdict').textContent,'crashed');
vm.runInContext('selected = 1; follow = false; render()', context);
assert.match(get('round-label').textContent, /Round 1 · selected/);
assert.match(get('verdict').textContent, /10.8%/);
vm.runInContext('active = false; controls()', context);
assert.equal(get('send').disabled,true); // Replay stays read-only, even after completion.
console.log('Web state checks passed (verdicts, chart, map, escaping, fixes, replay controls, selection).');
