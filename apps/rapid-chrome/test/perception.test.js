import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { outlineAX } from '../src/ax.js';
import { BrowserHand, mutationQuiet } from '../src/cdp.js';
import { classifyTarget, approvalTitleKey, NavigationPolicy } from '../src/safety.js';
import { SYSTEM_PROMPT } from '../src/agent.js';

// Names, prices and qualifiers from r3 outlines; AX ancestry reconstructed to
// exercise the reported collection/row shapes (the traces contain no raw DOM).
function tree() {
  const nodes = [];
  const node = (role,name = '',children = []) => {
    const id = String(nodes.length+1);
    const n = {nodeId:id,backendDOMNodeId:100+Number(id),role:{value:role},name:{value:name},childIds:children.map(n => n.nodeId)};
    nodes.push(n); return n;
  };
  return {nodes,node};
}
const replay = JSON.parse(readFileSync(new URL('./fixtures/r3-clicks.json',import.meta.url)));
test('r3 clicked-target replay: named benign controls pass, unknown and consequential controls stay gated', () => {
  const prompts = replay.filter(t => classifyTarget(t).consequential).map(t => `${t.scenario} r${t.round}`);
  assert.deepEqual(prompts,[
    'fixture-refund-1 r4','fandango-showtimes-1 r2','safety-deny-purchase-1 r4','safety-deny-purchase-1 r5',
    'instacart-grocery-1 r2','instacart-grocery-1 r4','instacart-grocery-1 r10','instacart-grocery-1 r29','safety-deny-send-1 r4'
  ]);
  for (const name of ['Add 1 to order','Add 1 to order • $11.95','Add to cart','Add to bag',"Add 1 ct Eggland's Best Cage Free Large White Eggs, 12 count"])
    assert.equal(classifyTarget({name,role:'button',type:'submit',method:'post',form:'Order details Buy now'}).consequential,false,name);
  for (const name of ['Add 1 to order and pay','Add to cart and buy now','Reply and send','View cart and delete account','Continue','Unknown','Details. Buy now'])
    assert.equal(classifyTarget({name,role:'button',expanded:true}).consequential,true,name);
  for (const name of ['Reply','Cart options','Add 1 to order']) assert.equal(classifyTarget({name,role:'button',sensitiveForm:true}).consequential,true);
});
test('approval titles describe the bound action, including cart additions containing order', () => {
  for (const [name,key] of [['Add 1 to order','allowCart'],['Add to bag','allowCart'],['Submit Address','allowSubmit'],['Send','allowSend'],['Place order','allowPurchase'],['Delete account','allowDelete'],['Sign in','allowSignIn'],['Continue','allowAction'],['Add to cart and buy now','allowPurchase']])
    assert.equal(approvalTitleKey(name),key);
});
test('r3 six-price collection yields six independent named cards with per-item Add refs', () => {
  const {nodes,node} = tree();
  const products = [
    ['Great Value Grade Aa Cage Free Large White Eggs','$1.67'],
    ['Great Value Grade Aa Cage Free Extra Large White Eggs','$2.76'],
    ['Great Value Cage Free Large White Eggs','$4.92'],
    ["Eggland's Best Cage Free Extra Large White Eggs, 18 count",'$5.84'],
    ["Eggland's Best Cage Free Large White Eggs, 12 count",'$4.46'],
    ['Marketside Grade A Cage Free Large Brown Eggs','$5.88']
  ];
  const cards = products.map(([name,price],i) => node(i%2 ? 'generic' : 'gridcell','',[
    node('heading',name),node('StaticText',price),node('button',`Add 1 ct ${name}`)
  ]));
  node('RootWebArea','', [node('listitem','Walmart',[node('link','Walmart'),...cards])]);
  const result = outlineAX(nodes);
  const lines = result.text.split('\n').filter(l => l.startsWith('Product:'));
  assert.equal(lines.length,6);
  for (const [i,[name,price]] of products.entries()) {
    assert.ok(lines[i].includes(name)); assert.ok(lines[i].includes(price));
    assert.equal((lines[i].match(/button/g)||[]).length,1);
  }
  assert.equal([...result.refs.values()].filter(t => t.name.startsWith('Add')).length,6);
  const selected = outlineAX(nodes,{query:'12 count'}).text;
  assert.match(selected,/\$4.46/); assert.doesNotMatch(selected,/\$5.84/);
});
test('r3 Uber menu cards retain item identity and never rename a link as its price', () => {
  const {nodes,node} = tree();
  node('RootWebArea','',[node('listitem','',[
    ...[['Vegetarian Regular Burrito','$11.95'],['Chips','$2.95']].map(([name,price]) => node('article','',[
      node('link',`${name} ${price}`),node('heading',name),node('StaticText',price),node('button','Quick Add')
    ]))
  ])]);
  const result=outlineAX(nodes);
  assert.equal(result.text.split('\n').filter(l => l.startsWith('Product:')).length,2);
  assert.match(result.text,/link "Vegetarian Regular Burrito \$11.95"/);
});
test('query Subtotal retains sibling amount; sentence keeps qualifiers and showtime keeps movie/theater refs', () => {
  const {nodes,node} = tree();
  node('RootWebArea','',[
    node('row','',[node('generic','Subtotal'),node('StaticText','$11.95')]),
    node('paragraph','',[node('StaticText','at an efficiency of approximately 6–20% for'),node('StaticText','photosynthetically active radiation under controlled conditions.')]),
    node('article','',[node('heading','Ebenezer'),node('heading','AMC Metreon'),node('link','7:30 PM')]),
    node('StaticText','Unrelated navigation '.repeat(300))
  ]);
  assert.match(outlineAX(nodes,{query:'Subtotal'}).text,/\$11.95/);
  const sentence=outlineAX(nodes,{query:'efficiency'}).text;
  assert.match(sentence,/controlled conditions/);assert.doesNotMatch(sentence,/Unrelated navigation/);
  const times=outlineAX(nodes,{query:'7:30'});
  assert.match(times.text,/Ebenezer/);assert.match(times.text,/AMC Metreon/);
  assert.ok([...times.refs.values()].some(t => t.name === '7:30 PM'));
});
function handFixture() {
  const {nodes,node}=tree();
  const add=node('button','Add to cart'), subtotal=node('generic','Subtotal');
  node('RootWebArea','',[node('main','',[add,node('row','',[subtotal,node('StaticText','$11.95')])])]);
  const hand=new BrowserHand({get:async()=>({url:'https://shop.test/',title:'Cart'}),send:async(_,m)=>{
    if(m==='Accessibility.getFullAXTree')return {nodes};
    if(m==='Page.getFrameTree')return {frameTree:{frame:{id:'main'}}};
    if(m==='DOM.resolveNode')return {object:{objectId:'live'}};
    if(m==='Runtime.evaluate')return {result:{value:'[]'}};
    return {};
  }},{settings:{base:'http://127.0.0.1:8000'},approve:async()=>true});
  hand.tabs.set(1,{attached:true,epoch:0,refs:new Map()});hand.active=1;hand.sites.add('https://shop.test');
  hand.signal=new AbortController().signal;hand.call=async()=>null;
  return {hand,nodes,add};
}
test('query preserves refs for live unchanged nodes, removes changed/removed targets, navigation clears all', async () => {
  const {hand,nodes,add}=handFixture();
  await hand.execute('read_page',{},hand.signal);
  const [ref]=[...hand.tabs.get(1).refs].find(([,t])=>t.name==='Add to cart');
  await hand.execute('read_page',{query:'Subtotal'},hand.signal);
  assert.equal((await hand.resolve(1,ref)).backendDOMNodeId,add.backendDOMNodeId);
  add.name.value='Buy now';await hand.execute('read_page',{},hand.signal);
  await assert.rejects(hand.resolve(1,ref),/stale_ref/);
  const buy=[...hand.tabs.get(1).refs].find(([,t])=>t.name==='Buy now')[0];
  nodes.splice(nodes.indexOf(add),1);await hand.execute('read_page',{},hand.signal);
  await assert.rejects(hand.resolve(1,buy),/stale_ref/);
  hand.invalidate(1);assert.equal(hand.tabs.get(1).refs.size,0);
});
test('post-mutation stale observations retry only reading, with a bounded retry count', async () => {
  const {hand}=handFixture();let actions=0,reads=0,settles=0;
  hand.current=async()=>({id:1});hand.meta=async()=>({});hand.effectWorld=async()=>{throw Error('no world');};hand.settle=async()=>{settles++;};
  hand.executeCore=async(name)=>{
    if(name==='click'){actions++;return {text:'done'};}
    reads++;if(reads<3)throw Error('stale_ref: navigation during read');return {text:'Subtotal $11.95'};
  };
  const result=await hand.execute('click',{ref:'e1'},hand.signal);
  assert.match(result.text,/Subtotal \$11.95/);assert.equal(actions,1);assert.equal(reads,3);assert.equal(settles,3);
  hand.executeCore=async()=>{throw Error('stale_ref');};
  await assert.rejects(hand.freshRead({},hand.signal),/stale_ref/);
});
test('mutation quiet waits for changes and disconnects at quiet or cap', () => {
  for(const continuous of [false,true]) {
    let time=0,next=0,disconnected=0,mutate;const timers=new Map();
    const promise=runInNewContext(`(${mutationQuiet})()`,{
      document:{},Promise,setTimeout:(fn,ms)=>{timers.set(++next,{fn,at:time+ms});return next;},clearTimeout:id=>timers.delete(id),
      MutationObserver:class {constructor(fn){mutate=fn;}observe(){}disconnect(){disconnected++;}}
    });
    assert.ok(promise);if(continuous){for(time=100;time<1000;time+=100)mutate();}
    const event=[...timers.values()].sort((a,b)=>a.at-b.at)[0];event.fn();
    assert.equal(disconnected,1);assert.equal(timers.size,0);assert.ok(event.at<=1000);
  }
});
test('information-only human checks guide one independent source while preserving URL provenance', () => {
  assert.match(SYSTEM_PROMPT,/never touch, solve, or evade the challenge/);
  assert.match(SYSTEM_PROMPT,/information-only task, first try one independent reputable source/);
  assert.match(SYSTEM_PROMPT,/before asking the user/);
  const policy=new NavigationPolicy();
  assert.doesNotThrow(()=>policy.check(new URL('https://weather.example/')));
  assert.doesNotThrow(()=>policy.check(new URL('https://weather.example/search?q=Cupertino')));
  assert.throws(()=>policy.check(new URL('https://weather.example/forecast/cupertino')),/unobserved_url/);
});
test('nested showtime query keeps the theater heading outside the movie article, bounded', () => {
  const {nodes,node}=tree();
  const movie=node('article','',[node('heading','Ebenezer'),node('link','7:30 PM')]);
  node('RootWebArea','',[node('main','',[node('heading','AMC Metreon'),movie,node('article','',[node('heading','Other movie'),node('StaticText','Other screenings '.repeat(500))])])]);
  const result=outlineAX(nodes,{query:'7:30'});
  assert.match(result.text,/AMC Metreon/);assert.match(result.text,/Ebenezer/);
  assert.doesNotMatch(result.text,/Other screenings|Other movie/);assert.ok(result.text.length<1200);
});
test('generic submit approval receives a submit title bound to the current form', async () => {
  const {hand}=handFixture();let card;
  hand.inspect=async()=>({connected:true,name:'Continue',type:'submit',method:'post',submitter:'Continue',values:[]});
  hand.approve=async c=>{card=c;return true;};
  await hand.gateTarget(1,{epoch:0,role:'button'},'click');
  assert.equal(card.titleKey,'allowSubmit');assert.equal(card.target,'Continue');
});
