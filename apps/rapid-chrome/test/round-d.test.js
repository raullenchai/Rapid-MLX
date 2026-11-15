import test from 'node:test';
import assert from 'node:assert/strict';
import { outlineAX } from '../src/ax.js';
import { BrowserHand } from '../src/cdp.js';
import { classifyTarget, NavigationPolicy, needsSiteApproval } from '../src/safety.js';

// r4 tool snippets supply labels/content; ancestry is reconstructed from the
// local fixture HTML. Chromium emits InlineTextBox children under StaticText.
function tree() {
  const nodes = [];
  const node = (role, name = '', children = [], extra = {}) => {
    const n = {nodeId:String(nodes.length + 1),backendDOMNodeId:nodes.length + 100,
      role:{value:role},name:{value:name},childIds:children.map(c => c.nodeId),...extra};
    nodes.push(n); return n;
  };
  const text = label => node('StaticText',label,[node('InlineTextBox',label)]);
  return {nodes,node,text};
}
test('r4 shop / chat: real StaticText prices survive compaction, with stock and other-offer qualifiers', () => {
  const {nodes,node,text} = tree();
  const cards = [['Bolt', '$29.99'],['Swift','$24.99']].map(([name,price]) => node('article','',[
    node('heading',`${name} 65W charger`),text(`List price: ${price} — 4.8 stars`),
    node('group','',[text('Other sellers from $9.99'),text('Used, no warranty; this is not the listed offer.')]),
    text('In stock'),node('spinbutton','Quantity',[],{value:{value:'1'}}),node('button','Add to cart')
  ]));
  node('RootWebArea','Market Square',cards);
  for (const query of ['', '65W', '$', 'Swift']) {
    const result = outlineAX(nodes,{query});
    assert.match(result.text,/\$24.99/);
    if (query !== 'Swift') assert.match(result.text,/\$29.99/);
    assert.match(result.text,/other offers from: \$9.99/);
    assert.match(result.text,/In stock/);
    assert.match(result.text,/Used, no warranty/);
    assert.match(result.text,/value="1"/);
  }
});
test('r4 subscriptions: StreamPlus cost, last use and active status remain observable', () => {
  const {nodes,node,text} = tree();
  node('RootWebArea','Subscriptions',[node('article','',[
    node('heading','StreamPlus'),text('$18.00/month. Last used 2026-07-01. Status: Active'),node('button','Cancel StreamPlus')
  ])]);
  for (const query of ['', 'last', 'used', 'StreamPlus']) {
    const result = outlineAX(nodes,{query});
    assert.match(result.text,/\$18.00/);assert.match(result.text,/Last used 2026-07-01/);
    assert.match(result.text,/Status: Active/);assert.match(result.text,/Cancel StreamPlus/);
  }
});
test('r4 invoice: amount, due date and sender survive article handling', () => {
  const {nodes,node,text} = tree();
  node('RootWebArea','Acme Mail',[node('article','',[
    node('heading','Acme November invoice'),text('From: billing@acme.test'),
    text('Invoice AC-2026-11: $1,240.50 due November 28, 2026.'),
    node('combobox','Label',[],{value:{value:'Finance'}}),node('button','Apply label'),node('button','Reply')
  ])]);
  for (const query of ['', 'due', 'invoice']) {
    const result = outlineAX(nodes,{query});
    assert.match(result.text,/\$1,240.50/);assert.match(result.text,/November 28, 2026/);
    assert.match(result.text,/billing@acme.test/);assert.match(result.text,/value="Finance"/);
  }
});
test('r4 Uber: a page wrapper with money is not one product swallowing the address field', () => {
  const {nodes,node,text} = tree();
  const links = Array.from({length:8},(_,i) => node('link',i ? `App ${i}` : 'Sign up',[],{
    properties:[{name:'url',value:{value:`https://uber.test/login?next=${'x'.repeat(700)}`}}]
  }));
  node('RootWebArea','Uber Eats',[node('generic','',[
    node('heading','Order delivery near you'),...links,node('textbox','Enter delivery address'),
    node('heading','Cities near me'),node('heading','Countries with Uber Eats'),text('Save $5 on your first order')
  ])]);
  for (const query of ['', 'address']) {
    const result = outlineAX(nodes,{query});
    assert.ok([...result.refs.values()].some(t => t.name === 'Enter delivery address'));
    assert.doesNotMatch(result.text,/Product: Order delivery near you/);
  }
});
test('r4 punctuation queries do not match every node or import unrelated ancestor cards', () => {
  const {nodes,node,text} = tree();
  node('RootWebArea','',[node('main','',[
    node('article','',[node('heading','Swift'),text('$24.99')]),
    node('article','',[node('heading','Unrelated'),text('No price')])
  ])]);
  assert.doesNotMatch(outlineAX(nodes,{query:'$'}).text,/Unrelated|No price/);
});
function gateHand(info) {
  const cards = [];
  const hand = new BrowserHand({get:async()=>({url:'http://127.0.0.1:55254/tickets'}),send:async()=>({})},
    {settings:{base:'http://127.0.0.1:8000'},approve:async c=>{cards.push(c);return false;}});
  hand.signal = new AbortController().signal;
  hand.tabs.set(1,{epoch:0,refs:new Map()});hand.inspect=async()=>info;
  return {hand,cards};
}
test('r4 tickets / calendar: preparatory buttons do not prompt; Buy and Save still gate', async () => {
  for (const name of ['C1 — $25','C2 — $25','Create event']) {
    const {hand,cards} = gateHand({connected:true,name,type:'button',role:'button'});
    await hand.gateTarget(1,{epoch:0,role:'button'},'click');assert.equal(cards.length,0);
  }
  for (const name of ['Buy tickets','Buy policy — Pine','Place order']) {
    const {hand,cards} = gateHand({connected:true,name,type:'submit'});
    await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/financial_handover/);assert.equal(cards.length,0);
  }
  for (const name of ['Save event','Confirm cancellation','C1 — $25 Buy now','Create event and pay']) {
    const {hand} = gateHand({connected:true,name,type:'button',role:'button'});
    await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/denied|financial_handover/);
  }
  for (const name of ['C1 — $25','Create event']) {
    for (const context of [{type:'submit',method:'post'},{sensitiveForm:true},{paymentForm:true},{form:'Confirm purchase'},{action:'https://shop.test/pay'}]) {
      const target = {name,type:'button',role:'button',...context};
      const c = classifyTarget(target);assert.ok(c.consequential || c.financial);
    }
  }
});
test('r4 Kyoto: MediaWiki search entry point works without allowing invented articles or mutation URLs', () => {
  const policy = new NavigationPolicy();
  assert.doesNotThrow(()=>policy.check(new URL('https://en.wikipedia.org/w/index.php?search=Kyoto')));
  for (const path of ['/wiki/Kyoto','/w/index.php?search=Kyoto&action=delete','/w/index.php?title=Kyoto','/w/index.php?search='])
    assert.throws(()=>policy.check(new URL('https://en.wikipedia.org'+path)),/unobserved_url/);
  assert.equal(needsSiteApproval('https://en.wikipedia.org/',new Set(['https://en.wikivoyage.org'])),true);
  assert.equal(needsSiteApproval('http://127.0.0.1:55254/',new Set(['http://127.0.0.1:55255'])),true);
});
test('r4 Kyoto navigation reports a settled fresh outline rather than the previous site URL', async () => {
  for (const name of ['navigate','open_tab']) {
    const {hand} = gateHand({});const calls=[];let settled=false,reads=0;
    hand.active=1;hand.tabs.set(2,{epoch:0,refs:new Map()});
    hand.settle=async id=>{calls.push(['settle',id]);settled=true;};
    hand.meta=async()=>({url:settled ? 'https://en.wikipedia.org/' : 'https://en.wikivoyage.org/wiki/Kyoto'});
    hand.executeCore=async(tool,args)=>{
      calls.push([tool,args.tabId]);
      if (tool !== 'read_page') return {text:'navigate completed',tabId:name==='open_tab'?2:1,url:'https://en.wikivoyage.org/wiki/Kyoto'};
      assert.equal(settled,true);reads++;
      if (reads===1) throw Error('stale_ref: navigation during read');
      return {text:'URL: https://en.wikipedia.org/\n[e1] searchbox "Search Wikipedia"'};
    };
    const result=await hand.execute(name,{tabId:1,url:'https://en.wikipedia.org/'},hand.signal);
    assert.equal(result.url,'https://en.wikipedia.org/');assert.match(result.text,/Search Wikipedia/);
    assert.equal(calls.filter(c=>c[0]===name).length,1);assert.equal(reads,2);
    assert.ok(calls.filter(c=>c[0]==='read_page').every(c=>c[1]===(name==='open_tab'?2:1)));
    hand.executeCore=async()=>{throw Error('denied: new origin');};
    hand.settle=async()=>assert.fail('must not inspect a denied navigation');
    await assert.rejects(hand.execute(name,{url:'https://private.test/'},hand.signal),/denied/);
  }
});
test('r4 queried district cell keeps its own description without swallowing neighboring cells', () => {
  const {nodes,node,text}=tree();
  node('RootWebArea','',[node('main','',[node('table','',[node('row','',[
    node('cell','',[node('link','South'),text('Fushimi-ku and Daigo')]),
    node('cell','',[node('link','Central'),text('Unrelated district')])
  ])])])]);
  const result=outlineAX(nodes,{query:'Fushimi'});
  assert.match(result.text,/South/);assert.doesNotMatch(result.text,/Unrelated district/);
});
