import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand, clickRect, canActivate, sameClickControl, fillNative } from '../src/cdp.js';
import { NavigationPolicy } from '../src/safety.js';
import { outlineAX } from '../src/ax.js';
import { RepeatGuard } from '../src/agent.js';
import { validateTool } from '../src/tools.js';

const base = 'http://127.0.0.1:8000';
const ax = (id, role, name, href) => ({nodeId:String(id),backendDOMNodeId:id,role:{value:role},name:{value:name},properties:href ? [{name:'url',value:{value:href}}] : []});
function setup(handler = async () => ({})) {
  const commands = [], approvals = [];
  const hand = new BrowserHand({get:async () => ({url:'https://www.amazon.com/s?k=charger',title:'Search'}),send:async (_,method,p) => {
    commands.push({method,p}); return handler(method,p);
  }},{settings:{base},approve:async card => {approvals.push(card);return true;}});
  hand.active = 1; hand.signal = new AbortController().signal;
  hand.tabs.set(1,{attached:true,epoch:0,refs:new Map()});
  return {hand,commands,approvals};
}

test('scrolled Amazon input overlay uses viewport rect after carousel scroll, not pageY subtraction', async () => {
  let scrolled = false;
  const {hand,commands} = setup(async (method,p) => {
    if (method === 'DOM.scrollIntoViewIfNeeded') { scrolled = true; return {}; }
    if (method === 'Runtime.callFunctionOn') {
      assert.ok(scrolled);
      assert.equal(p.functionDeclaration,clickRect.toString());
      const value = runInNewContext(`(${p.functionDeclaration}).call(node)`,{
        innerWidth:800,innerHeight:600,scrollX:900,scrollY:2400,
        node:{getBoundingClientRect:() => ({left:20,top:120,right:140,bottom:160})}
      });
      return {result:{value}};
    }
    if (method === 'DOM.getNodeForLocation') { assert.deepEqual([p.x,p.y],[80,140]); return {backendNodeId:7}; }
    if (method === 'DOM.getBoxModel' || method === 'Page.getLayoutMetrics') assert.fail('Do not mix document and viewport coordinates');
    return {};
  });
  assert.equal(await hand.click(1,{backendDOMNodeId:7,objectId:'input',epoch:0}),'pointer');
  assert.equal(commands.filter(c => c.method === 'Input.dispatchMouseEvent').length,2);
});

test('zero-area and horizontally offscreen targets never dispatch pointer events', async () => {
  for (const rect of [{left:10,top:10,right:10,bottom:10},{left:900,top:10,right:800,bottom:30}]) {
    const {hand,commands} = setup(async () => ({result:{value:rect}}));
    await assert.rejects(hand.click(1,{backendDOMNodeId:7,objectId:'input',epoch:0}),/offscreen_target/);
    assert.ok(!commands.some(c => c.method.startsWith('Input.')));
  }
});

test('navigation provenance allows only user URLs, observed hrefs, and root/search entries', () => {
  const policy = new NavigationPolicy('Use https://www.amazon.com/dp/USER123 and compare it.');
  const seen = 'https://www.bestbuy.com/site/real-charger/123.html?skuId=123';
  policy.observe(seen);
  for (const url of ['https://www.amazon.com/dp/USER123',seen,'https://resy.com/','https://www.amazon.com/s?k=65W+charger','https://www.google.com/travel/flights?q=SFO+JFK','https://www.bestbuy.com/site/searchpage.jsp?st=charger']) policy.check(new URL(url));
  for (const url of ['https://www.amazon.com/dp/B0CDB3K7R7','https://www.bestbuy.com/site/invented/6535055.html','https://resy.com/cities/sf/venues/made-up','https://resy.com/product?q=search','https://resy.com/?redirect=/venue','https://www.bestbuy.com/site/real-charger/123.html?skuId=999']) assert.throws(() => policy.check(new URL(url)),/unobserved_url:.*Click a link.*site search/);
  assert.throws(() => new NavigationPolicy().check(new URL(seen)),/unobserved_url/);
});

test('both URL tools reject invented deep URLs before approval, tab creation, or navigation', async () => {
  const {hand,commands,approvals} = setup();
  hand.sites.add('https://www.bestbuy.com'); // Approval for a site is not a deep URL grant.
  for (const tool of ['navigate','open_tab']) await assert.rejects(hand.executeCore(tool,{url:'https://www.bestbuy.com/site/invented/123.html'},hand.signal),/unobserved_url/);
  assert.equal(commands.length,0); assert.equal(approvals.length,0);
});

test('only emitted outline hrefs are remembered across reads; begin clears task provenance', async () => {
  const href = 'https://www.amazon.com/dp/REAL123';
  const hidden = 'https://www.amazon.com/dp/OMITTED';
  const {hand} = setup(async method => {
    if (method === 'Accessibility.getFullAXTree') return {nodes:[ax(1,'link','Charger',href),ax(2,'link','Other',hidden)]};
    return {};
  });
  const result = await hand.executeCore('read_page',{query:'Charger'},hand.signal);
  assert.match(result.text,/href="https:\/\/www.amazon.com\/dp\/REAL123"/);
  hand.navigation.check(new URL(href));
  assert.throws(() => hand.navigation.check(new URL(hidden)),/unobserved_url/);
  await hand.executeCore('read_page',{query:'Other'},hand.signal);
  hand.navigation.check(new URL(href)); // refs expire, task link evidence persists.
  hand.own = async () => {}; hand.showOverlay = async () => {};
  await hand.begin(hand.signal,'Use https://www.amazon.com/dp/USER');
  assert.throws(() => hand.navigation.check(new URL(href)),/unobserved_url/);
  hand.navigation.check(new URL('https://www.amazon.com/dp/USER'));
});

test('composite flight queries match words across prices, punctuation and itinerary text', () => {
  const result = outlineAX([ax(1,'button','From 1230 US dollars round trip total. Nonstop flight with JetBlue. Departs 3:25 PM and arrives 11:55 PM'),ax(2,'button','From 1400 US dollars round trip total. Delta')],{query:'1230 JetBlue 3:25 PM 11:55 PM'});
  assert.equal(result.refs.size,1); assert.match(result.text,/JetBlue/); assert.doesNotMatch(result.text,/Delta/);
});

test('successful repeated navigation keys and wheels are allowed; failed repeats still blocked', () => {
  for (const [name,args] of [['press',{key:'ArrowDown'}],['scroll',{direction:'down'}]]) {
    const guard = new RepeatGuard();
    guard.check(name,args); guard.record(name,args,{}); guard.check(name,args);
    guard.record(name,args,{error:'failed'});
    assert.throws(() => guard.check(name,args),/repeated_call/);
  }
});

test('fill choose types once and selects a unique observed option through the usual click gate', async () => {
  const {hand,commands} = setup(async (method,p) => {
    if (method === 'Runtime.callFunctionOn' && p.functionDeclaration === fillNative.toString()) return {result:{value:{handled:false}}};
    if (method === 'DOM.resolveNode') return {object:{objectId:String(p.backendNodeId)}};
    if (method === 'Accessibility.getFullAXTree') return {nodes:[ax(10,'option','John F. Kennedy International Airport JFK'),ax(11,'option','LaGuardia LGA')]};
    return {};
  });
  hand.tabs.get(1).refs.set('e1',{backendDOMNodeId:1,role:'combobox',name:'Destination'});
  const gated = [], clicked = [];
  hand.gateTarget = async (_,target,action,opts) => { gated.push([target.backendDOMNodeId,action,opts]); };
  hand.click = async (_,target) => {clicked.push(target.backendDOMNodeId);return 'pointer';};
  validateTool('fill',{ref:'e1',text:'JFK',choose:'JFK'});
  await hand.executeCore('fill',{ref:'e1',text:'JFK',choose:'JFK'},hand.signal);
  assert.equal(commands.filter(c => c.method === 'Input.insertText').length,1);
  assert.deepEqual(clicked,[10]); assert.deepEqual(gated.map(g => g[0]),[1,10]);
  assert.ok(!commands.some(c => c.p?.key === 'Enter'));
});

test('choose rejects ambiguous/missing options and invalid choose+submit combinations', async () => {
  for (const [nodes,error] of [[[ax(1,'option','JFK Airport'),ax(2,'option','JFK City')],/ambiguous_option/],[[],/option_not_found/]]) {
    const {hand,commands} = setup(async () => ({nodes}));
    await assert.rejects(hand.chooseOption(1,'JFK'),error);
    assert.ok(!commands.some(c => c.method.startsWith('Input.')));
  }
  for (const args of [{choose:'',submit:false},{choose:'---',submit:false},{choose:'JFK',submit:true}]) assert.throws(() => validateTool('fill',{ref:'e1',text:'JFK',...args}),/invalid_arguments/);
});


test('Google Flights ARIA option and its text share a click control', () => {
  const option = {closest:() => option};
  const text = {closest:() => option};
  assert.equal(sameClickControl.call(option,text,20,20),true);
});

test('non-focusable ARIA option fallback still activates only the approved visible node', async () => {
  const {hand,commands} = setup(async (method,p) => {
    if (method === 'DOM.getNodeForLocation') return {backendNodeId:99};
    if (method === 'DOM.resolveNode') return {object:{objectId:'overlay'}};
    if (method === 'DOM.focus') throw new Error('Element is not focusable');
    if (method === 'Runtime.callFunctionOn') {
      assert.equal(p.objectId,'approved-option');
      if (p.functionDeclaration === clickRect.toString()) return {result:{value:{left:0,top:0,right:100,bottom:100}}};
      if (p.functionDeclaration === sameClickControl.toString()) return {result:{value:false}};
      if (p.functionDeclaration === canActivate.toString()) return {result:{value:true}};
      assert.match(p.functionDeclaration,/this.click/); return {result:{}};
    }
    return {};
  });
  assert.match(await hand.click(1,{objectId:'approved-option',backendDOMNodeId:7,epoch:0}),/DOM activation/);
  assert.ok(!commands.some(c => c.method.startsWith('Input.')));
  assert.equal(commands.filter(c => c.p?.functionDeclaration?.includes('this.click()')).length,1);
});

test('client rects in same-origin frames map to the CDP top viewport', () => {
  const top = {innerWidth:800,innerHeight:600}; top.top = top;
  const frame = {clientLeft:2,clientTop:2,ownerDocument:{defaultView:top},getBoundingClientRect:() => ({left:100,top:200})};
  const view = {top,parent:top,frameElement:frame};
  const rect = runInNewContext(`(${clickRect.toString()}).call(node)`,{innerWidth:300,innerHeight:200,node:{ownerDocument:{defaultView:view},getBoundingClientRect:() => ({left:10,top:20,right:50,bottom:60})}});
  assert.deepEqual(JSON.parse(JSON.stringify(rect)),{left:112,top:222,right:152,bottom:262});
});
