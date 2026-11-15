import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { classifyTarget, additionKey } from '../src/safety.js';
import { effectReport, watchEffects, collectEffects } from '../src/effects.js';
import { outlineAX } from '../src/ax.js';
import { BrowserHand } from '../src/cdp.js';

test('non-submitting filter/dismissal names ignore POST and purchase form context, en + zh', () => {
  const context = {form:'Buy now Confirm purchase 删除',method:'post',type:'button'};
  for (const name of ['Stops, Not selected','Close dialog','Close','Dismiss','Cancel','Not now','Sort by price','关闭对话框','取消','暂不','经停','排序']) {
    assert.equal(classifyTarget({...context,name,role:'button'}).consequential,false,name);
    assert.equal(classifyTarget({...context,name,role:'button',type:'submit'}).consequential,true,name);
  }
  for (const role of ['checkbox','radio','tab','option','menuitem','switch']) {
    for (const name of ['Nonstop','Cheapest from 928 US dollars Learn more about ranking','直飞','最便宜','Order by price','订单记录','确认筛选']) {
      assert.equal(classifyTarget({...context,name,role}).consequential,false,`${role} ${name}`);
      assert.equal(classifyTarget({...context,name,role,type:'submit'}).consequential,true,`${role} ${name}`);
    }
    for (const name of ['Buy now','Delete account','Send message','购买','删除','发送']) assert.equal(classifyTarget({...context,name,role}).consequential,true,name);
  }
  for (const name of ['Close account','关闭账户','Cancel and delete order']) assert.equal(classifyTarget({name}).consequential,true,name);
  for (const name of ['Add to cart','Quick Add','加入购物车']) assert.equal(classifyTarget({...context,name}).consequential,true,name);
});

test('addition identity survives ref, tab and tracking changes while distinguishing products', () => {
  const key = (url,product,name = 'Add to cart') => additionKey(url,{name,product});
  assert.equal(key('https://shop.test/p?tracking=1','Charger'),key('https://shop.test/p?tracking=2','Charger'));
  assert.notEqual(key('https://shop.test/p','Charger'),key('https://shop.test/p','Cable'));
  for (const name of ['Quick Add','Increase quantity','+','加入购物车','增加数量']) assert.ok(key('https://shop.test/p','Charger',name));
  assert.equal(key('https://shop.test/p','Charger','View cart'),null);
});

test('DOM observer retains transient live text, sheet contents and nearby cart badge changes', () => {
  let callback, disconnected = 0, regions = [], badge = '0';
  const element = (text,role) => ({innerText:text,textContent:text,tagName:'DIV',getClientRects:() => [{}],getAttribute:k => k === 'role' ? role : null});
  const cart = {...element('Cart'),getAttribute:k => k === 'href' ? '/cart' : null,parentElement:{querySelector:() => ({textContent:badge})}};
  const document = {title:'Product',querySelectorAll:selector => selector === 'a,button,[role="link"]' ? [cart] : regions};
  const context = {document,location:{href:'https://shop.test/p'},getComputedStyle:() => ({visibility:'visible'}),MutationObserver:class {
    constructor(fn) { callback = fn; } observe() {} disconnect() { disconnected++; }
  }};
  const before = runInNewContext(`(${watchEffects.toString()})()`,context);
  regions = [element('Added to cart','status'),element('Cart drawer: Charger $25','dialog')]; badge = '1'; callback();
  regions = []; callback(); // toast disappeared before collection
  const samples = runInNewContext(`(${collectEffects.toString()})()`,context);
  const effect = effectReport(before,samples);
  assert.match(effect.regions.join(' '),/Added to cart.*Charger/);
  assert.deepEqual(effect.counters,[{before:'Cart 0',after:'Cart 1'}]);
  assert.equal(disconnected,1);
  assert.equal(effect.url,null);
  assert.deepEqual(effectReport(before,[{...before,url:'https://shop.test/cart',title:'Cart'}]).title,{before:'Product',after:'Cart'});
});

function shoppingHand() {
  let additions = 0, allowed = false;
  const approvals = [], commands = [];
  const hand = new BrowserHand({get:async () => ({url:'https://shop.test/p',title:'Charger'}),send:async (_id,method,params) => {
    commands.push({method,params});
    if (method === 'DOM.describeNode') return {node:{backendNodeId:7,nodeName:'BUTTON'}};
    if (method === 'DOM.resolveNode') return {object:{objectId:'button'}};
    if (method === 'Runtime.callFunctionOn' && params.functionDeclaration.includes('Math.min(innerWidth')) return {result:{value:{left:0,top:0,right:100,bottom:600}}};
    if (method === 'Page.getLayoutMetrics') return {cssLayoutViewport:{clientWidth:800,clientHeight:600}};
    if (method === 'DOM.getNodeForLocation') { assert.ok(params.y < 600); return {backendNodeId:7}; }
    if (method === 'Input.dispatchMouseEvent' && params.type === 'mouseReleased') additions++;
    return {};
  }},{settings:{base:'http://127.0.0.1:8000'},approve:async info => {approvals.push(info);return allowed;}});
  hand.sites.add('https://shop.test'); hand.active = 1; hand.tabs.set(1,{grantedOrigin:'https://shop.test',attached:true,epoch:0,refs:new Map()});
  hand.signal = new AbortController().signal;
  hand.inspect = async () => ({name:'Add to cart',product:'Charger',connected:true,type:'submit',method:'post',form:'Cart',action:'https://shop.test/api/cart'});
  hand.effectWorld = async () => 1;
  const snapshot = () => ({url:'https://shop.test/p',title:'Charger',regions:additions ? ['status: Added to cart'] : [],counters:[{key:'/cart',text:`Cart ${additions}`} ]});
  hand.effectEval = async (_id,_context,fn) => fn === watchEffects ? snapshot() : [snapshot()];
  const core = hand.executeCore.bind(hand);
  let ref = 0;
  hand.executeCore = async (name,args,signal) => {
    if (name !== 'read_page') return core(name,args,signal);
    const t = hand.tabs.get(1); hand.invalidate(1);
    t.refs.set(`e${++ref}`,{name:'Add to cart',role:'button',backendDOMNodeId:7});
    return {text:`[e${ref}] button "Add to cart"`};
  };
  return {hand,approvals,commands,count:() => additions,allow:() => {allowed = true;}};
}

test('fresh-ref repeated add denial persists; new task resets denial and addition history', async () => {
  const {hand,approvals,count,allow} = shoppingHand();
  await hand.execute('read_page',{},hand.signal);
  const result = await hand.execute('click',{ref:'e1'},hand.signal);
  assert.equal(count(),1); assert.equal(approvals.length,0);
  assert.match(result.text,/e2/); assert.deepEqual(result.effect.counters,[{before:'Cart 0',after:'Cart 1'}]);
  await assert.rejects(hand.execute('click',{ref:'e2'},hand.signal),/denied/);
  assert.equal(count(),1); assert.equal(approvals[0].action,'Add another one? (already added once)');
  allow(); await assert.rejects(hand.execute('click',{ref:'e3'},hand.signal),/already denied/);
  assert.equal(count(),1); assert.equal(approvals.length,1);
  hand.own = async () => {}; hand.showOverlay = async () => {};
  await hand.begin(hand.signal);
  await hand.execute('read_page',{},hand.signal);
  await hand.execute('click',{ref:'e5'},hand.signal);
  assert.equal(count(),2); assert.equal(approvals.length,1);
  await hand.execute('click',{ref:'e6'},hand.signal);
  await hand.execute('click',{ref:'e7'},hand.signal);
  assert.equal(count(),4); assert.equal(approvals.length,3);
});

test('keyboard activation cannot bypass the add guard', async () => {
  const {hand,approvals,count} = shoppingHand();
  await hand.execute('read_page',{},hand.signal);
  await hand.execute('click',{ref:'e1'},hand.signal);
  const send = hand.a.send;
  hand.a.send = async (id,method,params) => method === 'Runtime.evaluate' ? {result:{objectId:'button'}} : send(id,method,params);
  for (const key of ['Enter','Space']) await assert.rejects(hand.execute('press',{key},hand.signal),/denied/);
  assert.equal(count(),1); assert.equal(approvals.length,1);
});

test('dialogs and their controls outrank page clutter and survive product queries', () => {
  const nodes = Array.from({length:100},(_,i) => ({nodeId:`n${i}`,role:{value:'StaticText'},name:{value:'clutter '.repeat(50)}}));
  nodes.push({nodeId:'dialog',role:{value:'dialog'},childIds:['text','close']},
    {nodeId:'text',role:{value:'StaticText'},name:{value:'Added to cart'}},
    {nodeId:'close',role:{value:'button'},name:{value:'Close dialog'},backendDOMNodeId:1});
  for (const query of ['', 'charger']) {
    const result = outlineAX(nodes,{query,limit:900});
    assert.match(result.text,/Added to cart/); assert.match(result.text,/Close dialog/); assert.equal(result.refs.size,1);
  }
});
