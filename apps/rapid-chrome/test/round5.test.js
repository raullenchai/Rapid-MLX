import test from 'node:test';
import assert from 'node:assert/strict';
import { BrowserHand, sameClickControl, canActivate } from '../src/cdp.js';
import { classifyTarget, safeURL } from '../src/safety.js';
import { outlineAX } from '../src/ax.js';

const base = 'http://127.0.0.1:8000';
function setup(send = async () => ({}), approve = async () => true) {
  const commands = [];
  const hand = new BrowserHand({get:async () => ({url:'https://shop.test/product',title:'Shop'}),send:async (_,method,params) => {
    commands.push({method,params});
    return send(method,params);
  }},{settings:{base},approve});
  hand.sites.add('https://shop.test');
  hand.active = 1; hand.tabs.set(1,{attached:true,epoch:0,refs:new Map()});
  hand.signal = new AbortController().signal;
  return {hand,commands};
}

test('Send approval names the submitter for fill+submit and Enter, and the control for click', async () => {
  for (const [action,options] of [['Submit form',{fill:true,submit:true}],['Press Enter',{submit:true}],['click',{}]]) {
    const cards = [];
    const {hand} = setup(undefined,async card => { cards.push(card); return false; });
    hand.inspect = async () => ({connected:true,name:action === 'click' ? 'Send' : 'Message',submitter:'Send',method:'post'});
    await assert.rejects(hand.gateTarget(1,{epoch:0},action,options),/denied/);
    assert.match(cards[0].action,/'Send' on shop.test/);
    assert.equal(cards[0].target,'Send');
  }
});

test('non-submitting disclosures ignore POST context but explicit consequential language anywhere in the label wins', () => {
  const context = {role:'button',method:'post',type:'button',form:'Book now Confirm purchase'};
  for (const target of [
    {name:'Flight details. Leaves San Francisco International Airport'},
    {name:'Details about booking',expanded:'true'},
    {name:'View details '+ 'x'.repeat(100)}
  ]) {
    assert.equal(classifyTarget({...context,...target}).consequential,false);
    assert.equal(classifyTarget({...context,...target,type:'submit'}).consequential,true);
  }
  for (const name of ['Buy Now','Book flight','More options to book a flight','Flight details. Select and book','View details '+ 'x'.repeat(100) + ' buy now','Confirm purchase','Send','Delete account'])
    assert.equal(classifyTarget({...context,name}).consequential,true);
});

test('malformed hosts are rejected before site approval or tab creation', async () => {
  let approvals = 0;
  const {hand,commands} = setup(undefined,async () => { approvals++; return true; });
  for (const url of ['https://resy.com%20cities','https://resy.com cities','https://resy.com%2ecities','https://-resy.com','resy.com%20cities']) {
    assert.throws(() => safeURL(url,base),/blocked_url:.*(?:host|URL)/);
    await assert.rejects(hand.execute('open_tab',{url},hand.signal),/blocked_url/);
  }
  assert.equal(approvals,0); assert.equal(commands.length,0);
  assert.equal(safeURL('https://resy.com/cities?q=New%20York',base).hostname,'resy.com');
});

// Minimal DOM fixtures execute the actual browser-side predicate through fake CDP.
function node(parent = null, tag = 'SPAN', group = false) {
  const e = {parent,tagName:tag,children:[],isConnected:true,disabled:false,
    getBoundingClientRect:() => ({left:0,top:0,right:100,bottom:100,width:100,height:100}),
    contains(other) { return other === this || this.children.some(c => c.contains(other)); },
    closest(selector) {
      if (selector === 'button,a,input,[role="button"],[role="option"],label') return ['BUTTON','A','INPUT','LABEL'].includes(tag) ? this : parent?.closest(selector) || null;
      if (selector === '.a-button,[data-button],[role="group"]') return group ? this : parent?.closest(selector) || null;
      return null;
    },
    querySelectorAll() { return this.children.flatMap(c => [...(['BUTTON','A','INPUT'].includes(c.tagName) ? [c] : []),...c.querySelectorAll()]); }
  };
  parent?.children.push(e); return e;
}
async function clickCase(targetNode, hitNode, {fallback = false, hitError = false, stale = false} = {}) {
  const target = {objectId:'approved',backendDOMNodeId:7,epoch:0};
  const {hand,commands} = setup(async (method,p) => {
    if (method === 'Runtime.callFunctionOn' && p.functionDeclaration.includes('Math.min(innerWidth')) return {result:{value:{left:0,top:0,right:100,bottom:100}}};
    if (method === 'Page.getLayoutMetrics') return {cssLayoutViewport:{pageY:1000,clientWidth:800,clientHeight:600}};
    if (method === 'DOM.getNodeForLocation') { assert.equal(p.y,50); if (hitError) throw new Error('No node found at given location'); return {backendNodeId:9}; }
    if (method === 'DOM.resolveNode') return {object:{objectId:'hit'}};
    if (method === 'DOM.focus' && stale) hand.invalidate(1);
    if (method === 'Runtime.callFunctionOn') {
      assert.equal(p.objectId,'approved');
      if (p.functionDeclaration === sameClickControl.toString()) return {result:{value:sameClickControl.call(targetNode,hitNode,50,50)}};
      if (p.functionDeclaration === canActivate.toString()) return {result:{value:fallback}};
      assert.match(p.functionDeclaration,/this.click\(\)/); return {result:{value:undefined}};
    }
    return {};
  });
  return {hand,commands,run:() => hand.click(1,target)};
}

test('fake CDP accepts descendants, shared clickable ancestor, label-for and Amazon input wrappers', async () => {
  const button = node(null,'BUTTON'), child = node(button), sibling = node(button);
  const wrapper = node(null,'SPAN',true), input = node(wrapper,'INPUT'), text = node(wrapper);
  const label = node(null,'LABEL'); label.control = input;
  for (const [target,hit] of [[button,child],[child,sibling],[child,button],[input,label],[input,text],[text,input]]) {
    const c = await clickCase(target,hit);
    assert.equal(await c.run(),'pointer');
    assert.equal(c.commands.filter(c => c.method === 'Input.dispatchMouseEvent').length,2);
    assert.ok(!c.commands.some(c => c.method === 'DOM.focus'));
  }
});

test('fake CDP never clicks a different form control or unrelated overlay', async () => {
  const form = node(null,'FORM',true), add = node(form,'INPUT'), buy = node(form,'INPUT');
  for (const hit of [buy,node(null,'IFRAME')]) {
    const c = await clickCase(add,hit);
    await assert.rejects(c.run(),/obscured_target/);
    assert.ok(!c.commands.some(c => c.method.startsWith('Input.')));
  }
});

test('fake CDP fallback focuses and activates only the approved object; stale target blocks', async () => {
  for (const hitError of [false,true]) {
    const c = await clickCase(node(null,'INPUT'),node(null,'IFRAME'),{fallback:true,hitError});
    assert.match(await c.run(),/DOM activation of the approved element/);
    assert.equal(c.commands.find(c => c.method === 'DOM.focus').params.backendNodeId,7);
    assert.ok(!c.commands.some(c => c.method.startsWith('Input.')));
    assert.equal(c.commands.filter(c => c.params?.functionDeclaration?.includes('this.click()')).length,1);
  }
  const c = await clickCase(node(),node(),{fallback:true,stale:true});
  await assert.rejects(c.run(),/stale_ref/);
  assert.ok(!c.commands.some(c => c.params?.functionDeclaration?.includes('this.click()')));
});

test('Continue rechecks an unresolved challenge before returning any page or refs', async () => {
  let reads = 0, pauses = 0;
  const {hand} = setup(async method => {
    if (method === 'Accessibility.getFullAXTree') return {nodes:++reads < 3 ? [{name:{value:'Verify you are human'}}] : []};
    if (method === 'Page.getFrameTree') return {};
    return {};
  });
  hand.challenge = async info => { pauses++; assert.match(info.message,/shop.test.*Complete the verification/); };
  const result = await hand.execute('read_page',{},hand.signal);
  assert.equal(reads,3); assert.equal(pauses,2); assert.match(result.text,/Page data/);
});

test('dialog Done remains visible before calendar grids fill the outline budget', () => {
  const nodes = [{nodeId:'dialog',role:{value:'dialog'},name:{value:'Dates'},childIds:Array.from({length:100},(_,i) => String(i))}];
  for (let i=0;i<100;i++) nodes.push({nodeId:String(i),backendDOMNodeId:i+1,role:{value:'button'},name:{value:i === 99 ? 'Done' : `November ${i} date with price $500`}});
  const result = outlineAX(nodes,{limit:1000});
  assert.match(result.text,/button "Done"/);
  assert.ok(result.text.indexOf('"Done"') < result.text.indexOf('November'));
});
