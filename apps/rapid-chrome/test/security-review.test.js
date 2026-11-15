import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand, sensitiveDocument, fillNative } from '../src/cdp.js';
import { classifyTarget, sensitivePage } from '../src/safety.js';
import { assembleMessages } from '../src/agent.js';
import { boundedStored, storedBytes, runSnapshot } from '../src/ui-state.js';

const base = 'http://127.0.0.1:8000';
function setup({info = {connected:true,name:'Send',values:[{name:'amount',value:'1'}]}, approve = async () => false, handler = () => ({})} = {}) {
  const commands = [], cards = [];
  let url = 'https://shop.test/checkout';
  const hand = new BrowserHand({get:async () => ({url,title:'Checkout'}),send:async (_,method,p) => {
    commands.push({method,p});
    const result = await handler(method,p);
    if (result !== undefined) return result;
    return {};
  }},{settings:{base,vision:true},approve:async card => {cards.push(card);return approve(card);}});
  hand.signal = new AbortController().signal;
  hand.active = 1; hand.sites.add('https://shop.test');
  hand.tabs.set(1,{attached:true,mainFrame:'main',epoch:0,refs:new Map([['e1',{backendDOMNodeId:7,role:'button',name:'Pay'}]])});
  hand.inspect = async () => structuredClone(info);
  return {hand,cards,commands,setURL:value => {url=value;},info};
}

test('P1 disclosure ARIA cannot suppress consequential intent or sensitive-form ambiguity', async () => {
  for (const expanded of [false,true,'false','true']) {
    for (const name of ['Buy now','Pay','Place order','Submit','Send','Delete','Transfer','Book','Confirm','Add to cart and buy now','Details. Buy now','Show more '+ 'x'.repeat(100)+' Pay']) {
      const {hand,cards,commands} = setup({info:{connected:true,name,expanded}});
      await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/denied|financial_handover/);
      assert.equal(cards.length,classifyTarget({name}).financial ? 0 : 1,`${name} ${expanded}`);
      assert.ok(!commands.some(c => c.method.startsWith('Input.')));
    }
    for (const role of ['button','menuitem','checkbox','option'])
      assert.equal(classifyTarget({name:'Confirm',role,expanded}).consequential,true);
    assert.equal(classifyTarget({name:'Continue',type:'submit',method:'post',expanded}).consequential,true);
    assert.equal(classifyTarget({name:'Continue',role:'button',expanded}).consequential,true);
    for (const name of ['Continue','Next','Go']) assert.equal(classifyTarget({name,expanded,sensitiveForm:true}).consequential,true);
  }
});

function keyboard({frame = false, shadow = false, unresolved = false, changed = false, approve} = {}) {
  let focusReads = 0;
  const fixture = setup({approve,handler:(m,p) => {
    if (m === 'Page.getFrameTree') return {frameTree:{frame:{id:'main'}}};
    if (m === 'Page.createIsolatedWorld') return {executionContextId:p.frameId === 'main' ? 1 : 2};
    if (m === 'Runtime.evaluate') {
      if (p.expression.includes('sensitiveDocument')) return {result:{value:false}};
      focusReads++;
      if (unresolved) return {result:{subtype:'null'}};
      if (frame && p.contextId === 1) return {result:{objectId:'iframe'}};
      if (shadow) {
        const leaf = {tagName:'BUTTON'};
        const value = runInNewContext(p.expression,{document:{activeElement:{shadowRoot:{activeElement:leaf}}}});
        assert.equal(value,leaf);
      }
      return {result:{objectId:changed && focusReads > (frame ? 2 : 1) ? 'changed' : 'pay'}};
    }
    if (m === 'DOM.describeNode') return {node:p.objectId === 'iframe' ? {backendNodeId:2,nodeName:'IFRAME',frameId:'child'} : {backendNodeId:p.objectId === 'changed' ? 8 : 7,nodeName:'BUTTON'}};
    return {};
  }});
  return fixture;
}

test('P1 Enter and Space after Tab classify iframe and open-shadow focus; denials persist across keyboard paths', async () => {
  for (const options of [{frame:true},{shadow:true}]) {
    const {hand,cards,commands} = keyboard(options);
    await hand.executeCore('press',{key:'Tab'},hand.signal);
    for (const key of ['Enter','Space']) await assert.rejects(hand.executeCore('press',{key},hand.signal),/denied/);
    assert.equal(cards.length,1);
    assert.ok(!commands.some(c => c.method === 'Input.dispatchKeyEvent' && ['Enter',' '].includes(c.p.key)));
  }
});

test('P1 unresolved or changed focused control never receives keyboard activation', async () => {
  for (const options of [{unresolved:true},{changed:true,approve:async () => true}]) {
    const {hand,commands} = keyboard(options);
    await assert.rejects(hand.executeCore('press',{key:'Enter'},hand.signal),/unresolved_focus|focus_changed/);
    assert.ok(!commands.some(c => c.method === 'Input.dispatchKeyEvent'));
  }
});

test('P1 redirects and script navigations gate new main-frame origins; passive subframes continue', async () => {
  for (const redirectedRequestId of [undefined,'prior']) {
    const {hand,cards,commands} = setup();
    await hand.event(1,'Fetch.requestPaused',{frameId:'main',requestId:'r',redirectedRequestId,request:{url:'https://private.test/app'}});
    assert.equal(cards.length,1);
    assert.equal(commands.at(-1).method,'Fetch.failRequest');
    await hand.event(1,'Fetch.requestPaused',{frameId:'ad',requestId:'ad',request:{url:'https://ad.test/'}});
    assert.equal(cards.length,1); assert.equal(commands.at(-1).method,'Fetch.continueRequest');
  }
  const {hand,cards} = setup();
  await hand.event(1,'Page.frameNavigated',{frame:{id:'main',url:'https://private.test/app'}});
  assert.equal(cards.length,1); assert.equal(hand.tabs.size,0);
});

test('P1 read_page and actions on unapproved origins request approval before DOM access', async () => {
  for (const tool of ['read_page','screenshot','click','fill','press','list_tabs']) {
    const {hand,cards,commands,setURL} = setup();
    setURL('https://private.test/account?token=private');
    await assert.rejects(hand.executeCore(tool,{ref:'e1',text:'hello',key:'Enter'},hand.signal),/denied/);
    assert.equal(cards.length,1);
    assert.equal(commands.length,0);
    assert.doesNotMatch(JSON.stringify(cards),/token=private/);
  }
});

function formFixture() {
  const fields = [
    {tagName:'INPUT',name:'amount',type:'hidden',value:'1'},
    {tagName:'INPUT',name:'currency',type:'text',value:'USD'},
    {tagName:'INPUT',name:'recipient',type:'text',value:'Alice'},
    {tagName:'INPUT',name:'quantity',type:'number',value:'1'},
    {tagName:'INPUT',name:'address',type:'text',value:'10 Main St'},
    {tagName:'SELECT',name:'destination',type:'select-one',value:'a',selectedOptions:[{value:'a'}]},
    {tagName:'INPUT',name:'csrf_token',type:'hidden',value:'never-display-this'}
  ];
  const form = {elements:fields,innerText:'Submit refund',method:'post',action:'https://shop.test/refund',getAttribute:() => null,ownerDocument:{getElementById:() => null}};
  const node = {form,tagName:'BUTTON',type:'submit',innerText:'Submit refund',textContent:'Submit refund',isConnected:true,id:'pay',ownerDocument:{getElementById:() => null},getAttribute:() => null,closest:() => null};
  return {fields,node};
}

test('approval form details show ordinary hidden ids and mask only sensitive-looking fields', async () => {
  const {fields,node} = formFixture();
  const ordinary = ['id','subscription_id','customer_id','account_id'];
  const sensitive = ['card_number','cvv','password','ssn','account_number','routing_number','csrf_token','access_token','secret'];
  for (const name of [...ordinary,...sensitive]) fields.push({tagName:'INPUT',name,type:'hidden',value:`value-${name}`});
  fields.push({tagName:'INPUT',name:'credential',type:'password',value:'private-password'});
  const {hand} = setup();
  delete hand.inspect;
  hand.call = async (_,__,fn) => runInNewContext(`(${fn}).call(node)`,{node});
  const info = await hand.inspect(1,{objectId:'submit'});
  for (const name of ordinary) assert.ok(info.details.includes(`${name}: value-${name}`),name);
  for (const name of sensitive) {
    assert.ok(info.details.includes(`${name}: [masked]`),name);
    assert.ok(!info.details.includes(`value-${name}`),name);
  }
  assert.ok(info.details.includes('credential: [masked]'));
  assert.ok(!info.details.includes('private-password'));
  assert.equal(info.values.find(v => v.name === 'id').value,'value-id','approval binding retains raw ids');
});

test('P1 actual inspection binds hidden, input and select transaction values and masks sensitive card details', async () => {
  for (const index of [0,1,2,3,4,5,6,'selected']) {
    const {fields,node} = formFixture();
    const {hand,cards,commands} = setup({approve:async () => {if (index === 'selected') fields[5].selectedOptions = [{value:'b'}]; else fields[index].value='10000';return true;}});
    delete hand.inspect;
    hand.call = async (_,__,fn) => runInNewContext(`(${fn}).call(node)`,{node});
    await assert.rejects(hand.gateTarget(1,{epoch:0,objectId:'pay'},'click'),/stale_ref/);
    assert.match(cards[0].action,/amount: 1.*currency: USD.*recipient: Alice/);
    assert.match(cards[0].action,/quantity: 1.*address: 10 Main St/);
    assert.doesNotMatch(JSON.stringify(cards),/never-display-this/);
    assert.ok(!commands.some(c => c.method.startsWith('Input.')));
  }
});

test('P1 form mutations after card revalidation still invalidate approval immediately before pointer dispatch', async () => {
  const info = {connected:true,name:'Submit refund',values:[{name:'amount',value:'1'}],details:'amount: 1'};
  const {hand,commands} = setup({info,approve:async () => true,handler:(m) => {
    if (m === 'DOM.getNodeForLocation') {info.values[0].value='10000';return {backendNodeId:7};}
    return {};
  }});
  hand.call = async () => ({left:0,top:0,right:100,bottom:100});
  const target = {objectId:'pay',backendDOMNodeId:7,epoch:0};
  await hand.gateTarget(1,target,'click');
  await assert.rejects(hand.click(1,target),/stale_ref/);
  assert.ok(!commands.some(c => c.method.startsWith('Input.')));
});

test('P1 opaque financial transactions hand over without offering a label-only purchase approval', async () => {
  const {hand,cards,commands} = setup({info:{connected:true,name:'Buy now',values:[]},approve:async () => true});
  await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/financial_handover/);
  assert.equal(cards.length,0);
  assert.equal(commands.length,0);
});

test('P2 focus redirects on focus or select-all cannot insert text into a different recipient', async () => {
  for (const phase of ['focus','selectAll']) {
    let focused = 7;
    const {hand,commands} = setup({info:{connected:true,name:'Search',type:'search'},handler:(m,p) => {
      if (m === 'DOM.resolveNode') return {object:{objectId:'input'}};
      if (m === 'Runtime.callFunctionOn' && p.functionDeclaration === fillNative.toString()) return {result:{value:{handled:false}}};
      if (m === 'DOM.focus' && phase === 'focus') focused=8;
      if (m === 'Input.dispatchKeyEvent' && p.commands && phase === 'selectAll') focused=8;
      if (m === 'Runtime.evaluate') return {result:{objectId:'focus',value:false}};
      if (m === 'DOM.describeNode') return {node:{backendNodeId:focused,nodeName:'INPUT'}};
      return {};
    }});
    hand.tabs.get(1).refs.set('e1',{backendDOMNodeId:7,name:'Search',role:'textbox'});
    await assert.rejects(hand.executeCore('fill',{ref:'e1',text:'private'},hand.signal),/focus_changed/);
    assert.ok(!commands.some(c => c.method === 'Input.insertText'));
  }
});

test('P2 current-tab URL tokens and title never enter model messages before explicit sharing', () => {
  const messages = assembleMessages([],'Hi',{id:9,title:'Private document',url:'https://private.test/?token=secret'});
  assert.match(messages[0].content,/Current composer tab: available to share/);
  assert.doesNotMatch(JSON.stringify(messages),/Private document|private.test|token=secret/);
});

test('P2 stored tool arguments, results and individual trace events have UTF-8 byte limits', () => {
  const huge = '🔒'.repeat(100000);
  const snapshot = runSnapshot({messages:[],steps:[{id:'a',name:'fill',args:{text:huge},result:huge}],status:'Ready'});
  assert.equal(snapshot.steps[0].id,'a'); assert.equal(snapshot.steps[0].name,'fill');
  assert.equal(typeof snapshot.steps[0].result,'string');
  assert.ok(storedBytes(JSON.stringify(snapshot.steps[0].args)) <= 4096);
  assert.ok(storedBytes(JSON.stringify(snapshot.steps[0].result)) <= 8192);
  assert.ok(storedBytes(JSON.stringify(boundedStored({kind:'response',tool_calls:[{arguments:huge}],text:huge},16384))) <= 16384);
});

test('P2 sensitive vault and banking login URLs and sensitive fields hand over before reads or screenshots', async () => {
  for (const url of ['https://my.1password.com/vault','https://vault.bitwarden.com/','https://lastpass.com/vault','https://pass.proton.me/','https://secure.chase.com/login','https://bankofamerica.com/signin']) assert.equal(sensitivePage(url),true,url);
  for (const url of ['https://shop.test/password-guide','https://chase.com/about','https://bitwarden.com.evil.test/']) assert.equal(sensitivePage(url),false,url);
  for (const tool of ['read_page','screenshot','click']) {
    const {hand,commands,setURL} = setup(); setURL('https://vault.bitwarden.com/');
    await assert.rejects(hand.executeCore(tool,{ref:'e1'},hand.signal),/blocked_sensitive_page/);
    assert.equal(commands.length,0);
    const fields = setup({handler:(m,p) => m === 'Runtime.evaluate' ? {result:{value:true}} : {}});
    await assert.rejects(fields.hand.executeCore(tool,{ref:'e1'},fields.hand.signal),/blocked_sensitive_page/);
    assert.ok(!fields.commands.some(c => ['Page.captureScreenshot','Accessibility.getFullAXTree','Input.dispatchMouseEvent'].includes(c.method)));
  }
  for (const field of [{type:'password'},{autocomplete:'cc-number'},{name:'routing_number'}]) {
    const e = {tagName:'INPUT',getClientRects:() => [{}],...field};
    assert.equal(runInNewContext(`(${sensitiveDocument})()`,{document:{querySelectorAll:() => [e]},getComputedStyle:() => ({visibility:'visible'})}),true);
  }
});

test('benign add-to-cart, filters, pagination, show-more and search remain unattended', async () => {
  for (const target of [
    {name:'Add to cart',type:'submit',method:'post',form:'Cart',action:'https://shop.test/api/cart'},
    {name:'Price under $50',role:'checkbox',form:'Filters'},
    {name:'Filters',role:'button',expanded:false,method:'post'},
    {name:'Apply filters',role:'button',method:'post'},
    {name:'Next',role:'button',method:'post'},
    {name:'Next page',role:'button',method:'post'}, {name:'Show more',role:'button',expanded:false,method:'post'},
    {name:'Search',type:'submit',method:'post',form:'Search products'}
  ]) {
    assert.equal(classifyTarget(target).consequential,false,target.name);
    const {hand,cards} = setup({info:{connected:true,...target}});
    await hand.gateTarget(1,{epoch:0},'click'); assert.equal(cards.length,0);
  }
});


test('P1 a new turn cannot silently approve an origin changed since the tab grant', async () => {
  const {hand,cards,setURL} = setup();
  hand.tabs.get(1).grantedOrigin = 'https://shop.test';
  hand.own = async () => {}; hand.showOverlay = async () => {};
  setURL('https://private.test/account');
  await hand.begin(hand.signal);
  assert.equal(hand.sites.has('https://private.test'),false);
  await assert.rejects(hand.current({}),/denied/);
  assert.equal(cards.length,1);
});

test('P2 hidden login fields do not hand over an otherwise benign shopping page', () => {
  const field = {tagName:'INPUT',type:'password',getClientRects:() => []};
  assert.equal(runInNewContext(`(${sensitiveDocument})()`,{document:{querySelectorAll:() => [field]},getComputedStyle:() => ({visibility:'visible'})}),false);
});

test('P2 sensitive fields inside frames and open shadow roots force handover before screenshots', async () => {
  const {hand,commands} = setup({handler:(method,p) => {
    if (method === 'Page.getFrameTree') return {frameTree:{frame:{id:'main'},childFrames:[{frame:{id:'child'}}]}};
    if (method === 'Page.createIsolatedWorld') return {executionContextId:p.frameId === 'main' ? 1 : 2};
    if (method === 'Runtime.evaluate') return {result:{value:p.contextId === 2}};
    return {};
  }});
  await assert.rejects(hand.executeCore('screenshot',{},hand.signal),/blocked_sensitive_page/);
  assert.ok(!commands.some(c => c.method === 'Page.captureScreenshot'));
  const field = {tagName:'INPUT',type:'password',getClientRects:() => [{}]};
  const host = {tagName:'LOGIN-FORM',shadowRoot:{querySelectorAll:() => [field]}};
  assert.equal(runInNewContext(`(${sensitiveDocument})()`,{document:{querySelectorAll:() => [host]},getComputedStyle:() => ({visibility:'visible'})}),true);
});

test('re-review: cart labels cannot exempt a purchase form, using production DOM inspection', async () => {
  for (const name of ['Add to cart','Add 1 to order • $11.95','Add 1 ct Eggs']) {
    const {node} = formFixture();
    node.innerText = node.textContent = name;
    node.form.innerText = name;
    node.form.action = 'https://shop.test/purchase';
    const {hand,cards,commands} = setup({approve:async () => true});
    delete hand.inspect;
    hand.call = async (_,__,fn) => runInNewContext(`(${fn}).call(node)`,{node});
    await assert.rejects(hand.gateTarget(1,{epoch:0,objectId:'cart'},'click'),error => {
      assert.equal(error.handover.kind,'payment');
      assert.match(error.handover.reason,/^Review and press/);assert.doesNotMatch(error.handover.message+error.handover.reason,/Your turn/);
      assert.match(error.handover.message,/amount: 1/);
      return /financial_handover/.test(error.message);
    });
    assert.equal(cards.length,0);
    assert.ok(!commands.some(c => c.method.startsWith('Input.')));
  }
});

test('re-review: financial commits always hand over regardless of roles, fields or opaque JS amount changes', async () => {
  for (const name of ['Buy','Purchase','Pay','Place order','Checkout complete','Confirm order','Donate','Subscribe','Authorize','Authorize payment','Finalize','Complete purchase','Transfer','Send money','Tip','购买','付款','acheter','kaufen']) {
    for (const values of [[],[{name:'csrf_token',type:'hidden',value:'same'}],[{name:'amount',value:'1'}]]) {
      let jsAmount = 1, charged = 0;
      const {hand,cards,commands,info} = setup({info:{connected:true,name,role:'menuitem',values,details:'csrf_token: [masked]'},approve:async () => true,
        handler:m => {if (m === 'DOM.resolveNode') return {object:{objectId:'commit'}};if (m === 'Input.dispatchMouseEvent') charged = jsAmount;return {};}});
      // The human pauses with the card open; the page changes only JS state.
      for (const amount of [1,10000]) {
        jsAmount = amount;
        await assert.rejects(hand.executeCore('click',{ref:'e1'},hand.signal),error => {
          assert.match(error.handover.reason,new RegExp(`press ${name} yourself`));
          assert.match(error.handover.message,/csrf_token: \[masked\]/);
          return /financial_handover/.test(error.message);
        });
      }
      assert.deepEqual(info.values,values);
      assert.equal(charged,0); assert.equal(cards.length,0);
      assert.ok(!commands.some(c => c.method.startsWith('Input.')));
    }
  }
});

test('re-review: destructive names require approval across every activation role', async () => {
  for (const role of ['button','link','menuitem','option','switch']) {
    for (const name of ['Deactivate account','Close account','Delete','Erase all','Remove account','Unsubscribe','Cancel subscription','Cancel membership','Revoke','Reset']) {
      const {hand,cards,commands} = setup({info:{connected:true,name,role}});
      await assert.rejects(hand.gateTarget(1,{epoch:0,role},'click'),/denied/);
      assert.equal(cards.length,1,`${role}: ${name}`);
      assert.equal(cards[0].titleKey,'allowDelete');
      assert.ok(!commands.some(c => c.method.startsWith('Input.')));
    }
  }
});

test('re-review: cart exemption requires a benign name and form context', async () => {
  for (const name of ['Add to cart','Add 1 to order','Add 1 ct Eggs']) {
    for (const context of [{action:'https://shop.test/checkout'},{action:'https://shop.test/pay'},{action:'https://shop.test/payment'},{action:'https://shop.test/order.php',method:'post'},{action:'https://shop.test/order',method:'post'},{form:'Confirm purchase'},{sensitiveForm:true},{paymentForm:true}]) {
      const {hand,cards} = setup({info:{connected:true,name,role:'button',...context}});
      await assert.rejects(hand.gateTarget(1,{epoch:0},'click'),/financial_handover|denied/);
    }
    const {hand,cards} = setup({info:{connected:true,name,role:'button',type:'submit',method:'post',action:'https://shop.test/api/cart',form:'Cart'}});
    await hand.gateTarget(1,{epoch:0},'click');
    assert.equal(cards.length,0);
  }
});


test('re-review: production inspection detects payment fields on a misleading cart form', async () => {
  for (const field of [{name:'routing_number'},{name:'card_holder'},{autocomplete:'cc-number'}]) {
    const {node} = formFixture();
    node.innerText=node.textContent=node.form.innerText='Add to cart';
    node.form.action='https://shop.test/api/cart';
    node.form.elements=[{tagName:'INPUT',type:'text',value:'123',...field}];
    const {hand,cards} = setup();delete hand.inspect;
    hand.call=async (_,__,fn)=>runInNewContext(`(${fn}).call(node)`,{node});
    await assert.rejects(hand.gateTarget(1,{epoch:0,objectId:'cart'},'click'),/denied/);
    assert.equal(cards.length,1);
  }
});
