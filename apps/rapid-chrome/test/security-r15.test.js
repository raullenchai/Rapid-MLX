import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand } from '../src/cdp.js';
import { classifyTarget, preparatoryContext } from '../src/safety.js';

const settings = {base:'http://127.0.0.1:8000'};
const lookup = {tag:'BUTTON',type:'submit',role:'button',name:'Go',method:'get',
  action:'https://example.test/',heading:'Search recipients',singleQuery:true};

// Run the production DOM inspector against both reported form structures.
function fixture({name = 'Go',method = 'get',heading = 'Search recipients',action = 'https://example.test/'} = {}) {
  const doc = {location:{href:'https://example.test/'},querySelectorAll:()=>[],getElementById:()=>null};
  const title = {textContent:heading};
  const scope = {tagName:'SECTION',getAttribute:()=>'',querySelector:()=>title};
  const field = {tagName:'INPUT',type:'text',name:'q',id:'',value:'hello',autocomplete:''};
  const form = {ownerDocument:doc,method,action,innerText:'Send message',getAttribute:n=>n==='aria-label'?'Send message':''};
  const node = {ownerDocument:doc,form,tagName:'BUTTON',type:'submit',id:'',name:'',
    textContent:name,innerText:name,isConnected:true,getAttribute:n=>n==='role'?'menuitem':'',
    closest:selector=>selector.includes('section')?scope:null};
  form.elements = [field,node];
  let approvals = 0;
  const hand = new BrowserHand({get:async()=>({url:doc.location.href})},
    {settings,approve:async()=>{approvals++;return false;}});
  hand.call = async(_id,_object,fn)=>runInNewContext(`(${fn}).call(node)`,{node});
  return {hand,title,form,approvals:()=>approvals};
}

test('r15 preparatory and query/browse paths reject consequential context in every field', () => {
  for (const field of ['form','formName','submitter','action','label','id']) {
    for (const intent of ['Send message','Delete message','Deactivate','Donate']) {
      for (const singleQuery of [true,false]) {
        const target = {...lookup,singleQuery,[field]:field==='action'?`https://example.test/${intent.replaceAll(' ','/')}`:intent};
        assert.equal(preparatoryContext(target),false,JSON.stringify(target));
        assert.equal(classifyTarget(target).consequential,true,JSON.stringify(target));
      }
    }
  }
});

test('r15 production GET Go in Send message form requires approval', async () => {
  const f = fixture();
  await assert.rejects(f.hand.gateTarget(1,{objectId:'target',role:'button'},'click'),/denied:/);
  assert.equal(f.approvals(),1);
});

test('r15 production Submit search menuitem in POST Send message form requires approval', async () => {
  const f = fixture({name:'Submit search',method:'post',action:'https://example.test/send'});
  await assert.rejects(f.hand.gateTarget(1,{objectId:'target',role:'menuitem'},'click'),/denied:/);
  assert.equal(f.approvals(),1);
});

test('r15 denial survives heading changes and later fully benign classification', async () => {
  const f = fixture({heading:'Review recipients'});
  await assert.rejects(f.hand.gateTarget(1,{objectId:'old',role:'button'},'click'),/denied:/);
  f.title.textContent = 'Search recipients';
  await assert.rejects(f.hand.gateTarget(1,{objectId:'fresh',role:'button'},'click'),/already denied/);
  // Also prove denial enforcement is independent of the classification fix.
  f.form.innerText = 'Search recipients';
  f.form.getAttribute = ()=>'';
  assert.equal(classifyTarget(await f.hand.inspect(1,{objectId:'fresh'})).consequential,false);
  await assert.rejects(f.hand.gateTarget(1,{objectId:'fresh',role:'button'},'click'),/already denied/);
  assert.equal(f.approvals(),1);
});

test('r15 Submit search exemption requires benign GET or methodless lookup context', () => {
  for (const role of ['button','checkbox','radio','tab','option','menuitem','switch']) {
    for (const method of ['', 'get', 'GET']) {
      const target = {...lookup,name:'Submit search',role,method,form:'Search products',formName:'Search',submitter:'Submit search',action:'https://example.test/search'};
      assert.equal(classifyTarget(target).consequential,false,JSON.stringify(target));
      for (const patch of [{method:'post'},{method:'dialog'},{method:'put'},
        {formName:'Send message'},{submitter:'Send message'},{action:'https://example.test/send'},
        {label:'Delete'},{id:'send'},{paymentForm:true},{sensitiveForm:true}]) {
        assert.equal(classifyTarget({...target,...patch}).consequential,true,JSON.stringify({...target,...patch}));
      }
    }
  }
});

test('r15 sibling exemptions cannot hide consequential submit form context', () => {
  const names = ['Close','View cart','Filters','Reload','Sunday, November 22, 2026',
    'Add to cart','Go','Create event','Show more','Next page'];
  for (const name of names) for (const type of ['submit','image']) {
    for (const patch of [{form:'Send message'},{formName:'Send message'},
      {submitter:'Deactivate'},{action:'https://example.test/delete'},{label:'Send message'},{id:'send'}]) {
      const target = {...lookup,name,type,...patch};
      assert.equal(classifyTarget(target).consequential,true,JSON.stringify(target));
    }
  }
  for (const role of ['checkbox','radio','tab','option','menuitem','switch']) {
    for (const method of ['post','put','delete','dialog']) {
      const target = {...lookup,name:'Filters',role,method};
      assert.equal(classifyTarget(target).consequential,true,JSON.stringify(target));
    }
  }
});

test('a cart button whose own label is the submitter is not treated as form purchase context', () => {
  const base = {tag:'BUTTON',role:'button',type:'submit',method:'post',action:'/cart/add'};
  assert.equal(classifyTarget({...base,name:'Add 1 to order',submitter:'Add 1 to order'}).consequential,false);
  assert.equal(classifyTarget({...base,name:'Add 1 to order',submitter:'Add 1 to order',formName:'Send message'}).consequential,true);
  assert.equal(classifyTarget({...base,name:'Add 1 to order',submitter:'Add 1 to order',action:'/checkout'}).financial,true);
});
