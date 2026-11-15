import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand } from '../src/cdp.js';
import { classifyTarget } from '../src/safety.js';

// r14 flight-google-1, round 9 click e385: DOM text differs from the AX name.
const trace = {tag:'DIV',type:'',role:'button',name:'22\n$1,120',method:'',action:'',
  landmark:'DIV dialog ',heading:'',suggestionQuery:'',singleQuery:false,
  dateCell:'22\n$1,120',dateCellAX:'Sunday, November 22, 2026 ????',datePicker:false};
const names = ['22\n$1,120','22 $1,120','22, $1,120',
  'Sunday, November 22, 2026 $1,120 lowest fare',
  'Sunday, November 22, 2026 , 1123 US dollars',
  '22 lowest fare','22 unavailable','22, 3 seats available'];

test('r14 date cells with fare and availability hints are preparatory', () => {
  for (const name of names) for (const role of ['button','gridcell']) {
    const target = {...trace,name,role,dateCell:name};
    assert.equal(classifyTarget(target).consequential,false,name);
    assert.equal(classifyTarget(target).financial,false,name);
    assert.equal(classifyTarget({...target,dateCellAX:'',datePicker:true}).consequential,false,name);
  }
});

test('date fare exceptions require calendar evidence and preserve effect gates', () => {
  for (const name of names) {
    assert.equal(classifyTarget({...trace,name,dateCell:'',dateCellAX:''}).consequential,true,name);
    assert.equal(classifyTarget({...trace,name,dateCell:'Product 22',dateCellAX:''}).consequential,true,name);
    for (const patch of [{name:`${name} Buy tickets`},{name:`${name} Confirm booking`},
      {name:`${name} Send`},{name:`${name} Delete account`},{paymentForm:true},{sensitiveForm:true},
      {formName:'Save account address'},{method:'post'},{type:'submit'},
      {action:'https://example.test/checkout'},{submitter:'Confirm purchase'},{longName:true}]) {
      const c = classifyTarget({...trace,name,...patch});
      assert.ok(c.consequential || c.financial,JSON.stringify({name,...patch}));
    }
  }
  for (const name of ['$1,120','32 $1,120','0 $1,120','22 products $1,120']) {
    assert.equal(classifyTarget({...trace,name}).consequential,true,name);
  }
});

test('a price alone is not financial; commits and checkout context remain financial', () => {
  for (const name of ['$1,120','22 $1,120','Fare $1,120']) {
    const target = {name,role:'button',type:'button'};
    assert.equal(classifyTarget(target).financial,false,name);
    for (const patch of [{name:`Buy ${name}`},{name:`Pay ${name}`},
      {formName:'Payment'},{action:'https://example.test/checkout'}]) {
      assert.equal(classifyTarget({...target,...patch}).financial,true,JSON.stringify(patch));
    }
  }
});

test('r14 live inspection and gate accept the fare cell without an approval', async () => {
  const doc = {location:{href:'https://www.google.com/travel/flights'},querySelectorAll:()=>[],getElementById:()=>null};
  const cell = {innerText:trace.dateCell,ownerDocument:doc,getAttribute:()=>''};
  const e = {tagName:'DIV',type:'',id:'',name:'',textContent:trace.name,innerText:trace.name,isConnected:true,
    ownerDocument:doc,getAttribute:n=>n==='role'?'button':'',
    closest:selector=>selector==='[role=gridcell],td'?cell:null};
  const hand = new BrowserHand({get:async()=>({url:doc.location.href})},
    {settings:{base:'http://127.0.0.1:8000'},approve:async()=>assert.fail('date preparation prompted')});
  hand.call=async(_id,_object,fn)=>runInNewContext(`(${fn}).call(e)`,{e});
  await hand.gateTarget(1,{objectId:'target',role:'button',name:trace.dateCellAX},'click');
  assert.equal(hand.lastTargetContext.name,trace.name);
  assert.equal(hand.lastTargetContext.dateCell,trace.dateCell);
  assert.equal(hand.lastTargetContext.dateCellAX,trace.dateCellAX);
});
