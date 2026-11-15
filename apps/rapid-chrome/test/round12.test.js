import test from 'node:test';
import assert from 'node:assert/strict';
import { runInNewContext } from 'node:vm';
import { BrowserHand } from '../src/cdp.js';
import { classifyTarget } from '../src/safety.js';

const button = {tag:'BUTTON',type:'button',role:'button'};
// r12 target_context, plus structural facts from the saved form/AX outline.
const weather = {tag:'INPUT',type:'submit',role:'button',name:'Go',method:'get',action:'https://forecast.weather.gov/zipcity.php',singleQuery:true,form:'Local forecast by City, St or ZIP code',formName:'',submitter:'Go'};
const date = {tag:'DIV',type:'',role:'button',name:'22',landmark:'DIV dialog ',dateCell:'Sunday, November 22, 2026'};
const benign = [weather,date,
  {...button,name:'Submit search',landmark:'HEADER  ',heading:'Products for "airpods pro 2"'},
  {...button,name:'San Francisco',landmark:'MAIN  ',heading:'Discover Restaurants to Love in Sunnyvale, CA.'},
  {...button,name:'Location Sunnyvale, CA',landmark:'HEADER banner Main Resy',heading:'Popular Cities'},
  {tag:'SPAN',role:'button',name:'San Francisco Bay Area',landmark:'MAIN  ',heading:'Order groceries for delivery or pickup today'},
  {...button,name:'current address: 94303. Or select a new address.',landmark:'HEADER  '}
];
test('r12 benign approvals replay without weakening account, write or payment gates', () => {
  for (const target of benign) {
    assert.equal(classifyTarget(target).consequential,false,JSON.stringify(target));
    for (const patch of [{paymentForm:true},{sensitiveForm:true},{heading:'Saved account addresses'},
      {action:'https://example.test/profile/address',method:'post'}, {name:'Save address'},
      {name:'Delete account'},{name:'Confirm purchase'}, {longName:true}]) {
      const result=classifyTarget({...target,...patch});
      assert.ok(result.consequential || result.financial,JSON.stringify({...target,...patch}));
    }
  }
});
test('r12 query and calendar exceptions need positive structure, not generic Go or numeric buttons', () => {
  for (const target of [
    {...weather,singleQuery:false}, {...weather,method:'post'}, {...weather,action:'https://example.test/update'},
    {...weather,form:'Save location to address book'}, {...weather,submitter:'Send'},
    {...weather,submitter:'Confirm purchase'}, {...date,formName:'Save account address'},
    {...date,dateCell:''}, {...date,dateCell:'Product 22'},
    {...button,name:'Nov 22'}, {...button,name:'Continue',landmark:'DIV dialog Privacy'}
  ]) assert.equal(classifyTarget(target).consequential,true,JSON.stringify(target));
  assert.equal(classifyTarget({...button,name:'Nov 22',datePicker:true}).consequential,false);
});

// Exercise the actual DOM inspection function, not just classifier inputs.
test('r12 live inspection binds the single query and ancestor date cell without tracing values', async () => {
  const field = {tagName:'INPUT',type:'text',name:'inputstring',id:'inputstring',value:'Cupertino, CA',autocomplete:''};
  const form = {elements:[field],method:'get',action:weather.action,innerText:'Local forecast by City, St or ZIP code',getAttribute:()=>''};
  const cell = {innerText:'22',getAttribute:n=>n==='aria-label'?'Sunday, November 22, 2026':''};
  const e = {tagName:'INPUT',type:'submit',id:'btnSearch',name:'btnSearch',value:'Go',textContent:'',isConnected:true,form,
    getAttribute:()=>'',closest:selector=>selector==='[role=gridcell],td'?cell:null,
    ownerDocument:{location:{href:'https://www.weather.gov/'},querySelectorAll:()=>[],getElementById:()=>null}};
  form.ownerDocument=e.ownerDocument;
  const hand = new BrowserHand({get:async()=>({url:'https://www.weather.gov/'})},
    {settings:{base:'http://127.0.0.1:8000'},approve:async()=>assert.fail('preparation approval')});
  hand.call=async(_id,_object,fn)=>runInNewContext(`(${fn}).call(e)`,{e});
  const info=await hand.gateTarget(1,{objectId:'target',role:'button'},'click');
  assert.equal(info.singleQuery,true);
  assert.equal(hand.lastTargetContext.singleQuery,true);
  assert.ok(!JSON.stringify(hand.lastTargetContext).includes('Cupertino'));
  form.elements.push({...field,type:'email',name:'email'});
  assert.equal((await hand.inspect(1,{objectId:'target'})).singleQuery,false);
  e.form=null;e.tagName='DIV';e.type='';e.value='';e.textContent='22';
  const inspected=await hand.gateTarget(1,{objectId:'target',role:'button'},'click');
  assert.equal(inspected.dateCell,'Sunday, November 22, 2026');
  assert.equal(hand.lastTargetContext.dateCell,inspected.dateCell);
  // AX may derive a date from descendants while DOM innerText is numeric.
  cell.getAttribute=()=>'';cell.ownerDocument=e.ownerDocument;
  await hand.gateTarget(1,{objectId:'target',role:'button',name:'Sunday, November 22, 2026'},'click');
  assert.equal(hand.lastTargetContext.dateCell,'22');
  assert.equal(hand.lastTargetContext.dateCellAX,'Sunday, November 22, 2026');
});
