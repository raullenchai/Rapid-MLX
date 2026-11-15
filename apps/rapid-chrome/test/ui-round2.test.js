import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ScrollFollow, activityLabels, errorType, acceptsSnapshot, runSnapshot, restoreRun, quietSteps } from '../src/ui-state.js';

const messages = [{id:'user-1',role:'user',content:'Hi'},{id:'answer-1',role:'assistant',content:'A'}];
test('follow pauses for scroll-up and selection; unread counts message identity, never chunks', () => {
  const follow = new ScrollFollow();
  assert.equal(follow.update(messages),true);
  follow.scroll({scrollHeight:1000,scrollTop:100,clientHeight:400});
  assert.equal(follow.update([...messages.slice(0,1),{...messages[1],content:'A longer chunk'}]),false);
  assert.equal(follow.unread,0);
  follow.update([...messages,{id:'answer-2',content:'New'}]);
  follow.update([...messages,{id:'answer-2',content:'New tokens'}]);
  assert.equal(follow.unread,1);
  follow.bottom(); assert.equal(follow.unread,0);
  assert.equal(follow.update(messages,true),false);
  follow.scroll({scrollHeight:1000,scrollTop:565,clientHeight:400});
  assert.equal(follow.follow,true);
});
test('plural activity labels in both locales include zero, one, many and worked time', () => {
  for (const locale of ['en','zh_CN']) {
    const catalog = JSON.parse(readFileSync(new URL(`../_locales/${locale}/messages.json`,import.meta.url)));
    const t = (key,args=[]) => catalog[key].message.replace(/\$(\d)/g,(_,n)=>args[n-1]);
    const steps = Array.from({length:3},(_,i)=>({id:String(i),url:'https://example.org'}));
    assert.equal(activityLabels(steps,t,true),locale === 'en' ? 'Browsed 1 site · 3 steps' : '已浏览 1 个网站 · 3 步');
    assert.equal(activityLabels(steps.slice(0,1),t,true),locale === 'en' ? 'Browsed 1 site · 1 step' : '已浏览 1 个网站 · 1 步');
    assert.equal(activityLabels([],t,true),locale === 'en' ? 'Browsed 0 sites · 0 steps' : '已浏览 0 个网站 · 0 步');
    assert.equal(activityLabels(steps,t,false,42),locale === 'en' ? 'Worked for 42s · 3 steps' : '用时 42 秒 · 3 步');
  }
});
test('errors have actionable categories without guessing CORS from failed fetch or HTTP status', () => {
  for (const value of ['unreachable','Rapid engine not reachable at localhost','Failed to fetch']) assert.equal(errorType(value),'unreachable');
  for (const value of ['cors','origin_refused','CORS blocked']) assert.equal(errorType(value),'cors');
  assert.equal(errorType('Rapid engine HTTP 403. Check API key and --cors-origins.'),'unknown');
  assert.equal(errorType('noTools'),'noTools');
  assert.equal(errorType('Task completion is unconfirmed: step cap (3)'),'stepCap');
});
test('run restore retains partial conversation and steps but never revives approvals or browser execution', () => {
  const saved = runSnapshot({sequence:14,run:{id:'run',startedAt:10},messages,steps:[{id:'step',status:'running'}],approvals:[{id:'old-allow'}],running:true,paused:true,status:'Working'});
  const restored = restoreRun(JSON.parse(JSON.stringify(saved)));
  assert.deepEqual(restored.messages,messages); assert.equal(restored.steps[0].id,'step');
  assert.equal(restored.running,false); assert.equal(restored.paused,false);
  assert.deepEqual(restored.approvals,[]); assert.equal(restored.status,'Stopped');
  assert.equal(restored.run.interrupted,true); assert.ok(restored.run.endedAt);
  assert.equal(saved.approvals,undefined);
  assert.deepEqual(restoreRun({version:99}),{});
  const finished = restoreRun({...saved,running:false,status:'Ready'});
  assert.equal(finished.status,'Ready');
});
test('sequenced snapshots reject duplicates and older updates but accept a restarted worker', () => {
  assert.equal(acceptsSnapshot({epoch:'one',sequence:3},{epoch:'one',sequence:3}),false);
  assert.equal(acceptsSnapshot({epoch:'one',sequence:3},{epoch:'one',sequence:2}),false);
  assert.equal(acceptsSnapshot({epoch:'one',sequence:3},{epoch:'one',sequence:4}),true);
  assert.equal(acceptsSnapshot({epoch:'one',sequence:3},{epoch:'two',sequence:1}),true);
});
test('quiet timeline dedupes identical adjacent reads but retains actions, changed queries and failures', () => {
  const read = {name:'read_page',url:'https://example.org',args:{},status:'ok'};
  const steps = [read,{...read},{...read,args:{query:'price'}},{...read,status:'error'},{name:'click'},read];
  assert.equal(quietSteps(steps).length,5);
});
test('panel controls have unique IDs, bundled SVGs, and no inline event handlers', () => {
  const html = readFileSync(new URL('../panel.html',import.meta.url),'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match=>match[1]);
  assert.equal(new Set(ids).size,ids.length);
  for (const id of ['voice','send','stop','add-context','settings-toggle','history-toggle','new-chat']) {
    assert.match(html,new RegExp(`<button id="${id}"[^>]*><svg[^>]*stroke="currentColor"`));
  }
  assert.equal(/\bon\w+=/i.test(html),false);
});
