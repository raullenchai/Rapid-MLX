import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { searchResultReply } from './e2e/fixtures.mjs';
import { waitForSearchResult } from './e2e/run.mjs';

test('fake engine waits for committed search results, including interrupted reads', () => {
  const tool = (name,args) => ({name,args});
  const pending = 'URL: http://fixture/\nSearch query';
  const result = text => [{content:JSON.stringify({text})}];
  assert.equal(searchResultReply(result(pending),pending,tool).name,'wait');
  assert.equal(searchResultReply([{content:'{"error":"stale_ref"}'}],pending,tool).name,'wait');
  assert.equal(searchResultReply(result('Waited 50ms'),pending,tool).name,'read_page');
  const committed = 'URL: http://fixture/searched?q=local+browser+test\nSearch complete: local browser test';
  assert.equal(searchResultReply(result(committed),committed,tool).content,'Search fixture complete');
  assert.equal(searchResultReply(result(committed),committed.replace('local+browser+test','wrong'),tool).name,'wait');
});

test('harness waits for URL commit and result DOM after server request receipt', async () => {
  let commit, resultDOM, done = false;
  const navigation = new Promise(resolve => {commit = resolve;});
  const rendered = new Promise(resolve => {resultDOM = resolve;});
  const page = {
    waitForURL:async predicate => {
      assert.equal(predicate(new URL('http://fixture/')),false);
      assert.equal(predicate(new URL('http://fixture/searched?q=local+browser+test')),true);
      await navigation;
    },
    locator:selector => { assert.equal(selector,'#result'); return {filter:({hasText}) => {
      assert.equal(hasText,'Search complete: local browser test'); return {waitFor:() => rendered};
    }}; }
  };
  const waiting = waitForSearchResult(page).then(() => {done = true;});
  await Promise.resolve(); assert.equal(done,false); // server saw request; URL still old
  commit(); await Promise.resolve(); assert.equal(done,false); // DOM still loading
  resultDOM(); await waiting; assert.equal(done,true);
});

test('settings cannot be edited until first state snapshot; later snapshots preserve edits', () => {
  const html = readFileSync(new URL('../panel.html',import.meta.url),'utf8');
  assert.match(html,/id="settings-toggle" disabled/);
  const nodes = new Map();
  const node = () => ({dataset:{},options:[],elements:[],hidden:true,value:'',
    append(...children) { this.options.push(...children); },
    replaceChildren(...children) { this.options = children; },
    querySelectorAll:() => [],setAttribute() {}});
  const get = id => { if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id); };
  get('settings-toggle').disabled = true;
  let render;
  const sent = [];
  const document = {getElementById:get,createElement:node,body:{dataset:{}}};
  const chrome = {runtime:{connect:() => ({postMessage:m => sent.push(m),onMessage:{addListener:fn => {render = fn;}},onDisconnect:{addListener() {}}})}};
  runInNewContext(readFileSync(new URL('../panel.js',import.meta.url),'utf8'),{document,chrome});
  assert.equal(get('settings-toggle').disabled,true);
  assert.equal(document.body.dataset.ready,undefined);
  const state = {settings:{base:'http://localhost:8000',apiKey:'',model:'',vision:false,stepCap:25},models:[],messages:[],tabs:[],steps:[],approvals:[],running:false,status:'Ready'};
  render(state);
  assert.equal(get('settings-toggle').disabled,false);
  assert.equal(document.body.dataset.ready,'true');
  get('settings-toggle').onclick();
  get('base').value = 'http://localhost:9000';
  render(state);
  assert.equal(get('base').value,'http://localhost:9000');
  get('settings-form').onsubmit({preventDefault() {}});
  assert.equal(sent.at(-1).settings.base,'http://localhost:9000');
});
