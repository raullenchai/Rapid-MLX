import { markdown, stepLabel } from '../src/ui.js';
import { renderSteps, clip } from '../src/transcript.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { searchResultReply } from './e2e/fixtures.mjs';
import { waitForSearchResult } from './e2e/run.mjs';
import { connectPanel } from './e2e/helpers.mjs';

test('shared panel setup uses onboarding and custom mode before filling a readonly local URL', async () => {
  for (const firstRun of [true,false]) {
    let settingsOpen = false, custom = false, saved = false, statusReads = 0;
    const values = {};
    const panel = {
      goto:async url => assert.equal(url,'chrome-extension://test-extension/panel.html'),
      waitForFunction:async () => {},
      locator:selector => ({
        isVisible:async () => {assert.equal(selector,'#onboarding');return firstRun;},
        click:async () => {
          if (selector === '#onboard-connect') {assert.equal(firstRun,true);settingsOpen = true;}
          else {assert.equal(selector,'#settings-toggle');settingsOpen = !settingsOpen;}
        },
        selectOption:async value => {
          assert.equal(selector,'#endpoint-mode');assert.equal(settingsOpen,true);
          assert.equal(value,'custom');custom = true;
        },
        fill:async value => {
          assert.equal(settingsOpen,true);
          if (selector === '#base') assert.equal(custom,true,'Local URL is readonly until custom is selected');
          values[selector] = value;
        },
        textContent:async () => {
          assert.equal(selector,'#status');assert.equal(saved,true);
          return ++statusReads > 1 ? 'Connected' : 'Connecting…';
        }
      }),
      getByRole:(role,{name}) => ({click:async () => {
        assert.equal(role,'button');assert.equal(name,'Save & connect');saved = true;
      }})
    };
    const worker = {url:() => 'chrome-extension://test-extension/background.js'};
    const context = {serviceWorkers:() => firstRun ? [] : [worker],waitForEvent:async name => {assert.equal(name,'serviceworker');return worker;},newPage:async () => panel};
    assert.equal(await connectPanel(context,'http://127.0.0.1:43210',firstRun ? 'test-key' : '',firstRun ? {maxSteps:17} : {}),panel);
    assert.equal(settingsOpen,false);assert.equal(statusReads,2);
    assert.deepEqual(values,{'#base':'http://127.0.0.1:43210','#api-key':firstRun ? 'test-key' : '',...(firstRun ? {'#step-cap':'17'} : {})});
  }
});

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

test('settings cannot be edited until first state snapshot; later snapshots preserve edits', async () => {
  const html = readFileSync(new URL('../panel.html',import.meta.url),'utf8');
  assert.match(html,/id="settings-toggle" disabled/);
  const nodes = new Map();
  const node = () => ({dataset:{},options:[],elements:[],hidden:true,value:'',
    append(...children) { this.options.push(...children); },
    replaceChildren(...children) { this.options = children; },
    querySelectorAll:() => [],setAttribute() {},focus() {this.focused = true;}});
  const get = id => { if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id); };
  get('settings-toggle').disabled = true;
  let render;
  const sent = [], grants = [];
  let allowPermission = true;
  const document = {querySelector:get,getElementById:get,createElement:node,body:{dataset:{}}};
  const chrome = {permissions:{request:async grant => {grants.push(grant);return allowPermission;}},runtime:{connect:() => ({postMessage:m => sent.push(m),onMessage:{addListener:fn => {render = fn;}},onDisconnect:{addListener() {}}})}};
  const { endpoint, originPermission, LOCAL_ENDPOINTS, redact } = await import('../src/endpoint.js');
  runInNewContext(readFileSync(new URL('../panel.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replace(/^export .*;$/gm,''),{document,chrome,markdown,stepLabel,endpoint,originPermission,LOCAL_ENDPOINTS,redact,renderSteps,clip,t:k => k,localize:s => s,translateDocument() {},displayResult:s => s,toolLabel:s => s});
  assert.equal(get('settings-toggle').disabled,true);
  assert.equal(document.body.dataset.ready,undefined);
  const state = {settings:{base:'http://localhost:8000',apiKey:'',model:'',vision:false,stepCap:25},models:[],messages:[],tabs:[],steps:[],approvals:[],running:false,status:'Ready'};
  render(state);
  assert.equal(get('settings-toggle').disabled,false);
  assert.equal(document.body.dataset.ready,'true');
  assert.equal(get('onboarding').hidden,false);
  assert.equal(get('base').readOnly,true);
  get('onboard-connect').onclick();
  assert.equal(get('settings').hidden,false);
  assert.equal(get('endpoint-mode').focused,true);
  get('endpoint-mode').value = 'custom';
  get('endpoint-mode').onchange();
  assert.equal(get('base').readOnly,false);
  get('base').value = 'http://localhost:9000';
  render(state);
  assert.equal(get('base').value,'http://localhost:9000');
  get('endpoint-mode').value = 'custom';
  await get('settings-form').onsubmit({preventDefault() {}});
  assert.equal(sent.at(-1).settings.base,'http://localhost:9000');
  assert.equal(grants.length,0,'Bundled loopback must never request permission');
  get('base').value = 'http://127.0.0.1:43210';
  await get('settings-form').onsubmit({preventDefault() {}});
  assert.equal(sent.at(-1).settings.base,'http://127.0.0.1:43210');
  assert.equal(grants.length,0);
  get('endpoint-mode').value = 'local';
  get('endpoint-mode').onchange();
  assert.equal(get('base').readOnly,true);
  assert.equal(get('base').value,LOCAL_ENDPOINTS[0]);
  await get('settings-form').onsubmit({preventDefault() {}});
  assert.equal(sent.at(-1).settings.endpointMode,'local');
  assert.equal(grants.length,0);
  get('endpoint-mode').value = 'custom';get('endpoint-mode').onchange();
  for (const base of ['https://model.example','http://192.168.1.2:8000','https://localhost:9000','http://127.0.0.2:9000']) {
    get('base').value = base;
    await get('settings-form').onsubmit({preventDefault() {}});
    assert.equal(sent.at(-1).settings.base,base);
    assert.equal(JSON.stringify(grants.at(-1)),JSON.stringify({origins:[`${base}/*`]}));
  }
  allowPermission = false;
  const count = sent.length;
  await get('settings-form').onsubmit({preventDefault() {}});
  assert.equal(sent.length,count);
  assert.equal(get('status').textContent,'permissionMissing');
  render({...state,messages:[{role:'user',content:'Inspect this page'}]});
  assert.equal(get('onboarding').hidden,true,'First task completes inline onboarding');
});
