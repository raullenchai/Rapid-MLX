import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { endpoint, originPermission, privateHost, LOCAL_ENDPOINTS, reasoningOptions, redact, testConnection } from '../src/endpoint.js';
import { outlineAX } from '../src/ax.js';
import { runAgent, SYSTEM_PROMPT } from '../src/agent.js';
import { patterns } from '../src/i18n-patterns.js';
import { t, localize } from '../src/i18n.js';
const read = p => readFileSync(new URL(`../${p}`,import.meta.url),'utf8');
const settings = {base:'https://model.example',model:'m',apiKey:'secret-key',reasoningEffort:'high'};
const permissions = {contains:async () => true};
const reply = (body,status = 200) => ({ok:status >= 200 && status < 300,status,json:async () => body});

test('endpoint origins allow HTTPS and only specified private HTTP ranges', () => {
  for (const url of ['https://model.example','https://model.example:8443','http://localhost:8000','http://127.2.3.4:8000','http://[::1]:8000','http://10.2.3.4','http://172.16.0.1','http://172.31.255.255','http://192.168.1.2','http://100.64.0.0','http://100.127.255.255']) assert.equal(endpoint(`${url}/`),url);
  for (const url of ['http://example.org','http://172.15.1.1','http://172.32.0.1','http://100.63.255.255','http://100.128.0.1','http://192.169.1.1','http://localhost.example','file:///tmp','https://user:pass@example.org','https://example.org/v1','https://example.org?key=x','https://example.org/#x']) assert.throws(() => endpoint(url));
  assert.equal(privateHost('100.64.0.999'),false);
  assert.equal(originPermission('https://model.example:8443/'),'https://model.example:8443/*');
  assert.deepEqual(LOCAL_ENDPOINTS,['http://127.0.0.1:8000','http://127.0.0.1:7659']);
  const manifest = JSON.parse(read('manifest.json'));
  assert.deepEqual(manifest.optional_host_permissions,['https://*/*','http://*/*']);
  assert.deepEqual(manifest.host_permissions,['http://127.0.0.1/*','http://localhost/*']);
});

test('connection checks permission before fetch and probes a single non-browser tool', async () => {
  const requests = [], grants = [];
  const result = await testConnection(settings,{permissions:{contains:async p => {grants.push(p);return true;}},fetchImpl:async (url,options) => {
    requests.push({url,...options});
    return reply(url.endsWith('/models') ? {data:[{id:'m'}]} : {choices:[{message:{tool_calls:[{function:{name:'rapid_connection_test',arguments:'{"ok":true}'}}]}}]});
  }});
  assert.equal(result.model,'m');
  assert.deepEqual(grants,[{origins:['https://model.example/*']}]);
  assert.equal(requests.length,2);
  const body = JSON.parse(requests[1].body);
  assert.equal(body.tools.length,1); assert.equal(body.tools[0].function.name,'rapid_connection_test');
  assert.equal(body.reasoning_effort,'high'); assert.equal(body.chat_template_kwargs.enable_thinking,true);
  assert.equal(requests[1].headers.Authorization,'Bearer secret-key');
  await assert.rejects(testConnection(settings,{permissions:{contains:async () => false},fetchImpl:async () => {throw new Error('must not fetch');}}),/permissionMissing/);
});

test('connection failures identify auth, access, invalid payloads, empty models and absent tool support', async () => {
  for (const [status,key] of [[401,'unauthorized'],[403,'forbidden'],[500,'endpointHTTP']]) await assert.rejects(testConnection(settings,{permissions,fetchImpl:async () => reply({},status)}),new RegExp(key));
  await assert.rejects(testConnection(settings,{permissions,fetchImpl:async () => {throw new TypeError('network');}}),/unreachable/);
  await assert.rejects(testConnection(settings,{permissions,fetchImpl:async () => reply({data:[]})}),/noModels/);
  await assert.rejects(testConnection(settings,{permissions,fetchImpl:async () => ({ok:true,json:async () => {throw new Error();}})}),/invalidResponse/);
  const lists = [];
  await assert.rejects(testConnection(settings,{permissions,onModels:info => lists.push(info),fetchImpl:async url => reply(url.endsWith('/models') ? {data:[{id:'no-tools'},{id:'with-tools'}]} : {choices:[{message:{content:'no tool'}}]})}),/noTools/);
  assert.deepEqual(lists[0].models,['no-tools','with-tools'],'Dropdown remains usable when the first model fails the tool probe');
  for (const calls of [[],[{function:{name:'wrong',arguments:'{"ok":true}'}}],[{function:{name:'rapid_connection_test',arguments:'not JSON'}}]]) {
    await assert.rejects(testConnection(settings,{permissions,fetchImpl:async url => reply(url.endsWith('/models') ? {data:[{id:'m'}]} : {choices:[{message:{tool_calls:calls}}]})}),/noTools/);
  }
});

test('connection accepts tool calls alongside content without a tool_choice acknowledgement', async () => {
  const result = await testConnection(settings,{permissions,fetchImpl:async url => reply(url.endsWith('/models') ? {data:[{id:'m'}]} : {choices:[{finish_reason:'stop',message:{content:'Here is the test call.',tool_calls:[{function:{name:'rapid_connection_test',arguments:'{"ok":true}'}}]}}]})});
  assert.equal(result.model,'m');
  assert.equal(result.warning,undefined);
});

test('probe timeouts warn after model discovery; discovery failures and cancellation still reject', async () => {
  const originalTimeout = AbortSignal.timeout;
  try {
    // Expire only the probe signal, including while reading the response body.
    for (const phase of ['fetch','json']) {
      let n = 0;
      AbortSignal.timeout = ms => {
        assert.equal(ms,15000);
        return ++n === 2 ? AbortSignal.abort(new DOMException('Timed out','TimeoutError')) : new AbortController().signal;
      };
      const result = await testConnection(settings,{permissions,fetchImpl:async (url,{signal}) => {
        if (url.endsWith('/models')) return reply({data:[{id:'m'}]});
        if (phase === 'fetch') signal.throwIfAborted();
        return {ok:true,status:200,json:async () => {signal.throwIfAborted();}};
      }});
      assert.equal(result.model,'m');
      assert.equal(result.warning,'toolProbeTimeout');
    }
    AbortSignal.timeout = () => AbortSignal.abort(new DOMException('Timed out','TimeoutError'));
    await assert.rejects(testConnection(settings,{permissions,fetchImpl:async (_url,{signal}) => {signal.throwIfAborted();}}),/unreachable/);
    const controller = new AbortController();
    await assert.rejects(testConnection(settings,{permissions,signal:controller.signal,fetchImpl:async url => {
      if (url.endsWith('/models')) return reply({data:[{id:'m'}]});
      controller.abort(); controller.signal.throwIfAborted();
    }}),/unreachable/);
  } finally {AbortSignal.timeout = originalTimeout;}
});

test('reasoning options omit off effort; agent requests and traces protect credentials', async () => {
  assert.equal(reasoningOptions('off').reasoning_effort,undefined);
  for (const effort of ['low','high']) assert.deepEqual(reasoningOptions(effort),{reasoning_effort:effort,chat_template_kwargs:{enable_thinking:true}});
  const traces = [], requests = [];
  await runAgent({settings:{...settings,stepCap:1},task:'inspect',signal:new AbortController().signal,trace:async e => traces.push(e),execute:async () => ({text:'done'}),fetchImpl:async (_url,o) => {
    requests.push(JSON.parse(o.body));
    return reply({choices:[{message:{content:'secret-key',tool_calls:[{id:'a',function:{name:'read_page',arguments:'{}'}}]}}]});
  }});
  assert.equal(requests[0].reasoning_effort,'high'); assert.equal(requests[0].chat_template_kwargs.enable_thinking,true);
  assert.doesNotMatch(JSON.stringify(traces),/secret-key/);
  const value = redact({headers:{Authorization:'Bearer abc','x-api-key':'abc'},nested:['Authorization: Bearer abc','secret-key'],error:'Bearer another-key'},'secret-key');
  assert.doesNotMatch(JSON.stringify(value),/abc|secret-key|another-key/);
  assert.match(JSON.stringify(value),/REDACTED/);
});

test('commerce AX fixture separates main, struck-through, alternate offers and unit prices', () => {
  const nodes = JSON.parse(read('test/fixtures/price-roles-ax.json'));
  const output = outlineAX(nodes).text;
  assert.match(output,/price: \$29\.99/);
  assert.match(output,/was: \$39\.99/);
  assert.match(output,/other offers from: \$22\.49/);
  assert.match(output,/unit price: \$1\.25/);
  assert.doesNotMatch(output,/(?:^|\| )price: \$22\.49/);
  for (const heading of ['Used & new','Other sellers','More Buying Choices']) {
    const clone = structuredClone(nodes); clone.find(n => n.nodeId === 'offers-heading').name.value = heading;
    assert.match(outlineAX(clone).text,/other offers from: \$22\.49/);
  }
  const extra = structuredClone(nodes);
  extra[0].childIds.push('second-offer');
  extra.push({nodeId:'second-offer',role:{value:'StaticText'},name:{value:'$24.99'}});
  assert.match(outlineAX(extra).text,/other offers from: \$24\.99/);
  const onlyOffers = nodes.filter(n => !['price','old','unit'].includes(n.nodeId));
  assert.doesNotMatch(outlineAX(onlyOffers).text,/(?:^|\| )price:/);
  assert.match(SYSTEM_PROMPT,/main buy-box\/list price/); assert.match(SYSTEM_PROMPT,/cart\/subtotal/);
});

test('price roles survive combined labels, nested alternate offers and hidden prices', () => {
  const nodes = JSON.parse(read('test/fixtures/price-roles-ax.json'));
  const card = nodes[0];
  card.childIds = ['link','combined'];
  nodes.push({nodeId:'combined',role:{value:'StaticText'},name:{value:'$29.99 More Buying Choices $22.49'}});
  const combined = outlineAX(nodes.slice(0,2).concat(nodes.at(-1))).text;
  assert.match(combined,/price: \$29\.99/);assert.match(combined,/other offers from: \$22\.49/);
  const nested = [card,nodes[1],{nodeId:'group',role:{value:'generic'},name:{value:'Used & new'},childIds:['value']},{nodeId:'value',role:{value:'StaticText'},name:{value:'$22.49'}}];
  card.childIds = ['link','group'];
  assert.match(outlineAX(nested).text,/other offers from: \$22\.49/);
  assert.doesNotMatch(outlineAX(nested).text,/(?:^|\| )price:/);
  nested[2].properties = [{name:'hidden',value:{value:true}}];
  assert.doesNotMatch(outlineAX(nested).text,/22\.49/);
});

test('every manifest marker, UI attribute, catalog template and literal i18n key exists in both locales', () => {
  const en = JSON.parse(read('_locales/en/messages.json')), zh = JSON.parse(read('_locales/zh_CN/messages.json'));
  assert.deepEqual(Object.keys(en).sort(),Object.keys(zh).sort());
  assert.equal(new Set(Object.keys(en).map(k => k.toLowerCase())).size,Object.keys(en).length,'Browser message keys are case insensitive');
  const files = ['manifest.json','panel.html','panel.js','background.js',...readdirSync(new URL('../src/',import.meta.url)).filter(p => p.endsWith('.js')).map(p => `src/${p}`)];
  const keys = new Set(patterns.map(([,k]) => k));
  for (const f of files) for (const m of read(f).matchAll(/__MSG_([\w]+)__|data-i18n(?:-[\w-]+)?="([\w]+)"|\bt\(['"]([\w]+)['"]/g)) keys.add(m[1] || m[2] || m[3]);
  for (const kind of ['shopping','travel','reservations','mail']) {keys.add(kind);keys.add(`${kind}Example`);}
  for (const key of keys) {
    assert.ok(en[key]?.message,key); assert.ok(zh[key]?.message,key);
    assert.deepEqual(en[key].message.match(/\$\d/g)?.sort(),zh[key].message.match(/\$\d/g)?.sort(),key);
  }
});

test('browser locale renders status and approval templates while preserving page names', () => {
  const old = globalThis.chrome;
  const zh = JSON.parse(read('_locales/zh_CN/messages.json'));
  globalThis.chrome = {i18n:{getMessage:(key,args = []) => zh[key]?.message.replace(/\$(\d)/g,(_,n) => args[n-1]) || ''}};
  try {
    assert.equal(t('settings'),'设置');
    assert.equal(localize('Ready'),'就绪');
    assert.equal(localize("Click 'Buy item' on shop.example?"),'在 shop.example 点击“Buy item”？');
    assert.equal(localize('Shared tab: My page'),'已共享标签页：My page');
  } finally {globalThis.chrome = old;}
});

test('build produces a deterministic installable archive and rejects missing referenced files', async () => {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const {tmpdir} = await import('node:os');
  const {spawnSync} = await import('node:child_process');
  const {inflateRawSync} = await import('node:zlib');
  const scratch = fs.mkdtempSync(path.join(process.platform === 'darwin' ? '/private/tmp' : tmpdir(),'chrome-extension-production-build-'));
  try {
    const source = new URL('../',import.meta.url);
    for (const entry of ['scripts','src','icons','_locales','manifest.json','package.json','background.js','panel.html','panel.css','panel.js']) fs.cpSync(new URL(entry,source),path.join(scratch,entry),{recursive:true});
    fs.mkdirSync(path.join(scratch,'test'));fs.writeFileSync(path.join(scratch,'test','secret.txt'),'must not ship');
    const build = () => spawnSync(process.execPath,['scripts/build.mjs'],{cwd:scratch,encoding:'utf8'});
    const first = build();assert.equal(first.status,0,first.stderr);
    const output = path.join(scratch,'dist/rapid-mlx-chrome-0.1.0.zip');
    const bytes = fs.readFileSync(output);assert.equal(build().status,0);assert.deepEqual(fs.readFileSync(output),bytes);
    const entries = [];let offset = 0;
    while (bytes.readUInt32LE(offset) === 0x04034b50) {
      const size = bytes.readUInt32LE(offset+18), rawSize = bytes.readUInt32LE(offset+22), nameLength = bytes.readUInt16LE(offset+26), extra = bytes.readUInt16LE(offset+28);
      const name = bytes.subarray(offset+30,offset+30+nameLength).toString();entries.push(name);
      const start = offset+30+nameLength+extra;
      assert.equal(inflateRawSync(bytes.subarray(start,start+size)).length,rawSize);
      offset = start+size;
    }
    assert.deepEqual(entries,[...entries].sort()); assert.ok(entries.includes('manifest.json'));assert.ok(entries.includes('_locales/zh_CN/messages.json'));
    assert.equal(entries.some(e => /test|store|scripts|dist/.test(e)),false);
    const manifestPath = path.join(scratch,'manifest.json');
    const manifestText = fs.readFileSync(manifestPath,'utf8');
    const wider = JSON.parse(manifestText);wider.host_permissions.push('http://*/*');
    fs.writeFileSync(manifestPath,JSON.stringify(wider));
    assert.match(build().stderr,/Required hosts must be limited to HTTP loopback/);
    fs.writeFileSync(manifestPath,manifestText);
    fs.appendFileSync(path.join(scratch,'panel.js'),"\nimport './missing.js';\n");
    const broken = build();assert.notEqual(broken.status,0);assert.match(broken.stderr,/Missing referenced file: missing.js/);
    fs.rmSync(path.join(scratch,'icons/16.png'));
    assert.match(build().stderr,/Missing referenced file: icons\/16.png/);
  } finally {fs.rmSync(scratch,{recursive:true,force:true});}
});

test('worker restores settings and discovers the Desktop port without widening permissions', async () => {
  const oldChrome = globalThis.chrome, oldFetch = globalThis.fetch;
  const en = JSON.parse(read('_locales/en/messages.json'));
  const event = () => ({listeners:[],addListener(fn) {this.listeners.push(fn);}});
  const snapshots = [], origins = [], urls = [], saved = [];
  const runtime = {id:'unit-extension',onConnect:event(),onMessage:event(),getURL:p => `chrome-extension://unit-extension/${p}`};
  globalThis.chrome = {
    runtime,permissions:{contains:async p => {origins.push(p);return true;}},
    i18n:{getMessage:(k,args=[]) => en[k]?.message.replace(/\$(\d)/g,(_,n) => args[n-1]) || ''},
    storage:{local:{get:async () => ({settings:{base:LOCAL_ENDPOINTS[0],apiKey:'local-key',endpointMode:'local'}}),set:async o => saved.push(structuredClone(o))},session:{get:async () => ({}),set:async () => {}}},
    tabs:{onRemoved:event(),onUpdated:event()},debugger:{onEvent:event(),onDetach:event()},action:{onClicked:event()},commands:{onCommand:event()}
  };
  globalThis.fetch = async (url,options) => {
    urls.push(url);
    if (url.startsWith(LOCAL_ENDPOINTS[0])) throw new TypeError('offline');
    assert.equal(options.headers.Authorization,'Bearer local-key');
    return reply(url.endsWith('/models') ? {data:[{id:'desktop-model'}]} : {choices:[{message:{tool_calls:[{function:{name:'rapid_connection_test',arguments:'{"ok":true}'}}]}}]});
  };
  try {
    await import(`../background.js?production-test`);
    const port = {name:'rapid-panel',sender:{url:runtime.getURL('panel.html')},onDisconnect:event(),onMessage:event(),postMessage:s => snapshots.push(structuredClone(s))};
    runtime.onConnect.listeners[0](port);
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(snapshots.at(-1).connected,true);
    assert.equal(snapshots.at(-1).settings.base,LOCAL_ENDPOINTS[1]);
    assert.equal(snapshots.at(-1).settings.model,'desktop-model');
    assert.deepEqual(urls,[`${LOCAL_ENDPOINTS[0]}/v1/models`,`${LOCAL_ENDPOINTS[1]}/v1/models`,`${LOCAL_ENDPOINTS[1]}/v1/chat/completions`]);
    assert.equal(origins.every(p => JSON.stringify(p) === JSON.stringify({origins:['http://127.0.0.1:8000/*']}) || JSON.stringify(p) === JSON.stringify({origins:['http://127.0.0.1:7659/*']})),true);
    assert.equal(saved.findLast(value => value.settings).settings.base,LOCAL_ENDPOINTS[1]);
    globalThis.chrome.permissions.contains = async () => false;
    port.onMessage.listeners[0]({type:'connect'});
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(snapshots.at(-1).connected,false);
    assert.equal(snapshots.at(-1).status,en.permissionMissing.message);
    assert.equal(urls.length,3,'Revoked permission must prevent fetch');
  } finally {globalThis.chrome = oldChrome;globalThis.fetch = oldFetch;}
});
