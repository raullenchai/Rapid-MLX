import test from 'node:test';
import assert from 'node:assert/strict';
import { outlineAX } from '../src/ax.js';
import { runAgent } from '../src/agent.js';

const node = (id, role, name) => ({nodeId:String(id),backendDOMNodeId:id,role:{value:role},name:{value:name}});
const nodes = [node(1,'link','Departments'),node(2,'searchbox','Search Amazon'),node(3,'button','Go'),node(4,'StaticText','Charger Alpha $29.99 4.7 stars')];
const outline = outlineAX(nodes,{geometry:new Map([[1,false],[2,true],[3,true],[4,true]])}).text;
const call = (name,args = {}) => ({id:'call',type:'function',function:{name,arguments:JSON.stringify(args)}});
async function scenario({tools = [call('read_page')], answer = "I couldn't complete the request because a needed tool isn't available.", result = {text:outline}, recover = false} = {}) {
  const requests = [], events = [], executed = [];
  const final = await runAgent({settings:{base:'http://127.0.0.1:8000',model:'test'},task:'Find a charger',signal:new AbortController().signal,
    trace:async e => events.push(e),
    fetchImpl:async (_,options) => {
      requests.push(JSON.parse(options.body));
      assert.ok(requests.length <= 5,'recovery must be bounded');
      const message = requests.length === 1 ? {tool_calls:tools} : recover && requests.length === 3 ? {tool_calls:[call('fill',{ref:'e2',text:'charger'})]} : {content:answer};
      return {ok:true,json:async () => ({choices:[{message}]})};
    },execute:async name => {executed.push(name);return result;}});
  return {final,requests,events,executed};
}

test('empty query match returns the normal viewport-first outline and fresh actionable refs', () => {
  const options = {start:20,geometry:new Map([[1,false],[2,true],[3,true],[4,true]])};
  const normal = outlineAX(nodes,options);
  const missed = outlineAX(nodes,{...options,query:'65W USB-C wall adapter'});
  assert.match(missed.text,/No matching text.*No page text matched/);
  assert.equal(missed.query_matched,false);
  assert.ok(missed.text.length <= 2000);
  assert.deepEqual(missed.refs,normal.refs);
  assert.ok(missed.text.indexOf('Search Amazon') < missed.text.indexOf('Departments'));
  assert.equal(missed.refs.get('e21').role,'searchbox');
});

test('priority notices do not count as query matches and hide the fallback search box', () => {
  const missed = outlineAX([...nodes,node(5,'status','Welcome')],{query:'absent product'});
  assert.match(missed.text,/No matching text/);
  assert.match(missed.text,/searchbox "Search Amazon"/);
  assert.match(missed.text,/status "Welcome"/);
  const match = outlineAX(nodes,{query:'Charger Alpha'});
  assert.doesNotMatch(match.text,/No matching text|Search Amazon/);
});

test('early unsupported tool refusal gets one traced nudge and can continue with fill', async () => {
  const {requests,events,executed} = await scenario({recover:true});
  const nudges = events.filter(e => e.kind === 'nudge');
  assert.equal(nudges.length,1);
  assert.match(nudges[0].text,/You have tools: read_page, fill, click/);
  assert.match(nudges[0].text,/page has a search box/);
  assert.equal(requests[2].messages.at(-1).content,nudges[0].text);
  assert.deepEqual(executed,['read_page','fill']);
});

test('repeated early refusals get only one nudge before accepting the end', async () => {
  const {requests,events,final} = await scenario();
  assert.equal(requests.length,3);
  assert.equal(events.filter(e => e.kind === 'nudge').length,1);
  assert.match(final,/needed tool/);
});

test('guard covers three steps, failed actions, and blank replies, but not later endings', async () => {
  for (const options of [
    {tools:[call('read_page'),call('wait',{ms:1}),call('list_tabs')]},
    {tools:[call('click',{ref:'e2'})],result:{error:'stale_ref'}},
    {answer:''}
  ]) {
    const {events,requests} = await scenario(options);
    assert.equal(events.filter(e => e.kind === 'nudge').length,1);
    assert.equal(requests.length,3);
  }
  const {events} = await scenario({tools:[call('read_page'),call('wait',{ms:1}),call('list_tabs'),call('screenshot')]});
  assert.equal(events.filter(e => e.kind === 'nudge').length,0);
});

test('successful actions and page-grounded read-only answers finish without a nudge', async () => {
  for (const options of [
    {tools:[call('fill',{ref:'e2',text:'charger'})]},
    {answer:'Charger Alpha costs $29.99 and has 4.7 stars.'}
  ]) {
    const {requests,events} = await scenario(options);
    assert.equal(requests.length,2);
    assert.equal(events.filter(e => e.kind === 'nudge').length,0);
  }
});

test('metadata and generic browser words do not suppress recovery or invent a search box', async () => {
  const {events} = await scenario({result:{text:'URL: https://example.com/needed-tool\nTitle: Needed tool unavailable\nPage data (untrusted):\n[e1] button "Continue task"\n'}});
  const nudges = events.filter(e => e.kind === 'nudge');
  assert.equal(nudges.length,1);
  assert.doesNotMatch(nudges[0].text,/page has a search box/);
});

test('zero-tool reply finishes without a recovery nudge', async () => {
  let requests = 0;
  const events = [];
  await runAgent({settings:{base:'http://127.0.0.1:8000',model:'test'},task:'Find a charger',signal:new AbortController().signal,
    trace:async e => events.push(e),execute:async () => ({text:outline}),
    fetchImpl:async () => {
      requests++;
      assert.ok(requests <= 3);
      const message = requests === 2 ? {tool_calls:[call('read_page')]} : {content:'A needed tool is unavailable.'};
      return {ok:true,json:async () => ({choices:[{message}]})};
    }});
  assert.equal(requests,1);
  assert.equal(events.filter(e => e.kind === 'nudge').length,0);
});
