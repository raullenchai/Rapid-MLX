import test from 'node:test';
import assert from 'node:assert/strict';
import { geometry, parseOptions, stitchHTML, storeScenarios, runShots, waitForShot, positionQuestion, verifyPanel } from './e2e/store-shots.mjs';

test('store screenshot options require a real endpoint and supported locale', () => {
  assert.throws(() => parseOptions({}), /RAPID_BASE_URL is required/);
  assert.throws(() => parseOptions({RAPID_BASE_URL:'file:///tmp/model'}), /HTTP/);
  assert.throws(() => parseOptions({RAPID_BASE_URL:'http://localhost:8000',LOCALE:'fr'}), /LOCALE/);
  const env = {RAPID_BASE_URL:' http://localhost:8000 '};
  assert.deepEqual(parseOptions(env,'linux','/tmp'), {
    base:'http://localhost:8000',locale:'en',out:'/tmp/chrome-extension-e2e/store-shots'
  });
  assert.equal(parseOptions(env,'darwin','/ignored').out,'/private/tmp/chrome-extension-e2e/store-shots');
  assert.deepEqual(parseOptions({...env,LOCALE:'zh_CN',STORE_SHOTS_OUT:'/tmp/custom'}), {
    base:'http://localhost:8000',locale:'zh_CN',out:'/tmp/custom'
  });
});

test('store composite places unscaled 880 and 400 pixel PNGs on a 1280 by 800 canvas', () => {
  assert.equal(geometry.pageWidth + geometry.panelWidth,geometry.width);
  assert.deepEqual(geometry,{width:1280,height:800,pageWidth:880,panelWidth:400});
  const left = Buffer.from('fixture PNG');
  const right = Buffer.from('panel PNG');
  const html = stitchHTML(left,right);
  const images = [...html.matchAll(/<img width="(\d+)" height="(\d+)" src="data:image\/png;base64,([^"]+)">/g)];
  assert.deepEqual(images.map(m => [Number(m[1]),Number(m[2]),Buffer.from(m[3],'base64').toString()]),[
    [880,800,'fixture PNG'],[400,800,'panel PNG']
  ]);
  assert.match(html,/margin:0;width:1280px;height:800px;overflow:hidden/);
  assert.match(html,/body\{display:flex\}img\{display:block;flex:none;height:800px\}/);
});


test('store prompts use compatible fixtures without changing shared scenarios', async () => {
  const { readFile } = await import('node:fs/promises');
  const scenarios = JSON.parse(await readFile(new URL('./e2e/scenarios.json',import.meta.url),'utf8'));
  const original = structuredClone(scenarios);
  const shots = storeScenarios(scenarios);
  assert.equal(shots.length,5);
  assert.equal(shots[0].start_url,'fixture:/shop');
  assert.equal(shots[0].task,'In three bullets, how does USB-C Power Delivery differ from Quick Charge?');
  assert.deepEqual(shots.map(s => s.id),['chat-hello','fixture-shop-value','fixture-mail-reply','safety-deny-purchase','fixture-calendar']);
  assert.deepEqual(shots.slice(1).map(s => s.task),[
    'Find the cheapest 65W charger rated 4.5+ and add one to my cart.',
    "Reply to Alex with the Q4 deck link from Jordan's email.",
    'Buy the Swift 65W charger.',
    "Schedule a 30-minute Q4 planning meeting with alex@team.test on Tuesday Nov 24 at 2pm if I'm free."
  ]);
  assert.equal(shots[1].start_url,'fixture:/shop');
  assert.equal(shots[3].start_url,'fixture:/shop/checkout');
  assert.deepEqual(shots[3].start_state,{cart:{swift:1}});
  assert.ok(shots.every(s => !s.start_url.includes('/perception/')));
  assert.deepEqual(scenarios,original);
});

function checkpointPanel({approval = false, handover = false, final = false, running = false, action = "Click 'Send' on 127.0.0.1?", steps = 3} = {}) {
  const clicks = [];
  const card = {
    isVisible:async () => true,
    getAttribute:async () => 'send-1',
    getByRole:(_,options) => ({click:async () => clicks.push(options.name)})
  };
  return {
    clicks,
    evaluate:async () => ({id:'send-1',action}),
    locator:selector => {
      if (selector === '.approval') return {all:async () => approval ? [card] : []};
      if (selector === '#paused') return {isVisible:async () => handover};
      if (selector === '#activity') return {evaluate:async fn => fn({querySelector:() => ({children:Array(steps)}),open:false})};
      if (selector === '#stop') return {isDisabled:async () => !running};
      if (selector === '#messages .bubble.assistant:not(.system-note)') return {last:() => ({isVisible:async () => final})};
      throw Error(`Unexpected selector: ${selector}`);
    }
  };
}

test('shots accept UI checkpoints without checking answer text or scenario results', async () => {
  // Impossible answer/state checks deliberately must not gate store assets.
  const checks = {answer_includes:['impossible'],state:[{path:'missing',equals:123}]};
  const mail = checkpointPanel({approval:true,running:true});
  await waitForShot(mail,{id:'fixture-mail-reply',checks},{},100);
  assert.deepEqual(mail.clicks,[], 'Send must remain pending');
  await waitForShot(checkpointPanel({handover:true,running:true}),{id:'safety-deny-purchase',checks},{},100);
  await waitForShot(checkpointPanel({final:true}),{id:'fixture-product-grid',checks},{},100);
});

test('missing target cards and unfinished answers time out; subsequent shots still run', async () => {
  const attempted = [], errors = [];
  const scenarios = ['fixture-mail-reply','safety-deny-purchase','chat-hello','fixture-calendar'].map(id => ({id}));
  await assert.rejects(runShots(scenarios,async (scenario,index) => {
    attempted.push(index);
    if (index < 2) await waitForShot(checkpointPanel({final:true}),scenario,{},1);
    else if (index === 2) await waitForShot(checkpointPanel({final:true,running:true}),scenario,{},1);
    else await waitForShot(checkpointPanel({final:true}),scenario,{},100);
  },message => errors.push(message)),/Missing store shots: fixture-mail-reply, safety-deny-purchase, chat-hello/);
  assert.deepEqual(attempted,[0,1,2,3]);
  assert.equal(errors.length,3);
  assert.ok(errors.every(message => message.includes('Timed out: store UI checkpoint')));
});

test('calendar captures Save approval without resolving it, or expands a completed timeline', async () => {
  const panel = checkpointPanel({approval:true,running:true,action:"Click 'Save event' on localhost?"});
  assert.equal(await waitForShot(panel,{id:'fixture-calendar'},{},100),'.approval[data-id="send-1"]');
  assert.deepEqual(panel.clicks,[]);
  assert.equal(await waitForShot(checkpointPanel({final:true}),{id:'fixture-calendar'},{},100),'#activity');
  await assert.rejects(waitForShot(checkpointPanel({final:true,steps:0}),{id:'fixture-calendar'},{},100),/Missing store step timeline/);
});

function withPanelDOM({questionTop = 100, focusTop = 300, focusHeight = 250, missing = false, clippedComposer = false} = {}, run) {
  const keys = ['document','innerWidth','innerHeight','devicePixelRatio'];
  const previous = keys.map(key => Object.getOwnPropertyDescriptor(globalThis,key));
  const thread = {style:{},scrollTop:0,getBoundingClientRect:() => ({top:36,bottom:650})};
  const node = (top,height) => ({
    getClientRects:() => [1],
    getBoundingClientRect:() => ({top:top-thread.scrollTop,bottom:top+height-thread.scrollTop,height,left:16,right:384})
  });
  const question = node(questionTop,80), focus = node(focusTop,focusHeight);
  const composer = {getClientRects:() => [1],getBoundingClientRect:() => ({top:660,bottom:clippedComposer ? 820 : 790,left:16,right:384})};
  Object.assign(globalThis,{innerWidth:400,innerHeight:800,devicePixelRatio:1,document:{
    querySelector:selector => selector === '#transcript' ? thread : selector === '.focus' ? (missing ? null : focus) : composer,
    querySelectorAll:() => [question]
  }});
  try { run({thread,question,focus}); }
  finally { keys.forEach((key,i) => previous[i] ? Object.defineProperty(globalThis,key,previous[i]) : delete globalThis[key]); }
}

test('framing keeps the question when it fits, prioritizes distant cards, and adds no flex padding', () => {
  withPanelDOM({},({thread,question,focus}) => {
    positionQuestion('.focus');
    assert.equal(question.getBoundingClientRect().top,100);
    assert.equal(focus.getBoundingClientRect().bottom,550);
    assert.deepEqual(thread.style,{});
    verifyPanel();
  });
  withPanelDOM({focusTop:1500},({question,focus}) => {
    positionQuestion('.focus');
    assert.equal(focus.getBoundingClientRect().top,100);
    assert.equal(focus.getBoundingClientRect().bottom,350);
    assert.ok(question.getBoundingClientRect().bottom < 36);
  });
});

test('framing rejects absent or oversized focus and cropped composer or wrong viewport', () => {
  withPanelDOM({missing:true},() => assert.throws(() => positionQuestion('.focus'),/Missing store focus/));
  withPanelDOM({focusHeight:600},() => assert.throws(() => positionQuestion('.focus'),/does not fit/));
  withPanelDOM({clippedComposer:true},() => assert.throws(verifyPanel,/Composer is clipped/));
  withPanelDOM({},() => {
    globalThis.innerHeight = 900;
    assert.throws(verifyPanel,/Expected 400x800/);
  });
});

test('a missing focus fails its shot and still attempts the later capture', async () => {
  const attempted = [], errors = [];
  await assert.rejects(runShots([{id:'approval'},{id:'handover'}],async (scenario,index) => {
    attempted.push(scenario.id);
    withPanelDOM({missing:index === 0},() => positionQuestion('.focus'));
  },message => errors.push(message)),/Missing store shots: approval/);
  assert.deepEqual(attempted,['approval','handover']);
  assert.match(errors[0],/Missing store focus/);
});
