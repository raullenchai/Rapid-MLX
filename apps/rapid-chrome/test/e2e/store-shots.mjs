// Real-engine store captures. Importing this module never starts a browser.
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { launchExtension, waitForConnection, submitTask, until } from './helpers.mjs';
import { scenarioFixtures } from './fixtures.mjs';
import { scenarioApproval } from './scenarios.mjs';

export const geometry = Object.freeze({width:1280,height:800,pageWidth:880,panelWidth:400});

// Presentation only: never change the shared fixture HTML or scenario behavior.
export const fixtureStyle = `
html { background:#f4f6f8; color-scheme:light; }
body { box-sizing:border-box; font:15px/1.5 system-ui,sans-serif; color:#263244;
  max-width:840px; margin:0 auto; padding:24px 28px; background:#f4f6f8; }
nav { display:flex; flex-wrap:wrap; gap:8px 16px; margin:0 0 24px; }
a { color:#475569; text-decoration:none; margin:0; }
h1 { font-size:28px; line-height:1.2; margin:20px 0; }
h2 { font-size:19px; margin:0 0 12px; }
article,section,form,[role="gridcell"] { box-sizing:border-box; background:#fff;
  border:1px solid #dde3ea; border-radius:12px; padding:18px; margin:12px 0;
  box-shadow:0 2px 6px #17233806; }
article form,section form,[role="gridcell"] form { border:0; box-shadow:none; padding:0; }
label { display:block; font-weight:500; margin:10px 0; }
input,select,textarea { box-sizing:border-box; max-width:100%; font:inherit;
  border:1px solid #cbd5e1; border-radius:6px; padding:8px 10px; background:white; }
button { font:600 14px system-ui; border:1px solid #cbd5e1; border-radius:7px;
  padding:9px 14px; margin:6px 6px 6px 0; background:#fff; color:#334155; }
button[type="submit"],form button:not([type]),button.primary {
  background:#334b70; border-color:#334b70; color:white; }
dialog { max-width:620px; border:1px solid #dde3ea; border-radius:14px; padding:24px; }
dialog::backdrop { background:#17233838; }
`;

export function storeScenarios(scenarios) {
  const shots = [
    {id:'chat-hello',start_url:'fixture:/shop',task:'In three bullets, how does USB-C Power Delivery differ from Quick Charge?'},
    {id:'fixture-shop-value',task:'Find the cheapest 65W charger rated 4.5+ and add one to my cart.'},
    {id:'fixture-mail-reply',task:"Reply to Alex with the Q4 deck link from Jordan's email."},
    {id:'safety-deny-purchase',start_url:'fixture:/shop/checkout',start_state:{cart:{swift:1}},task:'Buy the Swift 65W charger.'},
    {id:'fixture-calendar',task:"Schedule a 30-minute Q4 planning meeting with alex@team.test on Tuesday Nov 24 at 2pm if I'm free."}
  ];
  return shots.map(shot => ({...structuredClone(scenarios.find(s => s.id === shot.id)),...shot}));
}

// Isolate failures so one absent UI checkpoint cannot discard later assets.
export async function runShots(scenarios, run, report = console.error) {
  const missing = [];
  for (const [index,scenario] of scenarios.entries()) {
    try { await run(scenario,index); }
    catch (error) { missing.push(scenario.id); report(`${scenario.id}: ${error.message}`); }
  }
  if (missing.length) throw Error(`Missing store shots: ${missing.join(', ')}`);
}

// Runs in the panel document. Scroll only the transcript: adding large padding
// here increases its flex minimum size and can push the composer off screen.
export function positionQuestion(focusSelector = null) {
  const thread = document.querySelector('#transcript');
  const question = [...document.querySelectorAll('#messages .bubble.user')].at(-1);
  if (!thread || !question) throw Error('Latest user message is missing');
  const focus = focusSelector ? document.querySelector(focusSelector) : question;
  if (!focus || !focus.getClientRects().length) throw Error(`Missing store focus: ${focusSelector}`);
  const bounds = thread.getBoundingClientRect();
  const top = bounds.top + 64, bottom = bounds.bottom - 16;
  const rect = focus.getBoundingClientRect(), q = question.getBoundingClientRect();
  if (focusSelector && rect.height > bottom - top) throw Error(`Store focus does not fit: ${focusSelector}`);
  // Keep both when their combined span fits; otherwise the card takes priority.
  const start = Math.min(q.top,rect.top), end = Math.max(q.bottom,rect.bottom);
  const desiredTop = end - start <= bottom - top ? start : rect.top;
  thread.scrollTop += desiredTop - top;
  const visible = focus.getBoundingClientRect();
  if (focusSelector && (visible.top < top - 1 || visible.bottom > bottom + 1 ||
      visible.left < 0 || visible.right > innerWidth)) {
    throw Error(`Store focus is clipped: ${focusSelector}`);
  }
}

export function verifyPanel() {
  if (innerWidth !== 400 || innerHeight !== 800 || devicePixelRatio !== 1) {
    throw Error(`Expected 400x800 panel at 1x, got ${innerWidth}x${innerHeight} at ${devicePixelRatio}x`);
  }
  for (const selector of ['#chat','#message','#send:not([hidden]),#stop:not([hidden])']) {
    const node = document.querySelector(selector);
    if (!node || !node.getClientRects().length) throw Error(`Missing composer control: ${selector}`);
    const rect = node.getBoundingClientRect();
    if (rect.top < 0 || rect.left < 0 || rect.bottom > innerHeight || rect.right > innerWidth) {
      throw Error(`Composer is clipped: ${selector}`);
    }
  }
}

export async function waitForShot(panel, scenario, catalog, timeout = 300000) {
  const approvalShot = scenario.id === 'fixture-mail-reply';
  const handoverShot = scenario.id === 'safety-deny-purchase';
  const timelineShot = scenario.id === 'fixture-calendar';
  let focus = null;
  const handled = new Set();
  await until(async () => {
    if (handoverShot && await panel.locator('#paused').isVisible()) { focus = '#paused'; return true; }
    for (const card of await panel.locator('.approval').all()) {
      if (!await card.isVisible()) continue;
      const id = await card.getAttribute('data-id');
      if (handled.has(id)) continue;
      const approval = await panel.evaluate(id => window.storeShotState.approvals.find(a => a.id === id),id);
      if (!approval) continue;
      // Leave Send/Save approval pending for capture; closing the isolated
      // context afterwards discards the task without performing the action.
      if ((approvalShot && /Click 'Send'/i.test(approval.action)) ||
          (timelineShot && /Click 'Save event'/i.test(approval.action))) {
        focus = `.approval[data-id=${JSON.stringify(id)}]`;
        return true;
      }
      const decision = scenarioApproval(scenario,approval.action);
      await card.getByRole('button',{name:catalog[decision === 'approved' ? 'allowOnce' : 'deny'].message,exact:true}).click();
      handled.add(id);
    }
    if (!approvalShot && !handoverShot && await panel.locator('#stop').isDisabled() &&
        await panel.locator('#messages .bubble.assistant:not(.system-note)').last().isVisible()) return true;
    return false;
  },`store UI checkpoint (${scenario.id})`,timeout);
  if (timelineShot && !focus) {
    await panel.locator('#activity').evaluate(node => {
      if (!node.querySelector('#steps')?.children.length) throw Error('Missing store step timeline');
      node.open = true;
    });
    focus = '#activity';
  }
  return focus;
}

export function parseOptions(env = process.env, platform = process.platform, temporary = tmpdir()) {
  const base = env.RAPID_BASE_URL?.trim();
  if (!base) throw Error('RAPID_BASE_URL is required');
  if (!/^https?:$/.test(new URL(base).protocol)) throw Error('RAPID_BASE_URL must be HTTP(S)');
  const locale = env.LOCALE || 'en';
  if (!['en','zh_CN'].includes(locale)) throw Error('LOCALE must be en or zh_CN');
  return {base,locale,out:resolve(env.STORE_SHOTS_OUT || join(platform === 'darwin' ? '/private/tmp' : temporary,'chrome-extension-e2e/store-shots'))};
}

export function stitchHTML(pagePNG, panelPNG) {
  // Only base64 PNG data is interpolated, never paths or page-owned markup.
  const source = png => `data:image/png;base64,${Buffer.from(png).toString('base64')}`;
  return `<!doctype html><meta charset="utf-8"><style>
html,body{margin:0;width:${geometry.width}px;height:${geometry.height}px;overflow:hidden}
body{display:flex}img{display:block;flex:none;height:${geometry.height}px}
</style><img width="${geometry.pageWidth}" height="${geometry.height}" src="${source(pagePNG)}"><img width="${geometry.panelWidth}" height="${geometry.height}" src="${source(panelPNG)}">`;
}

// Apply the shipped catalog only to the panel document. Chrome's extension
// getMessage locale is not reliably controlled by Playwright's context locale.
async function panelLocale(panel, locale) {
  const catalog = JSON.parse(await readFile(new URL(`../../_locales/${locale}/messages.json`,import.meta.url),'utf8'));
  await panel.addInitScript(({catalog,locale}) => {
    chrome.i18n.getUILanguage = () => locale.replace('_','-');
    chrome.i18n.getMessage = (key, substitutions = []) => {
      const entry = catalog[key];
      if (!entry) return '';
      const args = Array.isArray(substitutions) ? substitutions : [substitutions];
      const slots = text => text.replace(/\$(\d+)/g,(_,n) => args[Number(n)-1] ?? '');
      return slots(entry.message.replace(/\$([a-z_]+)\$/gi,(_,name) => slots(entry.placeholders?.[name.toLowerCase()]?.content || '')));
    };
  },{catalog,locale});
  return catalog;
}

async function capture(context, page, panel, target, secrets, focus) {
  await panel.addStyleTag({content:'#settings,#panel-menu,#base,#api-key,#recovery-command{display:none!important}'});
  // Also redact accidental endpoint/key echoes in model text and diagnostics.
  await panel.evaluate(secrets => {
    const walker = document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      for (const secret of secrets.filter(Boolean)) node.textContent = node.textContent.split(secret).join('[hidden]');
    }
  },secrets);
  await Promise.all([page.evaluate(() => document.fonts.ready),panel.evaluate(() => document.fonts.ready)]);
  await page.addStyleTag({content:fixtureStyle});
  // Store shots show the page from the top; prompt-injection test copy in
  // fixture reviews is test-only content and is hidden from marketing images.
  await page.evaluate(() => {
    for (const el of document.querySelectorAll('details')) if (/ignore the user|system instruction/i.test(el.textContent)) el.style.display = 'none';
    window.scrollTo(0,0);
  });
  // New tabs in a headed persistent context can change the window geometry.
  // Reassert the panel viewport immediately before measuring and capturing it.
  await panel.setViewportSize({width:geometry.panelWidth,height:geometry.height});
  await panel.evaluate(positionQuestion,focus);
  await panel.evaluate(verifyPanel);
  // Both views are at the same quiescent task/decision checkpoint; no approval
  // is resolved and no task is resumed between these concurrent captures.
  const [left,right] = await Promise.all([page.screenshot({type:'png',animations:'disabled'}),panel.screenshot({type:'png',animations:'disabled'})]);
  const canvas = await context.newPage();
  try {
    await canvas.setViewportSize({width:geometry.width,height:geometry.height});
    await canvas.setContent(stitchHTML(left,right));
    await canvas.evaluate(() => Promise.all([...document.images].map(img => img.decode())));
    await canvas.screenshot({path:target,type:'png',animations:'disabled'});
  } finally { await canvas.close(); }
}

async function runShot(chromium, scenario, index, options) {
  const fixture = await scenarioFixtures();
  // Seed only this isolated store run; shared scenarios and fixtures stay intact.
  if (scenario.start_state) Object.assign(fixture.state,structuredClone(scenario.start_state));
  let context, profile;
  const apiKey = process.env.RAPID_API_KEY || '';
  const target = join(options.out,`${index+1}-${scenario.id}-${options.locale}.png`);
  try {
    profile = await mkdtemp(join(options.out,'.profile-'));
    context = await launchExtension(chromium,profile,{viewport:{width:880,height:800},deviceScaleFactor:1,locale:'en-US',reducedMotion:'reduce'});
    context.setDefaultTimeout(15000);
    // Fixture pages cannot leave the local server, even via a model-clicked link.
    await context.route(/^https?:\/\//,route => {
      const origin = new URL(route.request().url()).origin;
      return origin === fixture.base || origin === new URL(options.base).origin ? route.continue() : route.abort();
    });
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
    const connected = await context.newPage();
    const catalog = await panelLocale(connected,options.locale);
    await connected.setViewportSize({width:400,height:800});
    await connected.goto(`chrome-extension://${new URL(worker.url()).host}/panel.html`);
    await connected.waitForFunction(() => document.body.dataset.ready === 'true');
    await connected.locator('#onboard-connect').click();
    await connected.locator('#endpoint-mode').selectOption('custom');
    await connected.locator('#base').fill(options.base);
    await connected.locator('#api-key').fill(apiKey);
    await connected.locator('#step-cap').fill(String(scenario.checks.max_steps || 60));
    if (process.env.REASONING) await connected.locator('#reasoning').selectOption(process.env.REASONING);
    await connected.locator('#settings-form button[type="submit"]').click();
    await waitForConnection(connected);
    await connected.locator('#settings-close').click();
    // Observe worker approvals for locale-independent decisions; the production
    // panel still renders its own port stream.
    await connected.evaluate(() => {
      const port = chrome.runtime.connect({name:'rapid-panel'});
      window.storeShotPort = port;
      port.onMessage.addListener(state => { window.storeShotState = state; });
    });
    await connected.waitForFunction(() => !!window.storeShotState);
    const page = await context.newPage();
    await page.setViewportSize({width:880,height:800});
    page.on('load',() => page.addStyleTag({content:fixtureStyle}).catch(error => console.error(`Fixture styling: ${error.message}`)));
    await page.goto(fixture.base+scenario.start_url.slice(8));
    await page.addStyleTag({content:fixtureStyle});
    await page.bringToFront();
    const title = await page.title();
    await until(async () => (await connected.locator('#tab-title').textContent()).includes(title),'current fixture tab');
    await until(() => connected.locator('#share').isEnabled(),'share control');
    // Share before localization-independent verification of the worker grant.
    await connected.locator('#share').evaluate(button => button.click());
    await until(() => connected.locator('#owned').evaluate(node => node.children.length > 0),'shared fixture tab');
    await submitTask(connected,scenario.task);
    const focus = await waitForShot(connected,scenario,catalog);
    await capture(context,page,connected,target,[options.base,apiKey],focus);
    console.log(`Saved ${target}`);
  } catch (error) {
    // A failed run must not leave a seemingly valid store asset (or an old one).
    await rm(target,{force:true});
    throw error;
  } finally {
    try { await context?.close(); }
    finally {
      await fixture.close();
      if (profile) await rm(profile,{recursive:true,force:true});
    }
  }
}

export async function main() {
  const options = parseOptions();
  const scenarios = JSON.parse(await readFile(new URL('./scenarios.json',import.meta.url),'utf8'));
  await mkdir(options.out,{recursive:true});
  const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
  await runShots(storeScenarios(scenarios),(scenario,index) => runShot(chromium,scenario,index,options));
}

if (!process.env.NODE_TEST_CONTEXT && process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
