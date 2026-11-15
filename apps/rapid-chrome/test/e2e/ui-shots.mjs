// Offline visual fixtures. Requires an already-installed Playwright + Chromium.
// Conversation, approval, takeover and overlay use the real worker/fake engine.
// Challenge and error fixtures use the production renderer deterministically.
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { fixtures, fakeEngine } from './fixtures.mjs';
import { launchExtension, connectPanel, shareTab, submitTask, until } from './helpers.mjs';

export async function main() {
  const {chromium} = await import('playwright');
  const output = process.env.SHOTS_OUT || '/private/tmp/chrome-extension-e2e/ui-shots';
  await mkdir(output,{recursive:true});
  const profile = await mkdtemp('/private/tmp/chrome-extension-ui-profile-');
  const fixture = await fixtures();
  const call = (name,args = {}) => ({role:'assistant',content:null,tool_calls:[{id:crypto.randomUUID(),type:'function',function:{name,arguments:JSON.stringify(args)}}]});
  let release, context;
  const engine = await fakeEngine({script:[call('read_page'), async () => {
    return new Promise(resolve => { release = () => resolve({...call('read_page',{query:'Send'}),content:'I’m reviewing the form before submission.'}); });
  }, body => {
    const observation = JSON.parse(body.messages.findLast(m => m.role === 'tool').content).text;
    const ref = observation.split('\n').find(line => /button "Send"/.test(line))?.match(/\[(e\d+)\]/)?.[1];
    if (!ref) throw new Error('Send control missing from fixture');
    return call('click',{ref});
  },{role:'assistant',content:`**Draft reviewed.** Nothing was sent.\n\n- The form is ready for your review.\n- You kept control of the submission.\n\n[Open the page](${fixture.base}/)`}]});
  try {
    context = await launchExtension(chromium,profile,{viewport:{width:400,height:860},colorScheme:'light',reducedMotion:'reduce'});
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
    const extensionId = new URL(worker.url()).host;
    const first = await context.newPage();
    await first.goto(`chrome-extension://${extensionId}/panel.html`);
    await first.waitForFunction(() => document.body.dataset.ready === 'true');
    await first.screenshot({path:path.join(output,'onboarding.png')});
    await first.close();
    const panel = await connectPanel(context,engine.base);
    const shot = name => panel.screenshot({path:path.join(output,`${name}.png`)});
    await shot('empty');
    await panel.locator('#settings-toggle').click(); await shot('settings');
    await panel.locator('#settings-close').click();
    const page = await context.newPage(); await page.goto(fixture.base); await page.bringToFront();
    await until(async () => (await panel.locator('#tab-title').textContent()).includes('Rapid fixture'),'current tab');
    await shareTab(panel);
    await submitTask(panel,'Review this form and ask before submitting.');
    await until(() => !!release,'held inference request');
    await panel.locator('#activity').evaluate(node => { node.open = true; }); await shot('running');
    await page.screenshot({path:path.join(output,'overlay.png')});
    await page.bringToFront();
    await until(async () => await panel.locator('#share').isEnabled(),'take-over control');
    await panel.locator('#share').evaluate(node => node.click());
    await panel.locator('#paused').waitFor({state:'visible'}); await shot('takeover');
    await panel.locator('#resume').evaluate(node => node.click());
    release(); // The response reaches the agent checkpoint that processes Resume.
    await panel.locator('#paused').waitFor({state:'hidden'});
    await panel.locator('.approval').waitFor(); await shot('approval');
    await panel.getByRole('button',{name:'Deny',exact:true}).click();
    await until(async () => await panel.locator('#stop').isDisabled(),'task finished');
    await panel.locator('#activity').evaluate(node => { node.open = false; }); await shot('done');
    await panel.emulateMedia({colorScheme:'dark'}); await shot('dark');
    await panel.setViewportSize({width:320,height:800}); await shot('dark-320');
    await panel.setViewportSize({width:400,height:860});
    const baseState = {settings:{base:engine.base,model:'fixture-model',stepCap:60},models:['fixture-model'],tabs:[],steps:[],approvals:[],connected:true,status:'Working',running:true,messages:[{role:'user',content:'Read the page and summarize the key points.'}]};
    const display = state => panel.evaluate(async state => {
      const {render} = await import(chrome.runtime.getURL('panel.js')); render(state);
    },state);
    await display({...baseState,paused:true,pauseReason:'challenge'}); await shot('challenge');
    await display({...baseState,running:false,connected:false,status:'Error',messages:[...baseState.messages,{role:'assistant',kind:'notice',content:'Engine unavailable.'}]}); await shot('error');
    console.log(`Saved UI screenshots to ${output}`);
  } finally {
    release?.(); await context?.close();
    await Promise.all([engine.close(),fixture.close()]);
    await rm(profile,{recursive:true,force:true});
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
