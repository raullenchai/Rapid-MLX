import { fileURLToPath } from 'node:url';

export const sleep = ms => new Promise(resolve => setTimeout(resolve,ms));
export async function until(fn, label, timeout = 15000) {
  const end = Date.now()+timeout;
  while (Date.now()<end) { if (await fn()) return; await sleep(50); }
  throw new Error(`Timed out: ${label}`);
}

export async function launchExtension(chromium, profile, options = {}) {
  const extension = fileURLToPath(new URL('../../',import.meta.url));
  return chromium.launchPersistentContext(profile,{
    ...(process.env.CHROME_EXECUTABLE ? {executablePath:process.env.CHROME_EXECUTABLE} : {channel:'chromium'}),headless:process.env.HEADLESS === '1',
    ...options,
    args:[`--disable-extensions-except=${extension}`,`--load-extension=${extension}`]
  });
}

export async function connectPanel(context, base, apiKey = '', options = {}) {
  const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
  const id = new URL(worker.url()).host;
  const panel = await context.newPage();
  await panel.goto(`chrome-extension://${id}/panel.html`);
  await panel.waitForFunction(() => document.body.dataset.ready === 'true');
  // First-run onboarding is inline, not a modal. Use its connection action;
  // the first submitted task completes the tour and hides it.
  if (await panel.locator('#onboarding').isVisible()) await panel.locator('#onboard-connect').click();
  else { await panel.locator('#settings-toggle').click(); await panel.locator('#menu-settings').click(); }
  await panel.locator('#endpoint-mode').selectOption('custom');
  await panel.locator('#base').fill(base);
  await panel.locator('#api-key').fill(apiKey);
  if (options.maxSteps) await panel.locator('#step-cap').fill(String(options.maxSteps));
  if (process.env.REASONING) await panel.locator('#reasoning').selectOption(process.env.REASONING);
  await panel.getByRole('button',{name:'Save & connect'}).click();
  await until(async () => /^Connected(?:$|\.)/.test(await panel.locator('#status').textContent()),'engine connection');
  await panel.locator('#settings-close').click();
  return panel;
}

export async function shareTab(panel) {
  // Keep the working tab active in Chrome while clicking Share in the panel.
  await panel.locator('#share').evaluate(button => button.click());
  await until(async () => (await panel.locator('#owned').textContent()).includes('shared'),'shared tab');
}

export const readTrace = panel => panel.evaluate(async () => (await chrome.storage.session.get('debugTrace')).debugTrace);

export async function submitTask(panel, text) {
  await until(async () => await panel.locator('#stop').isDisabled(),'previous task cleanup');
  const previous = (await readTrace(panel))?.id;
  await panel.locator('#message').fill(text);
  await panel.locator('#chat').evaluate(form => form.requestSubmit());
  await until(async () => {
    const trace = await readTrace(panel);
    return trace?.id !== previous && trace?.task === text;
  },'new task accepted');
}
