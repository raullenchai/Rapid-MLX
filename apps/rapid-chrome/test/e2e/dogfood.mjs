// Optional live-site runner. Importing this module does not launch a browser.
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launchExtension, connectPanel, shareTab, submitTask, readTrace, sleep, until } from './helpers.mjs';

export function approvalDecision(text) {
  // Site navigation is always allowed, even when the hostname contains "book".
  if (/Go to [^\s]+\?|Open [^\s]+ in a new tab\?/i.test(text)) return 'approved';
  return /\b(add another one|checkout|place order|pay|buy now|book|reserve|submit order|confirm purchase)\b/i.test(text) ? 'denied' : 'approved';
}

export async function runTask(chromium, task, out, base, options = {}) {
  const dir = join(out,task.id);
  await mkdir(dir,{recursive:true});
  const profile = await mkdtemp(join(out,'.profile-'));
  const started = Date.now();
  const result = {id:task.id, task:task.task, start_url:task.start_url, status:'error',
    seconds:0, steps:0, approvals:[], final_url:'', final_answer:'', errors:[]};
  let context, panel, page, trace;
  // Keep collecting other artifacts if one capture fails (e.g. a closed tab).
  const capture = async fn => {
    try { return await fn(); }
    catch (error) { result.errors.push(error.message); }
  };
  try {
    context = await launchExtension(chromium,profile,{viewport:{width:1440,height:900}});
    panel = await connectPanel(context,base,process.env.RAPID_API_KEY || '',options);
    page = await context.newPage();
    // DOMContentLoaded avoids waiting for every ad/resource on live storefronts.
    await capture(() => page.goto(task.start_url,{waitUntil:'domcontentloaded'}));
    await page.bringToFront();
    await page.screenshot({path:join(dir,'start.png')});
    await shareTab(panel);
    const deadline = Date.now()+(task.timeout_s ?? 300)*1000;
    await submitTask(panel,task.task);
    const handled = new Set();
    while (true) {
      if (await panel.locator('#stop').isDisabled()) {
        result.status = 'finished';
        break;
      }
      if (Date.now() >= deadline) {
        result.status = 'timeout';
        break;
      }
      if (options.maxSteps && ((await readTrace(panel))?.events?.filter(e => e.kind === 'tool').length || 0) > options.maxSteps) {
        result.status = 'step_limit';
        break;
      }
      if (await panel.locator('#paused').isVisible()) {
        result.status = 'needs_human';
        result.final_answer = await panel.locator('#status').innerText();
        break;
      }
      for (const card of await panel.locator('.approval').all()) {
        const id = await card.getAttribute('data-id');
        if (handled.has(id)) continue;
        const text = await card.innerText();
        const decision = (options.approvalDecision || approvalDecision)(text);
        const approval = {id,text,decision,at_seconds:(Date.now()-started)/1000};
        result.approvals.push(approval);
        // Persist the full card and policy decision before clicking.
        await writeFile(join(dir,'approvals.json'),JSON.stringify(result.approvals,null,2));
        await card.getByRole('button',{name:decision === 'approved' ? 'Allow once' : 'Deny',exact:true}).click();
        handled.add(id);
      }
      await sleep(100);
    }
  } catch (error) {
    result.errors.push(error.message);
  } finally {
    if (panel) {
      trace = await capture(() => readTrace(panel));
      await capture(async () => {
        if (!await panel.locator('#stop').isDisabled()) {
          await panel.locator('#stop').click();
          await until(() => panel.locator('#stop').isDisabled(),'stop cleanup');
          trace = await readTrace(panel) ?? trace;
        }
      });
      await capture(async () => writeFile(join(dir,'transcript.txt'),await panel.locator('body').innerText()));
      await capture(async () => {
        const answers = panel.locator('#messages .assistant');
        if (await answers.count()) if (!result.final_answer) result.final_answer = await answers.last().innerText();
      });
    }
    if (page) {
      result.final_url = page.url();
      await capture(() => page.screenshot({path:join(dir,'end.png')}));
    }
    result.seconds = Number(((Date.now()-started)/1000).toFixed(1));
    result.steps = trace?.events?.filter(event => event.kind === 'tool').length || 0;
    await capture(() => context?.close());
    await capture(() => rm(profile,{recursive:true,force:true}));
    await writeFile(join(dir,'approvals.json'),JSON.stringify(result.approvals,null,2));
    await writeFile(join(dir,'trace.json'),JSON.stringify(trace ?? null,null,2));
    await writeFile(join(dir,'result.json'),JSON.stringify(result,null,2));
  }
  return result;
}

const cell = value => String(value).replace(/\r?\n/g,' ').replace(/\|/g,'&#124;').replace(/</g,'&lt;');
export function summary(results) {
  const lines = [
    '| id | finished/timeout | seconds | steps | approvals asked/approved/denied | final URL | final answer (first 200 chars) |',
    '| --- | --- | ---: | ---: | --- | --- | --- |'
  ];
  for (const r of results) {
    const approved = r.approvals.filter(a => a.decision === 'approved').length;
    lines.push(`| ${[r.id,r.status,r.seconds,r.steps,`${r.approvals.length}/${approved}/${r.approvals.length-approved}`,r.final_url,r.final_answer.slice(0,200)].map(cell).join(' | ')} |`);
  }
  return lines.join('\n')+'\n';
}

export async function main() {
  if (!process.argv[2]) throw new Error('Usage: RAPID_BASE_URL=http://localhost:8000 node dogfood.mjs tasks.json');
  if (!process.env.RAPID_BASE_URL?.trim()) throw new Error('RAPID_BASE_URL is required');
  const tasks = JSON.parse(await readFile(process.argv[2],'utf8'));
  if (!Array.isArray(tasks)) throw new Error('Tasks must be a JSON array');
  const ids = new Set();
  for (const task of tasks) {
    if (!task || typeof task.id !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(task.id) || ids.has(task.id) ||
      typeof task.start_url !== 'string' || !/^https?:$/.test(new URL(task.start_url).protocol) ||
      typeof task.task !== 'string' || !task.task.trim() ||
      (task.timeout_s !== undefined && (!Number.isFinite(task.timeout_s) || task.timeout_s <= 0))) {
      throw new Error('Each task needs a unique safe id, HTTP(S) start_url, task text, and optional positive timeout_s');
    }
    ids.add(task.id);
  }
  const only = process.env.DOGFOOD_ONLY?.split(',').map(id => id.trim()).filter(Boolean);
  if (only?.some(id => !ids.has(id))) throw new Error('DOGFOOD_ONLY contains an unknown task id');
  const out = resolve(process.env.DOGFOOD_OUT || '/private/tmp/chrome-extension-e2e/dogfood');
  await mkdir(out,{recursive:true});
  const {chromium} = await import('playwright');
  const results = [];
  for (const task of tasks.filter(task => !only?.length || only.includes(task.id))) {
    const result = await runTask(chromium,task,out,process.env.RAPID_BASE_URL);
    results.push(result);
    const approved = result.approvals.filter(a => a.decision === 'approved').length;
    console.log(`${result.id} ${result.status} seconds=${result.seconds} steps=${result.steps} approvals=${result.approvals.length}/${approved}/${result.approvals.length-approved} url=${JSON.stringify(result.final_url)} answer=${JSON.stringify(result.final_answer.slice(0,200))}${result.errors.length ? ` errors=${JSON.stringify(result.errors)}` : ''}`);
    await writeFile(join(out,'summary.md'),summary(results));
  }
  if (results.some(result => result.status === 'error' || result.errors.length)) process.exitCode = 1;
}

if (!process.env.NODE_TEST_CONTEXT && process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
