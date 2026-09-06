import { expect, test } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * Prose rendering, measured in a real browser.
 *
 * Every bug here was invisible to the unit tests: the markup was correct and
 * the defect was in the computed style or the box. Tailwind's preflight is
 * the common cause — it strips the UA stylesheet's anchor and list-marker
 * defaults, so anything relying on them silently renders as plain text.
 */

const SAMPLE = [
  '- 列表A',
  '- 列表B',
  '',
  '1. 有序一',
  '2. 有序二',
  '',
  '[链接](https://example.com)',
  '',
  '```python',
  'def f(x):',
  '    return x * 2',
  '```',
].join('\n');

function frames(text: string) {
  // One character per frame would be closer to a real stream, but the whole
  // body in one frame is enough: these assert the SETTLED render.
  return [`data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`];
}

async function render(page: import('@playwright/test').Page, baseURL: string, markdown: string) {
  await page.goto(baseURL);
  await page.getByLabel('Message').fill('show me');
  await page.getByRole('button', { name: 'Send' }).click();
  // Settled, not streaming: the two renderers differ, and the streaming one
  // deliberately skips highlighting.
  await expect(page.getByRole('log').getByText(markdown.split('\n')[0]!.replace('- ', ''))).toBeVisible();
}

test('a code block fills the transcript width', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    chatFrames: [frames(SAMPLE)],
  });
  try {
    await render(page, stub.baseURL, SAMPLE);

    // The answer column is `items-start`, which shrink-wraps its children —
    // so without an explicit width the block is only as wide as its longest
    // line. A two-line snippet then renders as a narrow box in a wide
    // transcript (measured: 154px inside a 358px row).
    //
    // Compared against the ROW, not the holder: the holder is the element
    // that was shrink-wrapping, so measuring the block against it compares
    // two collapsed boxes and passes against the defect. Verified.
    const width = await page.evaluate(() => {
      const log = document.querySelector('[role="log"]')!;
      const block = log.querySelector('.md-code-wrap');
      if (!(block instanceof HTMLElement)) return null;
      const holder = block.closest('.leading-relaxed') as HTMLElement;
      const row = holder.parentElement as HTMLElement;
      return { block: block.getBoundingClientRect().width, row: row.getBoundingClientRect().width };
    });

    expect(width).not.toBeNull();
    expect(width!.block).toBeGreaterThan(width!.row - 2);
  } finally {
    await stub.close();
  }
});

test('a link is visibly a link and opens safely', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    chatFrames: [frames(SAMPLE)],
  });
  try {
    await render(page, stub.baseURL, SAMPLE);

    const link = page.getByRole('link', { name: '链接' });
    await expect(link).toHaveAttribute('href', 'https://example.com/');
    // A new tab, and no window.opener handle back into this page.
    await expect(link).toHaveAttribute('target', '_blank');
    await expect(link).toHaveAttribute('rel', /noopener/);

    // Preflight removes the UA's anchor styling, so a link with no rule of
    // its own is indistinguishable from the prose around it. Both signals are
    // asserted because either alone can be argued away.
    const style = await link.evaluate((node) => {
      const computed = getComputedStyle(node);
      return { decoration: computed.textDecorationLine, color: computed.color };
    });
    expect(style.decoration).toContain('underline');

    const proseColor = await page
      .getByRole('log')
      .locator('p')
      .first()
      .evaluate((node) => getComputedStyle(node).color);
    expect(style.color).not.toBe(proseColor);
  } finally {
    await stub.close();
  }
});

test('list items keep their markers', async ({ page }) => {
  const stub = await startStub({
    engineState: 'ready',
    model: 'qwen3-4b',
    chatFrames: [frames(SAMPLE)],
  });
  try {
    await render(page, stub.baseURL, SAMPLE);

    // Preflight sets `list-style: none` on ul/ol, so the list padding was
    // reserving space for markers that never drew — every list read as an
    // indented block of loose lines.
    const unordered = page.getByRole('log').locator('ul.md-list').first();
    const ordered = page.getByRole('log').locator('ol.md-list').first();

    await expect(unordered).toHaveCSS('list-style-type', 'disc');
    await expect(ordered).toHaveCSS('list-style-type', 'decimal');
  } finally {
    await stub.close();
  }
});
