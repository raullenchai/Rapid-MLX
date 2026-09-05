import { expect, test } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * Text-to-image.
 *
 * The engine serves ONE model at a time, so the surface is only usable when
 * the loaded model is an image model. That precondition is stated on screen
 * rather than worked around — silently restarting the engine would discard a
 * chat the user is in the middle of.
 */

async function openImages(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  const drawer = page.getByLabel('Open sidebar');
  if (await drawer.isVisible()) await drawer.click();
  await page.getByRole('button', { name: 'Images' }).click();
}

test('a chat model loaded means the surface explains what to do', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    await openImages(page, stub.baseURL);

    // The reason AND the consequence: switching stops the chat model, which
    // the user has to know before they go and do it.
    await expect(page.getByText(/Choose an image model/)).toBeVisible();
    await expect(page.getByText(/one model at a time/)).toBeVisible();
    await expect(page.getByLabel('Prompt')).toHaveAttribute(
      'placeholder',
      'Choose an image model first',
    );
    await expect(page.getByRole('button', { name: 'Generate' })).toBeDisabled();
  } finally {
    await stub.close();
  }
});

test('a model that is still starting reports itself on the images page', async ({ page }) => {
  const stub = await startStub({
    engineState: 'starting',
    model: 'flux2-klein-4b',
    detail: 'loading weights',
  });
  try {
    await openImages(page, stub.baseURL);

    // The same readiness band the chat uses — the images page used to say
    // nothing at all while a model was loading.
    await expect(page.getByText('Starting flux2-klein-4b')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Generate' })).toBeDisabled();
  } finally {
    await stub.close();
  }
});

test('a failed start shows the engine reason on the images page', async ({ page }) => {
  const stub = await startStub({
    engineState: 'failed',
    model: 'flux2-klein-4b',
    // Verbatim from the engine: the CLI's `Error:` line is the one that says
    // what to do, and a null detail leaves the page with "unknown error".
    detail: 'image generation requires the `rapid-mlx[image]` Python extra.',
  });
  try {
    await openImages(page, stub.baseURL);
    await expect(page.getByText(/rapid-mlx\[image\]/)).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('an image model chosen for Images does not leak into the chat', async ({ page }) => {
  // The engine is SHARED — one `serve` child — but the selection is not.
  // With a single `selectedAlias`, picking an image model retargeted the chat
  // and the chat then reported the image model's start failure as its own,
  // offering a Retry that would switch to a model it cannot use.
  const stub = await startStub({
    engineState: 'failed',
    model: 'flux2-klein-4b',
    detail: 'image generation requires the `rapid-mlx[image]` Python extra.',
  });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();

    // The chat is on its own model and says nothing about the image failure.
    await expect(page.getByText(/Couldn't start flux2-klein-4b/)).toHaveCount(0);
    await expect(page.getByText(/rapid-mlx\[image\]/)).toHaveCount(0);
    await expect(page.getByRole('button', { name: /^Model:/ })).not.toContainText(
      'flux2-klein-4b',
    );

    // The images surface, which IS on that model, reports it.
    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Images' }).click();
    await expect(page.getByText(/rapid-mlx\[image\]/)).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('the empty canvas previews the size that will be rendered', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
  try {
    await openImages(page, stub.baseURL);

    // Without a frame to act on, changing the size controls before the first
    // render appears to do nothing at all.
    await expect(page.getByText('Draw anything')).toBeVisible();
    await expect(page.getByText('512x512', { exact: true })).toBeVisible();

    await page.getByRole('radio', { name: '3:4' }).click();
    await expect(page.getByText('384x512', { exact: true })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test.describe('with an image model loaded', () => {
  test('renders a prompt and offers to save the result', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
    try {
      await openImages(page, stub.baseURL);

      await page.getByLabel('Prompt').fill('a cat on a bicycle');
      await page.getByRole('button', { name: 'Generate' }).click();

      const image = page.getByAltText('a cat on a bicycle');
      await expect(image).toBeVisible();
      await expect(page.getByRole('button', { name: 'Save' })).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('offers starter prompts and fills the box with one', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
    try {
      await openImages(page, stub.baseURL);

      // An empty prompt box invites a one-word prompt and a disappointing
      // result blamed on the model.
      const suggestion = page.getByRole('button', { name: /cozy ramen shop/ });
      await expect(suggestion).toBeVisible();
      await suggestion.click();

      await expect(page.getByLabel('Prompt')).toHaveValue(/cozy ramen shop/);
      await expect(page.getByRole('button', { name: 'Generate' })).toBeEnabled();
    } finally {
      await stub.close();
    }
  });

  test('the starter prompts give way to the result', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
    try {
      await openImages(page, stub.baseURL);

      await page.getByLabel('Prompt').fill('a cat on a bicycle');
      await page.getByRole('button', { name: 'Generate' }).click();
      await expect(page.getByAltText('a cat on a bicycle')).toBeVisible();

      // Once there is an image, the suggestions would compete with it.
      await expect(page.getByRole('button', { name: /cozy ramen shop/ })).toHaveCount(0);
    } finally {
      await stub.close();
    }
  });

  test('Generate stays disabled until there is a prompt', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
    try {
      await openImages(page, stub.baseURL);

      const generate = page.getByRole('button', { name: 'Generate' });
      await expect(generate).toBeDisabled();
      await page.getByLabel('Prompt').fill('something');
      await expect(generate).toBeEnabled();

      // Whitespace is not a prompt.
      await page.getByLabel('Prompt').fill('   ');
      await expect(generate).toBeDisabled();
    } finally {
      await stub.close();
    }
  });

  test('the size the engine will be sent follows the pickers', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'flux2-klein-4b' });
    try {
      await openImages(page, stub.baseURL);

      // Shown on the control itself so the user can see what they are asking
      // for before spending minutes of GPU time on it.
      const sizeButton = page.getByRole('button', { name: 'Long edge' });
      await expect(sizeButton).toContainText('512 × 512');

      await page.getByRole('radio', { name: '3:4' }).click();
      await expect(sizeButton).toContainText('384 × 512');

      await sizeButton.click();
      // Full dimensions, not the long edge alone: "1024 px" does not say what
      // a 3:4 render will actually come out as.
      await page.getByRole('menuitem', { name: '768 × 1024 px' }).click();
      await expect(sizeButton).toContainText('768 × 1024');
    } finally {
      await stub.close();
    }
  });

  test('a render in flight shows determinate progress and can be stopped', async ({ page }) => {
    const stub = await startStub({
      engineState: 'ready',
      model: 'flux2-klein-4b',
      imageDelayMs: 4000,
      imageProgress: { step: 3, total: 8 },
    });
    try {
      await openImages(page, stub.baseURL);

      await page.getByLabel('Prompt').fill('slow render');
      await page.getByRole('button', { name: 'Generate' }).click();

      // Diffusion has a fixed step count, so this is a true fraction rather
      // than a spinner pretending to know.
      await expect(page.getByText('3 / 8')).toBeVisible();
      // Over the canvas, not a strip above it: the canvas is the subject.
      await expect(page.getByText(/s left|Estimating/)).toBeVisible();

      // Exact: the HUD over the canvas carries its own "Cancel" control, and
      // an unanchored /Stop/ matches both.
      const stop = page.getByRole('button', { name: 'Stop', exact: true });
      await expect(stop).toBeVisible();
      await stop.click();

      // Stopping returns the surface immediately rather than waiting out the
      // render it just cancelled.
      await expect(page.getByRole('button', { name: 'Generate' })).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('the request that starts a render is not held open for it', async ({ page }) => {
    // The reason renders are jobs. The engine answers only once the whole
    // image is finished, so relaying inline left a connection with no bytes
    // flowing for minutes — Cloudflare cuts that at 100 s and returns 524.
    // A ~20 s generation survived the tunnel and a slower edit never did.
    const stub = await startStub({
      engineState: 'ready',
      model: 'flux2-klein-4b',
      imageDelayMs: 4000,
      imageProgress: { step: 2, total: 8 },
    });
    try {
      let startMs = 0;
      await page.route('**/api/images/jobs', async (route) => {
        const began = Date.now();
        await route.continue();
        startMs = Date.now() - began;
      });

      await openImages(page, stub.baseURL);
      await page.getByLabel('Prompt').fill('slow render');
      await page.getByRole('button', { name: 'Generate' }).click();

      // Progress proves the render is genuinely still going while the POST
      // that started it has already come back.
      await expect(page.getByText('2 / 8')).toBeVisible();
      expect(startMs).toBeLessThan(1000);

      // And the result still arrives, over the poll rather than that request.
      await expect(page.getByAltText('slow render')).toBeVisible({ timeout: 15000 });
    } finally {
      await stub.close();
    }
  });
});

test('the sidebar lists the destinations vertically', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    await page.goto(stub.baseURL);
    await expect(page.getByLabel('Message')).toBeVisible();
    await page.getByLabel('Open sidebar').click();

    const nav = page.getByRole('navigation', { name: 'Views' });
    await expect(nav.getByRole('button', { name: 'New Chat' })).toBeVisible();
    await expect(nav.getByRole('button', { name: 'Images' })).toBeVisible();
    await expect(nav.getByRole('button', { name: 'Audio' })).toBeEnabled();
  } finally {
    await stub.close();
  }
});

test('New Chat is the way back from images, and keeps earlier chats', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    await page.goto(stub.baseURL);
    await page.getByLabel('Message').fill('remember this');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByRole('log')).toContainText('Hello there.');

    await page.getByLabel('Open sidebar').click();
    await page.getByRole('button', { name: 'Images' }).click();
    await expect(page.getByRole('log')).toHaveCount(0);

    // Wait for the drawer's exit animation to finish before reopening. Radix
    // swallows a click that lands during it — reproducible on the pre-existing
    // New chat path too, which closes the drawer the same way.
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toHaveCount(0);
    await page.getByLabel('Open sidebar').click();
    // Returns to chat AND starts a new one: leaving the images view on screen
    // would make New Chat look like it had done nothing.
    await page.getByRole('button', { name: 'New Chat' }).click();
    await expect(page.getByLabel('Message')).toBeVisible();

    // The earlier conversation is untouched — switching surface is
    // navigation, not a reset.
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toHaveCount(0);
    await page.getByLabel('Open sidebar').click();
    await expect(page.getByRole('dialog', { name: 'Conversations' })).toContainText(
      'remember this',
    );
  } finally {
    await stub.close();
  }
});
