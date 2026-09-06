import { expect, test } from '@playwright/test';
import { startStub } from './stub-server';

/**
 * The audio lane.
 *
 * It rides on WHATEVER model the engine is serving — the child is spawned
 * with `--enable-audio` and the engine's gate short-circuits on that flag
 * before it looks at the model. So speech is usable while a CHAT model is
 * loaded, and no model switch is offered. What the pickers here choose is
 * which voice engine the lane loads, not what the engine SERVES.
 */

/** Vertical midpoint, for asserting two controls share a row. */
function centreY(box: { y: number; height: number }): number {
  return box.y + box.height / 2;
}

async function openAudio(page: import('@playwright/test').Page, baseURL: string) {
  await page.goto(baseURL);
  await expect(page.getByLabel('Message')).toBeVisible();
  const drawer = page.getByLabel('Open sidebar');
  if (await drawer.isVisible()) await drawer.click();
  await page.getByRole('button', { name: 'Audio' }).click();
}

test('audio works while a CHAT model is loaded', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    await openAudio(page, stub.baseURL);

    // The whole point of the lane: no audio model has to be STARTED, and no
    // switch is offered — speech runs on the chat model's process.
    await expect(page.getByLabel('Text')).toBeVisible();
    await expect(page.getByRole('button', { name: /af_heart/ })).toBeVisible();
    await expect(page.getByText(/Speech is running on qwen3-4b/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start', exact: true })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('a load in progress is reported, not treated as a dead end', async ({ page }) => {
  const stub = await startStub({ engineState: 'starting', model: 'qwen3-4b' });
  try {
    await openAudio(page, stub.baseURL);

    await expect(page.getByText(/qwen3-4b is loading/)).toBeVisible();
    // No second Start while one load is already running.
    await expect(page.getByRole('button', { name: 'Start', exact: true })).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('an idle engine offers to start the audio model here', async ({ page }) => {
  const stub = await startStub({ engineState: 'stopped', model: null });
  try {
    await openAudio(page, stub.baseURL);

    // `serve <audio-alias>` has a dedicated fork in the CLI, so this is a
    // real path. Sending the user to the chat to start something else is
    // what the Mac deliberately does NOT do (`ensureVoiceLane`).
    await expect(page.getByText(/engine is idle/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Start', exact: true })).toBeEnabled();
    await expect(page.getByText(/needs a running model/)).toHaveCount(0);
  } finally {
    await stub.close();
  }
});

test('starting the audio model makes the lane usable', async ({ page }) => {
  // `loadSettlesReady`: the real route answers `starting` and settles minutes
  // later, and this spec is about what the lane can do AFTERWARDS. The
  // loading window itself is covered by `model-start.spec.ts`.
  const stub = await startStub({
    engineState: 'stopped',
    model: null,
    loadSettlesReady: true,
  });
  try {
    await openAudio(page, stub.baseURL);
    await page.getByRole('button', { name: 'Start', exact: true }).click();

    // The voice list is fetched only once the lane can answer — asking a
    // stopped engine 503s, which reads as "this model has no voices".
    await expect(page.getByRole('button', { name: /af_heart/ })).toBeVisible();
    await expect(page.getByText(/Running as the served model/)).toBeVisible();
  } finally {
    await stub.close();
  }
});

test('the two modes are Text to Speech and Speech to Text', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    await openAudio(page, stub.baseURL);

    const modes = page.getByRole('radiogroup', { name: 'Audio mode' });
    await expect(modes.getByRole('radio', { name: 'Text to Speech' })).toBeVisible();
    await expect(modes.getByRole('radio', { name: 'Speech to Text' })).toBeVisible();
  } finally {
    await stub.close();
  }
});

test.describe('text to speech', () => {
  test('generates audio and offers to save it', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);

      const generate = page.getByRole('button', { name: /Generate speech/ });
      // Nothing to say yet.
      await expect(generate).toBeDisabled();

      await page.getByLabel('Text').fill('hello from rapid');
      await expect(generate).toBeEnabled();
      await generate.click();

      await expect(page.getByText('Speech ready')).toBeVisible();
      await expect(page.locator('audio')).toBeVisible();

      // Save acts on the audio, so it shares the player's row rather than
      // reading as a separate step below it. Measured, not inferred from the
      // markup: the native player has a wide intrinsic width and will push a
      // sibling off a phone screen without `min-w-0`.
      const player = (await page.locator('audio').boundingBox())!;
      const save = (await page.getByRole('link', { name: 'Save' }).boundingBox())!;
      expect(save.x).toBeGreaterThan(player.x + player.width - 1);
      expect(Math.abs(centreY(save) - centreY(player))).toBeLessThan(8);
      expect(save.x + save.width).toBeLessThanOrEqual(page.viewportSize()!.width);

      // The player must actually DECODE it, not merely exist. The result is
      // an object URL, and `media-src` falls back to `default-src 'self'`
      // without its own directive — which does not cover `blob:`, so the
      // element failed with MediaError 4 and sat at 0:00/0:00 while the
      // same URL still downloaded fine. `toBeVisible` cannot see that.
      const failure = await page.locator('audio').evaluate(
        (element: HTMLAudioElement) =>
          new Promise<string>((resolve) => {
            if (element.readyState > 0) return resolve('');
            element.addEventListener('loadedmetadata', () => resolve(''));
            element.addEventListener('error', () =>
              resolve(`MediaError ${element.error?.code}`),
            );
            setTimeout(() => resolve(`stalled readyState=${element.readyState}`), 4000);
          }),
      );
      expect(failure).toBe('');
      await expect(page.getByRole('link', { name: 'Save' })).toHaveAttribute(
        'download',
        'rapid-speech.wav',
      );
    } finally {
      await stub.close();
    }
  });

  test('whitespace alone is not something to say', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);
      await page.getByLabel('Text').fill('    ');
      await expect(page.getByRole('button', { name: /Generate speech/ })).toBeDisabled();
    } finally {
      await stub.close();
    }
  });

  test('a voice can be chosen from the ones the lane reports', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);

      await page.getByRole('button', { name: /af_heart/ }).click();
      await page.getByRole('menuitem', { name: /bf_emma/ }).click();
      await expect(page.getByRole('button', { name: /bf_emma/ })).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('each voice says what it is, decoded from its id', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);
      await page.getByRole('button', { name: /af_heart/ }).click();

      // `af_heart` is systematic — American English, female. A bare id is
      // unpickable, which is the whole reason the detail is there.
      await expect(page.getByRole('menuitem', { name: /af_heart/ })).toContainText(
        'American English · Female',
      );
      await expect(page.getByRole('menuitem', { name: /bf_emma/ })).toContainText(
        'British English · Female',
      );
    } finally {
      await stub.close();
    }
  });

  test('a voice can be previewed without closing the menu', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);
      await page.getByRole('button', { name: /af_heart/ }).click();

      // Previewing is COMPARISON: a menu that closes on the first sample
      // makes comparing two voices six clicks instead of two.
      await page.getByRole('button', { name: 'Preview am_adam' }).click();
      await expect(page.getByRole('menuitem', { name: /bf_emma/ })).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('the speech model is named, not left as a registry alias', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openAudio(page, stub.baseURL);

      // "Kokoro 82M", not "kokoro". The summary lives on the picker's rows —
      // the row's own caption states the LANE, which is the question the
      // setup card is answering.
      await expect(page.getByRole('button', { name: /^Model:/ })).toContainText('Kokoro 82M');
      await page.getByRole('button', { name: /^Model:/ }).click();
      await expect(page.getByRole('menuitem', { name: /Kokoro 82M/ }).first()).toContainText(
        '54 built-in voices',
      );
    } finally {
      await stub.close();
    }
  });

  test('a lane that cannot list voices explains itself inline', async ({ page }) => {
    const stub = await startStub({
      engineState: 'ready',
      model: 'qwen3-4b',
      voicesFailure: {
        status: 503,
        type: 'api_error',
        message: "Kokoro TTS requires the optional `misaki` package. Install with: pip install 'rapid-mlx[audio]'",
      },
    });
    try {
      await openAudio(page, stub.baseURL);

      // Inline, not a toast: without voices this panel cannot be used at
      // all, so the message belongs where the control is. And the engine's
      // own install instruction has to survive — it is the actionable part.
      await expect(page.getByText(/rapid-mlx\[audio\]/)).toBeVisible();
      await expect(page.getByRole('button', { name: /Generate speech/ })).toBeDisabled();
    } finally {
      await stub.close();
    }
  });
});

test.describe('speech to text', () => {
  async function openDictation(page: import('@playwright/test').Page, baseURL: string) {
    await openAudio(page, baseURL);
    await page.getByRole('radio', { name: 'Speech to Text' }).click();
  }

  test('offers a record control', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openDictation(page, stub.baseURL);

      await expect(page.getByRole('button', { name: 'Start recording' })).toBeVisible();
      await expect(page.getByText(/Tap to record/)).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('a denied microphone is reported rather than failing silently', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      // WebKit under Playwright has no real microphone, and permission
      // cannot be granted headlessly — so this drives the path that a real
      // denial takes, which is the one worth pinning.
      await page.addInitScript(() => {
        Object.defineProperty(navigator, 'mediaDevices', {
          configurable: true,
          value: { getUserMedia: () => Promise.reject(new Error('denied')) },
        });
      });
      await openDictation(page, stub.baseURL);
      await page.getByRole('button', { name: 'Start recording' }).click();

      await expect(page.getByText(/microphone access/)).toBeVisible();
    } finally {
      await stub.close();
    }
  });

  test('defaults to a model that is actually on disk', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openDictation(page, stub.baseURL);

      // `whisper-large-v3` is the catalog's first row but is NOT cached;
      // opening onto it would mean opening onto a model the lane cannot run.
      await expect(page.getByRole('button', { name: /^Model:/ })).toContainText(
        'Whisper Large v3 Turbo',
      );
    } finally {
      await stub.close();
    }
  });

  test('a model that is not downloaded says where to get it', async ({ page }) => {
    // Idle, because that is when the caption has something to decide: with a
    // model serving the lane already works and the row reports THAT.
    const stub = await startStub({ engineState: 'stopped', model: null });
    try {
      await openDictation(page, stub.baseURL);
      await page.getByRole('button', { name: /^Model:/ }).click();
      // Anchored on the name+badge run: "Whisper Large v3" is a PREFIX of
      // "Whisper Large v3 Turbo", and Playwright's name match is a substring
      // by default, so a plain name would resolve both rows.
      await page
        .getByRole('menuitem')
        .filter({ hasText: /^Whisper Large v3best quality/ })
        .click();

      // This surface cannot pull an audio model — the catalog reports no
      // size and the pull gate fails closed — so it says so and names the
      // command instead of offering a Start that would fail in the engine.
      await expect(page.getByText(/rapid-mlx pull org\/whisper-large-v3/)).toBeVisible();
      await expect(page.getByRole('button', { name: 'Start', exact: true })).toHaveCount(0);
    } finally {
      await stub.close();
    }
  });

  test.describe('vocabulary', () => {
    test('terms are added, parked and removed', async ({ page }) => {
      const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
      try {
        await openDictation(page, stub.baseURL);

        await page.getByLabel('Add a name').fill('Rapid-MLX');
        await page.getByRole('button', { name: 'Add' }).click();

        await expect(page.getByText('1 of 20 active')).toBeVisible();
        const chip = page.getByRole('button', { name: 'Rapid-MLX', exact: true });
        await expect(chip).toBeVisible();

        // Tapping parks it: still the user's term, just not being sent.
        await chip.click();
        await expect(page.getByText('0 of 20 active')).toBeVisible();

        await page.getByRole('button', { name: 'Remove Rapid-MLX' }).click();
        await expect(chip).toHaveCount(0);
      } finally {
        await stub.close();
      }
    });

    test('a term survives a reload', async ({ page }) => {
      const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
      try {
        await openDictation(page, stub.baseURL);
        await page.getByLabel('Add a name').fill('Kokoro');
        await page.getByRole('button', { name: 'Add' }).click();
        await expect(page.getByText('1 of 20 active')).toBeVisible();

        await openDictation(page, stub.baseURL);
        await expect(page.getByRole('button', { name: 'Kokoro', exact: true })).toBeVisible();
      } finally {
        await stub.close();
      }
    });

    test('the same name is not added twice', async ({ page }) => {
      const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
      try {
        await openDictation(page, stub.baseURL);
        const field = page.getByLabel('Add a name');
        const add = page.getByRole('button', { name: 'Add' });

        await field.fill('Rapid');
        await add.click();
        await field.fill('rapid');
        await add.click();

        // Case-insensitively one hint — sending both spends the budget twice.
        await expect(page.getByText('1 of 20 active')).toBeVisible();
      } finally {
        await stub.close();
      }
    });
  });

  test('recent dictations start empty', async ({ page }) => {
    const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
    try {
      await openDictation(page, stub.baseURL);
      await expect(page.getByText(/Dictations you make will show up here/)).toBeVisible();
    } finally {
      await stub.close();
    }
  });
});

/**
 * The upload must be a WAV, whatever the browser recorded.
 *
 * The engine decodes with libsndfile, which supports neither mp4 (Safari) nor
 * webm (Chrome/Firefox) — the only two containers `MediaRecorder` produces.
 * Sending the native take failed on EVERY browser with `could not decode
 * audio file`, so `Recorder.stop()` transcodes before the upload.
 *
 * This drives the real recorder against a fake `MediaRecorder` fed a real
 * encoded blob, then asserts on what the SERVER received — the only place the
 * bug was observable.
 */
test('a recording is uploaded as WAV, not as the recorded container', async ({ page }) => {
  const stub = await startStub({ engineState: 'ready', model: 'qwen3-4b' });
  try {
    // WebKit cannot grant a microphone headlessly, so `MediaRecorder` and
    // `getUserMedia` are replaced with doubles that emit a REAL webm/mp4-class
    // blob — produced here as an encoded WAV at a non-target rate, so a
    // pass-through would be visibly wrong (44.1 kHz stereo, not 16 kHz mono).
    await page.addInitScript(() => {
      const rate = 44100;
      const samples = rate; // one second
      const bytes = new ArrayBuffer(44 + samples * 2 * 2);
      const view = new DataView(bytes);
      const ascii = (o: number, t: string) => {
        for (let i = 0; i < t.length; i += 1) view.setUint8(o + i, t.charCodeAt(i));
      };
      ascii(0, 'RIFF');
      view.setUint32(4, 36 + samples * 4, true);
      ascii(8, 'WAVE');
      ascii(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 2, true); // stereo
      view.setUint32(24, rate, true); // 44.1 kHz
      view.setUint32(28, rate * 4, true);
      view.setUint16(32, 4, true);
      view.setUint16(34, 16, true);
      ascii(36, 'data');
      view.setUint32(40, samples * 4, true);
      for (let i = 0; i < samples; i += 1) {
        const s = Math.round(Math.sin((2 * Math.PI * 440 * i) / rate) * 16000);
        view.setInt16(44 + i * 4, s, true);
        view.setInt16(46 + i * 4, s, true);
      }
      const source = new Blob([bytes], { type: 'audio/wav' });

      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) },
      });

      class FakeRecorder {
        state = 'inactive';
        mimeType = 'audio/webm';
        ondataavailable: ((event: { data: Blob }) => void) | null = null;
        onstop: (() => void) | null = null;
        start() {
          this.state = 'recording';
        }
        stop() {
          this.state = 'inactive';
          this.ondataavailable?.({ data: source });
          this.onstop?.();
        }
      }
      (FakeRecorder as unknown as { isTypeSupported(t: string): boolean }).isTypeSupported = () =>
        true;
      Object.defineProperty(window, 'MediaRecorder', {
        configurable: true,
        value: FakeRecorder,
      });
    });

    await openAudio(page, stub.baseURL);
    await page.getByRole('radio', { name: 'Speech to Text' }).click();

    await page.getByRole('button', { name: 'Start recording' }).click();
    // Longer than MIN_RECORDING_MS, or the take is discarded as a mis-tap.
    await page.waitForTimeout(400);
    await page.getByRole('button', { name: 'Stop recording' }).click();

    await expect.poll(() => stub.scenario.audioUpload).not.toBeNull();

    // Decode the header the SERVER was sent.
    const header = Buffer.from(stub.scenario.audioUpload!, 'base64');
    expect(header.subarray(0, 4).toString()).toBe('RIFF');
    expect(header.subarray(8, 12).toString()).toBe('WAVE');
    expect(header.readUInt16LE(20)).toBe(1); // PCM
    // Down-mixed and resampled, not passed through: the source was 44.1 kHz
    // stereo, so these prove the transcode ran.
    expect(header.readUInt16LE(22)).toBe(1);
    expect(header.readUInt32LE(24)).toBe(16000);
    expect(header.readUInt16LE(34)).toBe(16);
  } finally {
    await stub.close();
  }
});
