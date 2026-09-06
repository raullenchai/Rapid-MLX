import { loadModel } from '@/api/models';
import type { StatusResponse } from '@/api/types';
import { useStore } from '@/state/store';

/**
 * Switch the loaded model AND adopt what the server says it is now doing.
 *
 * The adoption is not optional, and it is why this is not just `loadModel`.
 * `selectAlias` nulls `status` so the band stops describing the PREVIOUS
 * model, and the next status poll is up to 15 s away — in between, readiness
 * has no serving state for the new alias and resolves to `needsStart`, so the
 * surface offers a Start button for a model that is already starting. That
 * was user-visible from the composer picker, the one call site that had not
 * hand-rolled this.
 *
 * Not in `api/models.ts`: everything there is pure wire code, and reaching
 * into the store from it inverts the layering.
 */
export async function startModel(alias: string): Promise<void> {
  const result = await loadModel(alias);
  useStore.getState().setStatus(
    {
      state: result.state as StatusResponse['state'],
      model: result.model,
      port: null,
      detail: null,
      can_switch: true,
    },
    false,
  );
}
