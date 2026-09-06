/**
 * Copy to clipboard, with a fallback for a non-secure context.
 *
 * `navigator.clipboard` is undefined outside a secure context, and a plain
 * HTTP tunnel or a bare `--host 0.0.0.0` bind is not one — this is the normal
 * deployment, not a legacy-browser concern.
 */
export async function copyText(text: string): Promise<boolean> {
  if (window.isSecureContext && navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Permission denied, or the document lost focus mid-write. Fall through.
    }
  }

  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  const field = document.createElement('textarea');
  field.value = text;
  // Off-screen rather than hidden: `display: none` and `visibility: hidden`
  // both make the element unselectable, and selection is what execCommand
  // copies.
  field.style.position = 'fixed';
  field.style.top = '-9999px';
  field.setAttribute('readonly', '');
  field.setAttribute('aria-hidden', 'true');

  document.body.appendChild(field);
  try {
    field.select();
    return document.execCommand('copy');
  } catch {
    return false;
  } finally {
    field.remove();
  }
}
