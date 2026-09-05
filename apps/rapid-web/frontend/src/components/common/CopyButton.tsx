import { useState } from 'react';
import { Check, Copy } from 'lucide-react';
import { copyText } from '@/lib/clipboard';
import { Button } from '@/components/ui/button';

/**
 * Copy to clipboard, as an icon.
 *
 * The tick is the only feedback there is, so the label moves with it: an
 * icon-only control still announces "Copy"/"Copied" and still shows a
 * tooltip. Nothing is shown on failure — `copyText` already falls back, and
 * a banner for a copy that did not take is louder than the action was.
 */
export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <Button
      variant="ghost"
      size="icon"
      className="text-muted-foreground size-7 [&_svg:not([class*=size-])]:size-3.5"
      aria-label={copied ? 'Copied' : 'Copy'}
      title={copied ? 'Copied' : 'Copy'}
      onClick={() => {
        void copyText(text).then((ok) => {
          if (!ok) return;
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      {copied ? <Check className="text-success" /> : <Copy />}
    </Button>
  );
}
