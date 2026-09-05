import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';
import './styles/tokens.css';
import './styles/base.css';
// Last: preflight must not undo base.css's overrides.
import './styles/tailwind.css';

const host = document.getElementById('root');
if (!host) throw new Error('missing #root');

createRoot(host).render(
  <StrictMode>
    {/* Wraps the whole app, including the boot sequence. Reading
        localStorage can THROW in Safari private browsing, and a throw there
        happens before anything renders — so without this the failure mode is
        a white screen rather than a usable, non-persisting chat. */}
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
);
