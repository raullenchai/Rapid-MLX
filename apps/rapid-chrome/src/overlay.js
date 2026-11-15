// Runs in an isolated world; untrusted page text never becomes markup.
export function overlay(show, binding = '', labels = ['Rapid is working','Stop','Take over']) {
  const key = '__rapidOverlay';
  globalThis[key]?.remove();
  if (!show) { delete globalThis[key]; return; }
  const host = document.createElement('div');
  host.style.cssText = 'all:initial!important;position:fixed!important;inset:0!important;z-index:2147483647!important;pointer-events:none!important';
  const shadow = host.attachShadow({mode:'closed'});
  const style = document.createElement('style');
  style.textContent = ':host{pointer-events:none}.frame{position:fixed;inset:2px;border:2px solid #819aff99;box-shadow:inset 0 0 24px #708bff33;border-radius:12px;pointer-events:none;animation:breath 3s ease-in-out infinite}.pill{position:fixed;left:50%;transform:translateX(-50%);top:12px;display:flex;align-items:center;gap:8px;max-width:calc(100vw - 24px);padding:8px 12px;background:#242936;color:#f4f6ff;font:13px/1.4 system-ui,sans-serif;border:1px solid #7080a8;border-radius:24px;pointer-events:auto;box-shadow:0 4px 20px #0003;white-space:nowrap}.pill span{overflow:hidden;text-overflow:ellipsis}button{border:0;background:transparent;color:#d3ddff;padding:4px;font:inherit;cursor:pointer}button:focus-visible{outline:2px solid #c4d1ff;outline-offset:2px;border-radius:4px}.pointer{position:fixed;width:14px;height:14px;left:-7px;top:-7px;border-radius:50%;background:#92aaff;box-shadow:0 0 0 6px #829aff33;opacity:0;pointer-events:none;transition:transform .3s ease-out,opacity .2s}@keyframes breath{50%{opacity:.55}}@media(prefers-reduced-motion:reduce){.frame{animation:none}.pointer{transition:none}}';
  const frame = document.createElement('div'); frame.className = 'frame';
  const pointer = document.createElement('div'); pointer.className = 'pointer';
  const pill = document.createElement('div'); pill.className = 'pill'; pill.setAttribute('role','region'); pill.setAttribute('aria-label',labels[0]);
  const title = document.createElement('span'); title.textContent = labels[0]; pill.append(title);
  for (const [index,action] of [[1,'stop'],[2,'take-over']]) {
    const button = document.createElement('button'); button.textContent = labels[index];
    button.onclick = () => { if (binding) globalThis[binding](action); else void chrome.runtime.sendMessage({type:action}); };
    pill.append(button);
  }
  let lastHeartbeat = Date.now();
  // A detached debugger cannot remove an isolated overlay. Expire it unless
  // the worker acknowledges ownership, including after an unexpected detach.
  const heartbeat = setInterval(() => {
    if (Date.now() - lastHeartbeat > 3500) { host.remove(); delete globalThis[key]; }
    else if (binding) globalThis[binding]('heartbeat');
    else void chrome.runtime.sendMessage({type:'overlay-heartbeat'}).then(owned => { if (owned) lastHeartbeat = Date.now(); }).catch(() => {});
  },1000);
  let pointerTimer;
  const remove = host.remove.bind(host);
  host.remove = () => { clearInterval(heartbeat); clearTimeout(pointerTimer); remove(); };
  host.heartbeat = () => { lastHeartbeat = Date.now(); };
  host.point = (x,y) => { pointer.style.transform = `translate(${x}px,${y}px)`; pointer.style.opacity = '1'; clearTimeout(pointerTimer); pointerTimer = setTimeout(() => { pointer.style.opacity = '0'; },1000); };
  shadow.append(style,frame,pointer,pill); document.documentElement.append(host);
  globalThis[key] = host;
}
export function overlayPointer(x,y) { globalThis.__rapidOverlay?.point(x,y); }
