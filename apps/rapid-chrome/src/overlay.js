// Runs in an isolated world. The page cannot call the stop binding or inspect the shadow tree.
export function overlay(show, binding = '') {
  const key = '__rapidOverlay';
  globalThis[key]?.remove();
  if (!show) { delete globalThis[key]; return; }
  const host = document.createElement('div');
  host.style.cssText = 'all:initial!important;position:fixed!important;inset:0!important;z-index:2147483647!important;pointer-events:none!important';
  const shadow = host.attachShadow({mode:'closed'});
  const style = document.createElement('style');
  style.textContent = ':host{pointer-events:none}.border{position:fixed;inset:2px;border:2px solid #6587ff;box-shadow:inset 0 0 12px #6587ff66;border-radius:8px;pointer-events:none}.pill{position:fixed;right:16px;top:12px;padding:7px 10px;background:#172242;color:white;font:13px system-ui;border-radius:20px;pointer-events:auto;box-shadow:0 2px 12px #0005}button{margin-left:10px;border:0;border-radius:12px;padding:4px 10px;cursor:pointer;font:inherit}';
  const border = document.createElement('div'); border.className = 'border';
  const pill = document.createElement('div'); pill.className = 'pill'; pill.textContent = 'Rapid is working';
  const button = document.createElement('button'); button.textContent = 'Stop';
  button.onclick = () => { if (binding) globalThis[binding]('stop'); else chrome.runtime.sendMessage({type:'stop'}); };
  pill.append(button); shadow.append(style,border,pill); document.documentElement.append(host);
  globalThis[key] = host;
}
