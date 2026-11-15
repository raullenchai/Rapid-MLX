import { t } from './i18n.js';
import { watchEffects, collectEffects, effectReport } from './effects.js';
import { detectChallenge, informationTask } from './challenge.js';
import { outlineAX } from './ax.js';
import { safeURL, classifyTarget, approvalTitleKey, needsSiteApproval, additionKey, NavigationPolicy, sensitivePage } from './safety.js';
import { delay } from './agent.js';
import { overlay, overlayPointer } from './overlay.js';

// Executed on the approved node, never on a replacement hit-test node.
export function sameClickControl(other, x, y) {
  if (!other) return false;
  const selector = 'button,a,input,[role="button"],[role="option"],label';
  const control = node => {
    const c = node.closest(selector);
    return c?.tagName === 'LABEL' ? (c.control || c) : c;
  };
  if (this === other) return true;
  const a = control(this), b = control(other);
  if (a && a === b) return true;
  const r = this.getBoundingClientRect();
  if (x < r.left || x > r.right || y < r.top || y > r.bottom) return false;
  // A form alone is too broad: Buy Now and Add to cart must stay distinct.
  const groupSelector = '.a-button,[data-button],[role="group"]';
  const group = this.closest(groupSelector);
  if (!group || group !== other.closest(groupSelector)) return false;
  const controls = [...group.querySelectorAll('button,a,input,[role="button"]')];
  const roots = controls.filter(c => !controls.some(parent => parent !== c && parent.contains(c)));
  return roots.length === 1 && (!a || a === roots[0]) && (!b || b === roots[0]);
}
export function canActivate() {
  const r = this.getBoundingClientRect(), style = getComputedStyle(this);
  return this.isConnected && !this.disabled && !this.matches(':disabled') &&
    !this.closest('[inert],[aria-disabled="true"]') && style.display !== 'none' &&
    style.visibility !== 'hidden' && style.visibility !== 'collapse' &&
    r.width > 0 && r.height > 0;
}

// Fixed browser code, executed only on the resolved and gated control. Native
// selects/options and segmented date inputs cannot be edited with insertText.
export function fillNative(text, choose) {
  let e = this;
  if (text === undefined && e.tagName !== 'OPTION') return {handled:false};
  if (e.tagName === 'OPTION' && text === undefined) text = e.value;
  if (e.tagName === 'OPTION') e = e.closest('select');
  const host = e?.getRootNode?.()?.host;
  if (host?.tagName === 'INPUT' && ['date','time','datetime-local'].includes(host.type)) e = host;
  if (!e || !(e.tagName === 'SELECT' || (e.tagName === 'INPUT' && ['date','time','datetime-local'].includes(e.type)))) return {handled:false};
  if (!e.isConnected || !e.getClientRects().length) return {handled:true,error:'hidden_control: use a visible native control'};
  if (e.disabled || e.readOnly || e.closest('[inert],[aria-disabled="true"]')) return {handled:true,error:'disabled_control: choose an enabled editable control'};
  let wanted = text;
  if (e.tagName === 'SELECT') {
    const normalize = s => String(s).normalize('NFKC').trim().toLocaleLowerCase();
    const query = normalize(choose || text);
    const options = [...e.options].filter(o => !o.disabled && !o.parentElement?.disabled);
    let matches = options.filter(o => normalize(o.value) === query || normalize(o.label) === query);
    if (!matches.length) matches = options.filter(o => normalize(o.label).includes(query));
    if (!query || matches.length !== 1) return {handled:true,error:'option_not_unique: use one exact enabled option label or value'};
    wanted = matches[0].value;
  } else {
    const format = e.type === 'date' ? /^\d{4}-\d{2}-\d{2}$/ : e.type === 'time' ? /^\d{2}:\d{2}(?::\d{2})?$/ : /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$/;
    if (!format.test(text)) return {handled:true,error:`invalid_value: ${e.type} requires ${e.type === 'date' ? 'YYYY-MM-DD' : e.type === 'time' ? 'HH:mm (24-hour)' : 'YYYY-MM-DDTHH:mm'}; fill the whole field, not a segment`};
  }
  const proto = e.tagName === 'SELECT' ? e.ownerDocument.defaultView.HTMLSelectElement.prototype : e.ownerDocument.defaultView.HTMLInputElement.prototype;
  const previous = e.value;
  Object.getOwnPropertyDescriptor(proto,'value').set.call(e,wanted);
  if (e.value !== wanted) {
    Object.getOwnPropertyDescriptor(proto,'value').set.call(e,previous);
    return {handled:true,error:'invalid_value: browser rejected the value; use a valid date/time or enabled option'};
  }
  e.dispatchEvent(new Event('input',{bubbles:true}));
  e.dispatchEvent(new Event('change',{bubbles:true}));
  return {handled:true,value:e.value};
}

export function clickRect() {
  const r = this.getBoundingClientRect();
  const rect = {left:Math.max(0,r.left),top:Math.max(0,r.top),
    right:Math.min(innerWidth,r.right),bottom:Math.min(innerHeight,r.bottom)};
  let view = this.ownerDocument?.defaultView;
  while (view && view !== view.top) {
    const frame = view.frameElement;
    if (!frame) throw new Error('Cannot map cross-origin frame coordinates');
    const f = frame.getBoundingClientRect();
    const x = f.left+frame.clientLeft, y = f.top+frame.clientTop;
    rect.left = Math.max(0,rect.left+x); rect.top = Math.max(0,rect.top+y);
    rect.right = Math.min(view.parent.innerWidth,rect.right+x);
    rect.bottom = Math.min(view.parent.innerHeight,rect.bottom+y);
    view = frame.ownerDocument.defaultView;
  }
  return rect;
}

// Wait for a short quiet period, capped even on pages with continuous updates.
export function mutationQuiet() {
  return new Promise(resolve => {
    let quiet, cap;
    const finish = () => { clearTimeout(quiet); clearTimeout(cap); observer.disconnect(); resolve(true); };
    const reset = () => { clearTimeout(quiet); quiet = setTimeout(finish,150); };
    const observer = new MutationObserver(reset);
    observer.observe(document,{subtree:true,childList:true,characterData:true,attributes:true});
    cap = setTimeout(finish,1000); reset();
  });
}

export class ChromeAdapter {
  constructor(api = chrome) { this.api = api; }
  send(id, method, params = {}) { return this.api.debugger.sendCommand({tabId:id}, method, params); }
  attach(id) { return this.api.debugger.attach({tabId:id}, '1.3'); }
  detach(id) { return this.api.debugger.detach({tabId:id}); }
  get(id) { return this.api.tabs.get(id); }
  create() { return this.api.tabs.create({url:'about:blank'}); }
  async group(id) {
    const groupId = await this.api.tabs.group({tabIds:[id]});
    await this.api.tabGroups.update(groupId, {title:'Rapid',color:'blue'});
    return groupId;
  }
  inject(id, show, labels) { return this.api.scripting.executeScript({target:{tabId:id},func:overlay,args:[show,'',labels]}); }
}

// All browser operations pass through the adapter; no model-provided JS is evaluated.
export class BrowserHand {
  constructor(adapter, {settings, approve, emit = () => {}, stop = () => {}, takeover = () => {}, challenge = async info => { throw new Error(`needs_human: ${info.message}`); }}) {
    this.a = adapter; this.settings = settings; this.approve = approve; this.emit = emit; this.stop = stop; this.pauseForTakeover = takeover; this.challenge = challenge;
    this.tabs = new Map(); this.active = null; this.nextRef = 1; this.sites = new Set(); this.signal = null;
    this.navigation = new NavigationPolicy(); this.additions = new Set(); this.denials = new Set(); this.notices = []; this.pendingSites = new Map();
  }
  check() { this.signal?.throwIfAborted(); if (this.paused) throw new Error('paused: debugger detached by user'); }
  async send(id, method, params) {
    this.check();
    if (!this.tabs.get(id)?.attached) throw new Error('tab_taken_over: user owns this tab now');
    const value = await this.a.send(id, method, params);
    this.check();
    if (method === 'Page.navigate' && value.errorText) {
      await this.challenge(detectChallenge({url:params.url,error:`navigation_failed: ${value.errorText}`}));
      await this.executeCore('read_page',{tabId:id},this.signal);
      return value;
    }
    return value;
  }
  invalidate(id) { const t = this.tabs.get(id); if (t) { t.refs.clear(); t.epoch++; } }
  async site(url, action = 'Navigate') {
    const u = safeURL(url, this.settings.base);
    if (sensitivePage(u.href)) throw new Error('blocked_sensitive_page: complete this page yourself');
    if (needsSiteApproval(u.href, this.sites)) {
      if (!this.pendingSites.has(u.origin)) {
        const request = (async () => {
          if (!await this.approve({action:action === 'Open tab' ? `Open ${u.hostname} in a new tab?` : `Go to ${u.hostname}?`,target:u.hostname,site:u.origin})) throw new Error(`denied: Go to ${u.hostname}?`);
          this.check(); this.sites.add(u.origin);
        })();
        this.pendingSites.set(u.origin,request);
        request.finally(() => this.pendingSites.delete(u.origin)).catch(() => {});
      }
      await this.pendingSites.get(u.origin);
    }
    return u.href;
  }
  async own(id, {created = false} = {}) {
    const tab = await this.a.get(id);
    if (!created && !(tab.url?.startsWith('chrome-error://') && this.tabs.has(id))) safeURL(tab.url, this.settings.base);
    let t = this.tabs.get(id);
    if (!t) {
      t = {attached:false,groupId:tab.groupId,created,bootstrapping:created,refs:new Map(),epoch:0,binding:`rapidStop_${crypto.randomUUID().replaceAll('-','')}`};
      this.tabs.set(id,t);
    }
    this.active = id;
    if (!t.attached) {
      await this.a.attach(id); t.attached = true;
      try {
        await this.send(id,'Page.enable');
        const {frameTree} = await this.send(id,'Page.getFrameTree');
        t.mainFrame = frameTree?.frame.id;
        await this.send(id,'DOM.enable');
        await this.send(id,'Accessibility.enable');
        await this.send(id,'Runtime.enable');
        await this.send(id,'Fetch.enable',{patterns:[{urlPattern:'*',resourceType:'Document',requestStage:'Request'}]});
      } catch (e) { await this.release(id); throw e; }
    }
    if (created) { t.grouping = true; try { t.groupId = await this.a.group(id); } finally { t.grouping = false; } }
    return t;
  }
  async begin(signal, task = '', history = []) {
    this.navigation = new NavigationPolicy([task,...history.filter(m => m.role === 'user').map(m => m.content)].join('\n'));
    this.informationOnly = informationTask([task,...history.filter(m => m.role === 'user').map(m => m.content)].join('\n'));
    this.humanCheckOffered = false;
    this.signal = signal; this.additions.clear(); this.denials.clear(); this.sites.clear(); this.pendingSites.clear();
    if (this.active !== null && this.tabs.has(this.active)) {
      const tab = await this.a.get(this.active);
      const owned = this.tabs.get(this.active);
      if (owned.grantedOrigin && owned.grantedOrigin === new URL(tab.url).origin) this.sites.add(owned.grantedOrigin);
    }
    for (const [id] of this.tabs) { await this.own(id); this.invalidate(id); await this.showOverlay(id,true); }
  }
  async showOverlay(id, show) {
    try { await this.a.inject(id,show,[t('overlayWorking'),t('stop'),t('takeOver')]); return; } catch { /* Host permission is deliberately absent for arbitrary sites. */ }
    try {
      const {frameTree} = await this.a.send(id,'Page.getFrameTree');
      const {executionContextId} = await this.a.send(id,'Page.createIsolatedWorld',{frameId:frameTree.frame.id,worldName:'Rapid isolated overlay'});
      const owned = this.tabs.get(id);
      const binding = owned?.binding;
      if (owned) owned.overlayContext = executionContextId;
      if (!binding) return;
      await this.a.send(id,'Runtime.addBinding',{name:binding,executionContextId});
      await this.a.send(id,'Runtime.evaluate',{contextId:executionContextId,expression:`(${overlay.toString()})(${JSON.stringify(show)},${JSON.stringify(binding)},${JSON.stringify([t('overlayWorking'),t('stop'),t('takeOver')])})`});
    } catch (e) { if (show) this.emit({kind:'notice',text:`Overlay unavailable: ${e.message}. Panel Stop remains available.`}); }
  }
  async release(id) {
    const t = this.tabs.get(id); if (!t) return;
    t.attached = false; t.refs.clear();
    await this.showOverlay(id,false);
    try { await this.a.detach(id); } catch { /* Already detached. */ }
  }
  async finish() { await Promise.allSettled([...this.tabs.keys()].map(id => this.release(id))); }
  async takeover(id, reason) {
    if (!this.tabs.has(id)) return;
    await this.release(id); this.tabs.delete(id);
    if (this.active === id) this.active = this.tabs.keys().next().value ?? null;
    const text = `User takeover: tab ${id} ${reason}. It is no longer available.`;
    this.notices.push(text); this.emit({kind:'notice',text});
  }
  async event(id, method, p) {
    const t = this.tabs.get(id); if (!t?.attached) return;
    if (method === 'Runtime.bindingCalled' && p.name === t.binding && p.payload === 'stop') { this.stop(); return; }
    if (method === 'Runtime.bindingCalled' && p.name === t.binding) {
      if (p.payload === 'take-over') { await this.pauseForTakeover(); return; }
      if (p.payload === 'heartbeat' && !this.paused && !this.signal?.aborted) {
        const contextId = t.overlayContext;
        if (contextId === undefined) return;
        try { await this.a.send(id,'Runtime.evaluate',{contextId,expression:'globalThis.__rapidOverlay?.heartbeat()'}); }
        catch (error) {
          // A navigation can destroy the isolated world while its last heartbeat
          // is in flight. The next page load installs a fresh overlay.
          if (!/Cannot find context with specified id|Execution context was destroyed/i.test(error.message)) throw error;
          if (t.overlayContext === contextId) delete t.overlayContext;
        }
        return;
      }
    }
    if (method === 'Page.frameNavigated') {
      if (!p.frame.parentId) {
        delete t.overlayContext;
        t.mainFrame = p.frame.id; this.invalidate(id);
        if (p.frame.url?.startsWith('chrome-error://')) { t.loadError = p.frame.url; return; }
        delete t.loadError;
        if (t.bootstrapping && p.frame.url === 'about:blank') return;
        try { await this.site(p.frame.url); } catch { await this.takeover(id,'navigated to a blocked URL'); return; }
      }
    }
    if (method === 'Page.navigatedWithinDocument' && (!p.frameId || p.frameId === t.mainFrame)) this.invalidate(id);
    if (method === 'Page.frameRequestedNavigation' && p.frameId === t.mainFrame &&
        t.agentInputUntil > Date.now() && (['anchorClick','formSubmissionGet','formSubmissionPost'].includes(p.reason) || (p.reason === 'scriptInitiated' && t.dispatchingInput))) {
      t.agentNavigation = p.url;
      t.agentInputUntil = 0;
    }
    if (method === 'Page.loadEventFired' && this.signal && !this.signal.aborted) await this.showOverlay(id,true);
    if (method === 'Fetch.requestPaused') {
      try {
        if (!this.signal || this.signal.aborted) throw new Error('Task stopped');
        // Every main-frame request, including redirects and script navigation,
        // must pass site policy. Passive subframe requests remain independent.
        if (p.frameId === t.mainFrame) {
          t.agentNavigation = null;
          t.navigationGate = this.site(p.request.url);
          try { await t.navigationGate; } finally { t.navigationGate = null; }
        }
        this.check();
        await this.send(id,'Fetch.continueRequest',{requestId:p.requestId});
      } catch (e) {
        if (/^denied:/.test(e.message)) t.navigationDenied = e.message;
        try { await this.a.send(id,'Fetch.failRequest',{requestId:p.requestId,errorReason:'BlockedByClient'}); } catch {}
        if (!/^denied:/.test(e.message)) this.notices.push(`Navigation blocked: ${e.message}`);
      }
    }
  }
  async current(args) {
    const id = args.tabId ?? this.active;
    if (!this.tabs.has(id)) throw new Error('unowned_tab: use open_tab or ask the user to share a tab');
    if (!this.tabs.get(id).attached) throw new Error('tab_detached: ask the user to keep working');
    try { await this.tabs.get(id).navigationGate; } catch { /* surfaced below */ }
    let tab = await this.a.get(id);
    // Observing a page-initiated redirect is not an agent navigation.
    if (!tab.url?.startsWith('chrome-error://')) {
      for (let attempt=0;attempt<3;attempt++) {
        await this.site(tab.url);
        const fresh = await this.a.get(id);
        if (fresh.url === tab.url) break;
        tab = fresh;
        if (attempt === 2) throw new Error('stale_ref: page kept navigating during site approval');
      }
      await this.guardSensitivePage(id);
    }
    const denied = this.tabs.get(id).navigationDenied;
    if (denied) { delete this.tabs.get(id).navigationDenied; throw new Error(denied); }
    this.active = id; return {id,tab,t:this.tabs.get(id)};
  }
  async guardSensitivePage(id) {
    const {frameTree} = await this.send(id,'Page.getFrameTree');
    const frames = frameTree ? [frameTree] : [null];
    while (frames.length) {
      const frame = frames.shift(); frames.push(...(frame?.childFrames || []));
      const world = frame ? await this.send(id,'Page.createIsolatedWorld',{frameId:frame.frame.id,worldName:'Rapid sensitive-page inspection'}) : {};
      const r = await this.send(id,'Runtime.evaluate',{...(world.executionContextId ? {contextId:world.executionContextId} : {}),expression:`(${sensitiveDocument.toString()})()`,returnByValue:true});
      if (r.exceptionDetails || r.result?.value === true) throw new Error('blocked_sensitive_page: sensitive fields present or inspection unavailable; complete this page yourself');
    }
  }
  async focused(id) {
    const {frameTree} = await this.send(id,'Page.getFrameTree');
    let frameId = frameTree?.frame.id;
    const visited = new Set();
    for (let depth=0;depth<16;depth++) {
      if (frameId && visited.has(frameId)) break;
      if (frameId) visited.add(frameId);
      const world = frameId ? await this.send(id,'Page.createIsolatedWorld',{frameId,worldName:'Rapid focus inspection'}) : {};
      const r = await this.send(id,'Runtime.evaluate',{...(world.executionContextId ? {contextId:world.executionContextId} : {}),expression:'(() => { let e = document.activeElement; while (e?.shadowRoot?.activeElement) e = e.shadowRoot.activeElement; return e; })()',returnByValue:false});
      if (!r.result?.objectId) break;
      const {node} = await this.send(id,'DOM.describeNode',{objectId:r.result.objectId,depth:1});
      if (!node?.backendNodeId) break;
      if (['IFRAME','FRAME'].includes(node.nodeName)) {
        frameId = node.contentDocument?.frameId || node.frameId;
        if (!frameId) break;
        continue;
      }
      if (node.shadowRoots?.some(root => root.shadowRootType === 'closed')) break;
      return {objectId:r.result.objectId,backendDOMNodeId:node.backendNodeId,epoch:this.tabs.get(id).epoch,name:'Focused element'};
    }
    throw new Error('unresolved_focus: keyboard action requires a resolved control; use a fresh ref or take over');
  }
  async verifyFocus(id, target) {
    await this.verifyApproval(id,target);
    const actual = await this.focused(id);
    if (actual.backendDOMNodeId !== target.backendDOMNodeId || actual.epoch !== target.epoch)
      throw new Error('focus_changed: input recipient changed; read_page before trying again');
    const info = await this.inspect(id,actual);
    if (classifyTarget(info).sensitive) throw new Error('blocked_sensitive_field: focused field is sensitive');
    const latest = await this.focused(id);
    if (latest.backendDOMNodeId !== actual.backendDOMNodeId || latest.epoch !== actual.epoch)
      throw new Error('focus_changed: input recipient changed during inspection');
  }
  async verifyApproval(id,target) {
    await this.current({tabId:id});
    const snapshot = target.approvedInfo || target.inspectedInfo;
    if (snapshot && (target.epoch !== this.tabs.get(id)?.epoch || JSON.stringify(snapshot) !== JSON.stringify(await this.inspect(id,target))))
      throw new Error('stale_ref: form values or target changed after approval; request a fresh approval');
  }
  async resolve(id, ref) {
    // Accept the outline's display brackets, but only for a current exact ref.
    ref = typeof ref === 'string' ? ref.replace(/^\[(e\d+)\]$/, '$1') : ref;
    const t = this.tabs.get(id), target = t?.refs.get(ref);
    if (!target) throw new Error('stale_ref: call read_page for fresh refs');
    const epoch = t.epoch;
    try {
      const {object} = await this.send(id,'DOM.resolveNode',{backendNodeId:target.backendDOMNodeId});
      if (!object.objectId || epoch !== t.epoch) throw new Error('stale');
      return {...target,objectId:object.objectId,epoch};
    } catch { throw new Error('stale_ref: call read_page for fresh refs'); }
  }
  async call(id, objectId, fn, args = []) {
    const r = await this.send(id,'Runtime.callFunctionOn',{objectId,functionDeclaration:fn.toString(),arguments:args.map(value => ({value})),returnByValue:true});
    if (r.exceptionDetails) throw new Error('DOM inspection failed');
    return r.result.value;
  }
  async inspect(id, target) {
    return this.call(id,target.objectId,function () {
      const e = this;
      const form = e.form || e.closest('form');
      const label = [e.getAttribute('aria-label'),e.labels ? [...e.labels].map(l => l.textContent).join(' ') : '', e.textContent].filter(Boolean).join(' ').slice(0,1000);
      const accessibleName = node => node ? (node.getAttribute('aria-label') || (node.getAttribute('aria-labelledby') || '').split(/\s+/).map(id => node.ownerDocument.getElementById(id)?.textContent || '').join(' ').trim() || node.innerText || node.value || '') : '';
      const submitters = form ? [...form.elements].filter(el => ['submit','image'].includes(el.type)) : [];
      const recipients = form ? [...form.elements].filter(el => /^(?:to|cc|bcc)$/i.test(el.name || '')).map(el => `${el.name}: ${el.value}`).join('; ').slice(0,500) : '';
      const fields = form ? [...form.elements].filter(el => ['INPUT','SELECT','TEXTAREA'].includes(el.tagName)) : [];
      if (fields.length > 100) throw new Error('Form too large to bind');
      const values = fields.map(el => ({name:el.name || el.id || '',type:el.type || '',value:el.value,checked:el.checked,disabled:el.disabled,selected:el.tagName === 'SELECT' ? [...el.selectedOptions].map(o => o.value) : undefined}));
      if (JSON.stringify(values).length > 16000) throw new Error('Form too large to bind');
      const sensitiveField = el => el.type === 'password' || /^(?:cc-|current-password|new-password)/i.test(el.autocomplete || '') || /password|cvv|cvc|card.?number|security.?code/i.test(el.name+' '+el.id);
      const sensitiveForm = fields.some(sensitiveField);
      const paymentForm = fields.some(el => /(?:^|\s)cc-|credit.?card|card.?number|card.?holder|security.?code|\bcvv\b|\bcvc\b|routing|bank.?account|iban|信用卡|银行卡|銀行卡|卡号|卡號/i.test([el.name,el.id,el.autocomplete].join(' ')));
      const details = [...values].sort((a,b) => Number(/amount|currency|recipient|quantity|qty|address|destination|total|price|^to$/i.test(b.name))-Number(/amount|currency|recipient|quantity|qty|address|destination|total|price|^to$/i.test(a.name))).map(v => `${v.name || v.type}: ${(v.type === 'hidden' && !/amount|currency|recipient|quantity|qty|address|destination|total|price|^to$/i.test(v.name)) || /password|cc-|cvv|cvc|card|token|secret|account|routing|iban/i.test(v.name+' '+v.type) || fields.some(el => (el.name || el.id || '') === v.name && sensitiveField(el)) ? '[masked]' : String(v.selected?.length > 1 ? v.selected.join(', ') : v.value).slice(0,160)}${['checkbox','radio'].includes(v.type) ? v.checked ? ' [checked]' : ' [unchecked]' : ''}${v.disabled ? ' [disabled]' : ''}`).join('; ').slice(0,3000);
      const card = e.closest('[data-asin]:not([data-asin=""]),[data-product-id],article,[role="listitem"],li');
      const product = card?.getAttribute('data-asin') || card?.getAttribute('data-product-id') || card?.querySelector('h1,h2,h3,[role="heading"]')?.textContent || card?.querySelector('a[href]')?.getAttribute('href') || e.closest('[role="dialog"],dialog')?.querySelector('h1,h2,h3,[role="heading"]')?.textContent || '';
      return {values,details,sensitiveForm,paymentForm,longName:accessibleName(e).length > 1000,expanded:e.getAttribute('aria-expanded'),recipients,product:product.trim().slice(0,400),role:e.getAttribute('role') || (e.tagName === 'BUTTON' ? 'button' : ['checkbox','radio'].includes(e.type) ? e.type : ''),formName:accessibleName(form),submitter:submitters.map(accessibleName).join(' '),method:form?.method || '',name:(accessibleName(e) || label).slice(0,1000),type:e.type || e.getAttribute('type') || '',autocomplete:e.getAttribute('autocomplete') || '',id:[e.id,e.getAttribute('name')].join(' '),form:form?.innerText.slice(0,2000) || '',href:e.closest('a')?.href || '',action:form?.action || '',connected:e.isConnected};
    });
  }
  async gateTarget(id, target, action, {fill = false, submit = false, always = false} = {}) {
    const info = await this.inspect(id,target);
    if (!info.connected) throw new Error('stale_ref: call read_page');
    const classification = classifyTarget({...info,role:target.role || info.role,name:info.name || target.name});
    if (fill && (classification.sensitive || classifyTarget({name:target.name}).sensitive)) throw new Error('blocked_sensitive_field: ask the user to enter passwords or payment details');
    const tab = await this.a.get(id);
    target.additionKey = (!fill || submit) ? additionKey(tab.url,{...info,name:info.name || target.name}) : null;
    const repeat = target.additionKey && this.additions.has(target.additionKey);
    const approvalName = submit && info.submitter ? info.submitter : info.name || target.name || 'Focused element';
    if ((!fill || submit) && classification.financial) {
      const message = `Your turn: review and press ${approvalName} yourself${info.details ? `. Form values: ${info.details}` : ''}`;
      const error = new Error(`financial_handover: ${message}`);
      error.handover = {kind:'payment',message,reason:message};
      throw error;
    }
    if (repeat || always || ((!fill || submit) && classification.consequential)) {
      // Submission via Enter, fill+submit, and a submit button represents the
      // same effect. Keys exclude refs, field values, and input method so a
      // refreshed outline or edited form cannot re-request a denied effect.
      const effectName = (submit || info.type === 'submit') && info.submitter ? info.submitter : approvalName;
      const denialKey = JSON.stringify([new URL(tab.url).origin,info.action || new URL(tab.url).pathname,effectName.trim().toLocaleLowerCase()]);
      if (this.denials.has(denialKey)) throw new Error(`denied: '${effectName}' was already denied in this task; approval will not be requested again`);
      const titleKey = approvalTitleKey(approvalName);
      if (!await this.approve({titleKey:titleKey === 'allowAction' && (submit || info.type === 'submit') ? 'allowSubmit' : titleKey,action:repeat ? 'Add another one? (already added once)' : `${action === 'click' ? 'Click' : action} '${approvalName.slice(0,100)}' on ${new URL(tab.url).hostname}?${info.details ? ` Form values: ${info.details}` : info.recipients ? ` Recipients: ${info.recipients}` : ''}`,target:approvalName,site:new URL(tab.url).origin})) {
        this.denials.add(denialKey);
        throw new Error(`denied: ${action} '${effectName}'`);
      }
      this.check();
      target.approvedInfo = info;
      if (target.epoch !== this.tabs.get(id)?.epoch || JSON.stringify(info) !== JSON.stringify(await this.inspect(id,target))) throw new Error('stale_ref: target changed during approval; call read_page');
    }
    target.inspectedInfo = info;
    // Link/form targets can address an iframe; gate the actual main-frame request.
    return info;
  }
  async click(id,target) {
    if (target.role === 'option') {
      const native = await this.call(id,target.objectId,fillNative,[]);
      if (native?.error) throw new Error(native.error);
      if (native?.handled) return 'native option selection';
    }
    await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId});
    // Use client rects directly: CDP input/hit tests and CSSOM client rects
    // share viewport coordinates. Subtracting pageY again rejects scrolled nodes.
    const {left,top,right,bottom} = await this.call(id,target.objectId,clickRect);
    if (right <= left || bottom <= top) throw new Error('offscreen_target: no visible area; inspect dialogs or choose another control');
    const x = Math.round((left+right)/2), y = Math.round((top+bottom)/2);
    try {
      const contextId = this.tabs.get(id)?.overlayContext;
      if (contextId) await this.a.send(id,'Runtime.evaluate',{contextId,expression:`(${overlayPointer.toString()})(${x},${y})`});
      else if (this.a.api?.scripting) await this.a.api.scripting.executeScript({target:{tabId:id},func:overlayPointer,args:[x,y]});
    } catch { /* A visual hint never blocks the browser action. */ }
    let accepted = false;
    try {
      const hit = await this.send(id,'DOM.getNodeForLocation',{x,y,includeUserAgentShadowDOM:true});
      accepted = hit.backendNodeId === target.backendDOMNodeId;
      if (!accepted) {
        const {object} = await this.send(id,'DOM.resolveNode',{backendNodeId:hit.backendNodeId});
        const r = await this.send(id,'Runtime.callFunctionOn',{objectId:target.objectId,functionDeclaration:sameClickControl.toString(),arguments:[{objectId:object.objectId},{value:x},{value:y}],returnByValue:true});
        accepted = !r.exceptionDetails && r.result.value === true;
      }
    } catch (error) {
      this.check();
      if (!/No node found at given location/.test(error.message)) throw error;
    }
    if (!accepted && await this.call(id,target.objectId,canActivate) !== true)
      throw new Error('obscured_target: hit test failed and approved target is not visible/enabled');
    await this.verifyApproval(id,target);
    if (target.epoch !== this.tabs.get(id)?.epoch) throw new Error('stale_ref: call read_page');
    this.tabs.get(id).agentInputUntil = Date.now()+1500;
    const t = this.tabs.get(id);
    t.dispatchingInput = true;
    if (target.additionKey) this.additions.add(target.additionKey);
    try {
      if (accepted) {
        await this.send(id,'Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});
        await this.send(id,'Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:1});
      } else {
        try { await this.send(id,'DOM.focus',{backendNodeId:target.backendDOMNodeId}); }
        catch (error) {
          this.check();
          // ARIA options and other clickable containers need not be focusable.
          // Still activate only the same approved, visible, enabled node.
          if (!/Element is not focusable/i.test(error.message)) throw error;
        }
        if (target.epoch !== t.epoch || await this.call(id,target.objectId,canActivate) !== true)
          throw new Error('stale_ref: approved target changed before DOM activation');
        await this.verifyApproval(id,target);
        await this.call(id,target.objectId,function () { this.click(); });
      }
    } finally { t.dispatchingInput = false; }
    await this.navigationResult(id);
    return accepted ? 'pointer' : 'DOM activation of the approved element (hit test failed; DOM.focus + this.click())';
  }
  async chooseOption(id, label) {
    const normalize = text => String(text).normalize('NFKC').toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
    const terms = normalize(label);
    if (!terms.length) throw new Error('invalid_arguments: choose must contain an option label');
    // Input handlers often populate a listbox asynchronously. Retry only the
    // observation, never the activation; every candidate still passes click gates.
    for (let attempt=0;attempt<5;attempt++) {
      await delay(100,this.signal);
      const {nodes} = await this.send(id,'Accessibility.getFullAXTree');
      const options = nodes.filter(n => !n.ignored && n.role?.value === 'option' && n.backendDOMNodeId &&
        !n.properties?.some(p => ['hidden','disabled'].includes(p.name) && p.value?.value === true));
      const exact = options.filter(n => normalize(n.name?.value).join(' ') === terms.join(' '));
      const matches = exact.length ? exact : options.filter(n => terms.every(t => normalize(n.name?.value).includes(t)));
      if (matches.length > 1) throw new Error('ambiguous_option: multiple options match choose; use the returned outline to select an exact option');
      if (!matches.length) continue;
      const node = matches[0];
      const {object} = await this.send(id,'DOM.resolveNode',{backendNodeId:node.backendDOMNodeId});
      const target = {objectId:object.objectId,backendDOMNodeId:node.backendDOMNodeId,role:'option',name:node.name?.value,epoch:this.tabs.get(id).epoch};
      await this.gateTarget(id,target,'click');
      return this.click(id,target);
    }
    throw new Error('option_not_found: text was filled, but no unique autocomplete option matched choose; inspect the fresh outline before continuing');
  }
  async navigationResult(id) {
    const t = this.tabs.get(id);
    // Give ordered CDP navigation events a chance to reach the request gate.
    await delay(100,this.signal);
    try { await t.navigationGate; } catch { /* rejection is recorded by event */ }
    if (t.navigationDenied) { const error = t.navigationDenied; delete t.navigationDenied; throw new Error(error); }
  }
  async key(id,key) {
    const codes = {Enter:13,Tab:9,Escape:27,Backspace:8,Delete:46,Space:32,ArrowLeft:37,ArrowUp:38,ArrowRight:39,ArrowDown:40,Home:36,End:35,PageUp:33,PageDown:34};
    const p = {key:key === 'Space' ? ' ' : key,code:key === 'Space' ? 'Space' : key,windowsVirtualKeyCode:codes[key]};
    const t = this.tabs.get(id);
    t.dispatchingInput = true;
    if (key === 'Enter') t.agentInputUntil = Date.now()+1500;
    try {
      await this.send(id,'Input.dispatchKeyEvent',{type:'keyDown',...p,...(key === 'Enter' ? {text:'\r'} : key === 'Space' ? {text:' '} : {})});
      await this.send(id,'Input.dispatchKeyEvent',{type:'keyUp',...p});
    } finally { t.dispatchingInput = false; }
    if (key === 'Enter') await this.navigationResult(id);
  }
  async effectWorld(id) {
    const {frameTree} = await this.send(id,'Page.getFrameTree');
    const {executionContextId} = await this.send(id,'Page.createIsolatedWorld',{frameId:frameTree.frame.id,worldName:'Rapid action effects'});
    return executionContextId;
  }
  async effectEval(id, contextId, fn) {
    const params = {contextId,expression:`(${fn.toString()})()`,returnByValue:true};
    // Disconnect the observer even when Stop has aborted guarded operations.
    const r = fn === collectEffects ? await this.a.send(id,'Runtime.evaluate',params) : await this.send(id,'Runtime.evaluate',params);
    if (r.exceptionDetails) throw new Error('Effect observation unavailable');
    return r.result.value;
  }
  async settle(id) {
    try {
      const contextId = await this.effectWorld(id);
      await this.send(id,'Runtime.evaluate',{contextId,expression:`(${mutationQuiet.toString()})()`,awaitPromise:true,returnByValue:true});
    } catch { this.check(); await delay(150,this.signal); }
  }
  async freshRead(args, signal) {
    // Retry observations only. Never replay a mutation after navigation churn.
    for (let attempt = 0; ; attempt++) {
      try { return await this.executeCore('read_page',args,signal); }
      catch (error) {
        this.check();
        if (attempt >= 2 || !/stale_ref/.test(error.message)) throw error;
        await this.settle(args.tabId ?? this.active);
      }
    }
  }
  async execute(name,args,signal) {
    const state = this.tabs.get(args.tabId ?? this.active);
    if (state?.humanCheckHost && ['click','fill','press','scroll'].includes(name))
      throw new Error('human_check: do not interact with the check; use one independent source or ask the user');
    if (name === 'read_page') { this.signal = signal; return this.freshRead(args,signal); }
    if (['wait','scroll','navigate','open_tab'].includes(name)) {
      const result = await this.executeCore(name,args,signal);
      const id = result.tabId ?? args.tabId ?? this.active;
      if (!this.tabs.has(id)) return result;
      if (['navigate','open_tab'].includes(name)) await this.settle(id);
      const observation = await this.freshRead({tabId:id},signal);
      // Navigation can return before commit. Do not present its old URL/title
      // as the final state; the outline is the authoritative fresh observation.
      const meta = await this.meta(id);
      return {...result,...meta,text:`${result.text}. Fresh observation:\n${observation.text}`};
    }
    if (!['click','fill','press'].includes(name)) return this.executeCore(name,args,signal);
    this.signal = signal;
    const {id} = await this.current(args);
    let context, before, samples = [], observation;
    try {
      context = await this.effectWorld(id);
      before = await this.effectEval(id,context,watchEffects);
    } catch { this.check(); }
    let result, actionError;
    try {
      result = await this.executeCore(name,args,signal);
      // Bounded settling window; MutationObserver retains transient notifications.
      await this.settle(id);
    } catch (error) { this.check(); actionError = error; } finally {
      if (context) {
        try { samples = await this.effectEval(id,context,collectEffects); }
        catch { this.check(); } // A committed navigation destroys the old world.
      }
    }
    const meta = await this.meta(id);
    if (before && !samples.length) {
      try {
        context = await this.effectWorld(id);
        const after = await this.effectEval(id,context,watchEffects);
        samples = [after];
        await this.effectEval(id,context,collectEffects);
      } catch { this.check(); }
    }
    const effect = before ? effectReport(before,[...samples,{...(samples.at(-1) || before),...meta}]) : {note:'Effect observation unavailable; verify in the cart before adding again.'};
    if (result?.activation) effect.activation = result.activation;
    try { observation = await this.freshRead({tabId:id},signal); }
    catch (error) { this.check(); observation = {text:`Fresh observation unavailable: ${error.message}. Wait then read_page; do not repeat the action.`}; }
    if (actionError) {
      actionError.effect = effect; actionError.observation = observation; throw actionError;
    }
    return {effect,...result,text:`${name} completed. Fresh observation:\n${observation.text}`};
  }
  async executeCore(name,args,signal) {
    this.signal = signal; this.check();
    if (name === 'list_tabs') {
      const tabs = [];
      for (const id of this.tabs.keys()) {
        const {tab,t} = await this.current({tabId:id});
        tabs.push({tabId:id,title:tab.title,url:tab.url,attached:t.attached});
      }
      return {tabs};
    }
    if (name === 'wait') { await delay(args.ms,signal); return {text:`Waited ${args.ms}ms`,...(this.tabs.has(this.active) ? await this.meta(this.active) : {})}; }
    if (['navigate','open_tab'].includes(name) && args.url) {
      const url = safeURL(args.url,this.settings.base);
      this.navigation.check(url);
      if ([...this.tabs.values()].some(tab => tab.humanCheckHost === url.hostname))
        throw new Error('human_check: leave the challenged site alone; use one independent source or ask the user');
    }
    if (name === 'open_tab') {
      this.navigation.check(safeURL(args.url,this.settings.base));
      const url = await this.site(args.url,'Open tab');
      const tab = await this.a.create();
      this.check(); await this.own(tab.id,{created:true});
      await this.send(tab.id,'Page.navigate',{url});
      this.tabs.get(tab.id).bootstrapping = false;
      await this.showOverlay(tab.id,true);
      return {text:'Opened tab',tabId:tab.id,...await this.meta(tab.id)};
    }
    if (name === 'navigate' && args.url) this.navigation.check(safeURL(args.url,this.settings.base));
    const {id,tab,t} = await this.current(args);
    if (t.humanCheckHost && !['read_page','screenshot'].includes(name) && !(name === 'navigate' && args.url))
      throw new Error('human_check: do not interact with the check; use one independent source or ask the user');
    if (name === 'read_page') {
      t.agentInputUntil = 0;
      const epoch = t.epoch;
      const tree = await this.send(id,'Accessibility.getFullAXTree');
      const nodes = [...tree.nodes];
      // Same-process frames are addressable by frameId. Out-of-process frames
      // may reject this request; their accessible iframe label remains visible.
      const {frameTree} = await this.send(id,'Page.getFrameTree');
      const frames = [...(frameTree?.childFrames || [])];
      const seen = new Set(nodes.map(n => n.backendDOMNodeId).filter(Boolean));
      while (frames.length) {
        const frame = frames.shift(); frames.push(...(frame.childFrames || []));
        try {
          const child = await this.send(id,'Accessibility.getFullAXTree',{frameId:frame.frame.id});
          for (const node of child.nodes) if (!node.backendDOMNodeId || !seen.has(node.backendDOMNodeId)) {
            nodes.push(node); if (node.backendDOMNodeId) seen.add(node.backendDOMNodeId);
          }
        } catch { this.check(); }
      }
      if (epoch !== t.epoch) throw new Error('stale_ref: navigation during read; call read_page');
      let visibleFrames = [], geometry = new Map();
      try {
        const r = await this.send(id,'Runtime.evaluate',{expression:`JSON.stringify([...document.querySelectorAll('iframe')].map(f => { const r=f.getBoundingClientRect(); return {url:f.src,visible:r.width>40 && r.height>40 && r.bottom>0 && r.top<innerHeight && getComputedStyle(f).visibility!=='hidden'}; }))`,returnByValue:true});
        visibleFrames = JSON.parse(r.result.value);
        const snapshot = await this.send(id,'DOMSnapshot.captureSnapshot',{computedStyles:[]});
        const {layoutViewport:v} = await this.send(id,'Page.getLayoutMetrics');
        for (const doc of snapshot.documents || []) {
          if (doc !== snapshot.documents[0]) continue;
          doc.layout.nodeIndex.forEach((nodeIndex,i) => {
            const [x,y,w,h] = doc.layout.bounds[i];
            geometry.set(doc.nodes.backendNodeId[nodeIndex],w>0 && h>0 && y+h>v.pageY && y<v.pageY+v.clientHeight && x+w>v.pageX && x<v.pageX+v.clientWidth);
          });
        }
      } catch { this.check(); }
      const blocked = detectChallenge({url:t.loadError || tab.url,title:tab.title,nodes,frames:visibleFrames});
      if (blocked) {
        delete t.loadError;
        this.invalidate(id);
        if (blocked.kind === 'human_check') {
          t.humanCheckHost = new URL(tab.url).hostname;
          if (t.created && this.informationOnly && !this.humanCheckOffered) {
            this.humanCheckOffered = true;
            return {text:`human_check: ${t.humanCheckHost} requires a human check. Do not interact with, solve, or bypass it. Leave this site alone. You may try one independent source using a site root/search or an observed/user-supplied URL (URL provenance and site approval still apply), or ask the user. If that source is also blocked or insufficient, ask the user.`,human_check:true};
          }
        }
        await this.challenge(blocked);
        this.check();
        // Continue always takes a fresh observation; never reuse pre-pause refs.
        const observation = await this.execute('read_page',args,signal);
        return {...observation,note:'The user pressed Continue after a page check/error. This is a fresh observation; do not solve or bypass checks.'};
      }
      delete t.humanCheckHost;
      // Retain refs only for the same live backend node, name and role. Resolving
      // and inspecting again at action time still binds approval to current DOM.
      const live = new Map(nodes.filter(n => !n.ignored && n.backendDOMNodeId).map(n => [n.backendDOMNodeId,n]));
      const retained = new Map([...t.refs].filter(([,target]) => {
        const n = live.get(target.backendDOMNodeId);
        return n && n.role?.value === target.role && String(n.name?.value || '').replace(/\s+/g,' ').trim() === target.name &&
          !n.properties?.some(p => p.name === 'hidden' && p.value?.value === true);
      }));
      const stableRefs = new Map([...retained].map(([ref,target]) => [target.backendDOMNodeId,ref]));
      const outline = outlineAX(nodes,{url:tab.url,title:tab.title,start:this.nextRef,query:args.query,geometry,stableRefs});
      t.refs = new Map([...retained,...outline.refs]); this.nextRef = outline.next;
      for (const target of outline.refs.values()) if (target.href) this.navigation.observe(target.href);
      const forms = new Map(), nativeControls = [];
      for (const [ref, target] of outline.refs) {
        if (!['button','textbox','searchbox','combobox','Date','DateTime','InputTime','spinbutton'].includes(target.role)) continue;
        try {
          const resolved = await this.resolve(id,ref);
          const form = await this.call(id,resolved.objectId,function () {
            const f = this.form || this.closest('form');
            const type = this.tagName === 'SELECT' ? 'select' : this.type;
            const native = ['select','date','time','datetime-local'].includes(type) ? {type,value:this.value} : null;
            if (!f && !native) return null;
            const recipients = f ? [...f.elements].filter(el => /^(?:to|cc|bcc)$/i.test(el.name || '')).map(el => `${el.name}: ${el.value}`).join('; ').slice(0,500) : '';
            return {index:f ? [...document.forms].indexOf(f) : -1,name:f?.getAttribute('aria-label') || f?.getAttribute('name') || '',method:f?.method || '',native,recipients};
          });
          if (form) {
            if (form.index >= 0) {
              const key = `${form.index}: ${JSON.stringify(form.name)} (${form.method})${form.recipients ? ` Recipients: ${form.recipients}` : ''}`;
              forms.set(key,[...(forms.get(key) || []),`[${ref}]`]);
            }
            if (form.native) nativeControls.push(`Native [${ref}] ${form.native.type} value=${JSON.stringify(form.native.value)}; fill with ${form.native.type === 'select' ? 'exact option label/value' : form.native.type === 'date' ? 'YYYY-MM-DD' : form.native.type === 'time' ? 'HH:mm (24-hour)' : 'YYYY-MM-DDTHH:mm'}`);
          }
        } catch { this.check(); }
      }
      if (epoch !== t.epoch) throw new Error('stale_ref: page changed while reading forms; call read_page again');
      const formText = [...forms].map(([form,refs]) => `Form ${form}: ${refs.join(' ')}`).join('\n');
      return {text:`${formText ? `Form controls (use refs in outline):\n${formText}\n` : ''}${nativeControls.length ? `${nativeControls.join('\n')}\n` : ''}${outline.text}`};
    }
    if (name === 'screenshot') {
      if (!this.settings.vision) return {note:'Vision is off. Use read_page.',...await this.meta(id)};
      const {data} = await this.send(id,'Page.captureScreenshot',{format:'jpeg',quality:65,captureBeyondViewport:false});
      return {text:'Screenshot captured',image:data,...await this.meta(id)};
    }
    let activation;
    if (name === 'navigate') {
      if (args.url) this.navigation.check(safeURL(args.url,this.settings.base));
      this.invalidate(id);
      if (args.url) {
        await this.send(id,'Page.navigate',{url:await this.site(args.url)});
      }
      else if (args.action === 'reload') await this.send(id,'Page.reload');
      else {
        const h = await this.send(id,'Page.getNavigationHistory');
        const entry = h.entries[h.currentIndex + (args.action === 'back' ? -1 : 1)];
        if (!entry) throw new Error('No history entry');
        await this.site(entry.url); await this.send(id,'Page.navigateToHistoryEntry',{entryId:entry.id});
      }
    } else if (name === 'click' || name === 'fill') {
      const target = await this.resolve(id,args.ref);
      await this.gateTarget(id,target,name,{fill:name === 'fill'});
      if (name === 'click') activation = await this.click(id,target);
      else {
        const native = await this.call(id,target.objectId,fillNative,[args.text,args.choose]);
        if (native?.error) throw new Error(native.error);
        if (!native?.handled) {
          await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId});
          await this.send(id,'DOM.focus',{backendNodeId:target.backendDOMNodeId});
          await this.verifyFocus(id,target);
          // CDP editing command is platform independent (unlike Ctrl/Command+A).
          await this.send(id,'Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',commands:['selectAll']});
          await this.send(id,'Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA'});
          await this.verifyFocus(id,target);
          await this.send(id,'Input.insertText',{text:args.text});
          if (args.choose) activation = await this.chooseOption(id,args.choose);
        }
        if (args.submit) {
          // Recheck after input handlers may have changed the form.
          await this.gateTarget(id,target,'Submit form',{fill:true,submit:true});
          const info = await this.inspect(id,target);
          if (classifyTarget(info).sensitive) throw new Error('blocked_sensitive_field');
          if (target.additionKey) this.additions.add(target.additionKey);
          await this.verifyFocus(id,target);
          await this.key(id,'Enter');
        }
      }
    } else if (name === 'press') {
      if (!['Enter','Space','Delete','Backspace'].includes(args.key)) {
        await this.key(id,args.key);
        return {text:`${name} completed`,...await this.meta(id)};
      }
      const target = await this.focused(id);
      // Enter submits the focused form; classify that effect just like fill+submit.
      if (['Enter','Space','Delete','Backspace'].includes(args.key))
        await this.gateTarget(id,target,`Press ${args.key}`,{submit:args.key === 'Enter'});
      if (['Enter','Space'].includes(args.key) && target.additionKey) this.additions.add(target.additionKey);
      if (args.key === 'Enter') t.agentInputUntil = Date.now()+1500;
      await this.verifyFocus(id,target);
      await this.key(id,args.key);
    } else if (name === 'scroll') {
      if (args.ref) { const target = await this.resolve(id,args.ref); await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId}); }
      else await this.send(id,'Input.dispatchMouseEvent',{type:'mouseWheel',x:100,y:200,deltaX:args.direction === 'right' ? 600 : args.direction === 'left' ? -600 : 0,deltaY:args.direction === 'down' ? 600 : args.direction === 'up' ? -600 : 0});
    }
    return {text:`${name} completed`,...(activation ? {activation} : {}),...await this.meta(id)};
  }
  async meta(id) { const {tab} = await this.current({tabId:id}); return {url:tab.url,title:tab.title}; }
}

export function sensitiveDocument() {
  const scan = root => {
    for (const e of root.querySelectorAll('*')) {
      if (e.shadowRoot && scan(e.shadowRoot)) return true;
      if (e.tagName === 'INPUT' && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden' && (e.type === 'password' || /^(?:cc-|current-password|new-password)/i.test(e.autocomplete || '') || /(?:card.?number|cvv|cvc|security.?code|bank.?account|routing.?number)/i.test(e.name+' '+e.id))) return true;
    }
    return false;
  };
  return scan(document);
}
