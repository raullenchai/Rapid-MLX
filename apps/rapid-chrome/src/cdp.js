import { outlineAX } from './ax.js';
import { safeURL, classifyTarget, needsSiteApproval } from './safety.js';
import { delay } from './agent.js';
import { overlay } from './overlay.js';

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
  inject(id, show) { return this.api.scripting.executeScript({target:{tabId:id},func:overlay,args:[show]}); }
}

// All browser operations pass through the adapter; no model-provided JS is evaluated.
export class BrowserHand {
  constructor(adapter, {settings, approve, emit = () => {}, stop = () => {}}) {
    this.a = adapter; this.settings = settings; this.approve = approve; this.emit = emit; this.stop = stop;
    this.tabs = new Map(); this.active = null; this.nextRef = 1; this.sites = new Set(); this.signal = null;
    this.notices = []; this.pendingSites = new Map();
  }
  check() { this.signal?.throwIfAborted(); if (this.paused) throw new Error('paused: debugger detached by user'); }
  async send(id, method, params) {
    this.check();
    if (!this.tabs.get(id)?.attached) throw new Error('tab_taken_over: user owns this tab now');
    const value = await this.a.send(id, method, params);
    this.check();
    if (method === 'Page.navigate' && value.errorText) throw new Error(`navigation_failed: ${value.errorText}`);
    return value;
  }
  invalidate(id) { const t = this.tabs.get(id); if (t) { t.refs.clear(); t.epoch++; } }
  async site(url, action = 'Navigate') {
    const u = safeURL(url, this.settings.base);
    if (needsSiteApproval(u.href, this.sites)) {
      if (!this.pendingSites.has(u.origin)) {
        const request = (async () => {
          if (!await this.approve({action,target:u.href,site:u.origin})) throw new Error('denied: site navigation');
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
    if (!created) safeURL(tab.url, this.settings.base);
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
        await this.send(id,'DOM.enable');
        await this.send(id,'Accessibility.enable');
        await this.send(id,'Runtime.enable');
        await this.send(id,'Fetch.enable',{patterns:[{urlPattern:'*',resourceType:'Document',requestStage:'Request'}]});
      } catch (e) { await this.release(id); throw e; }
    }
    if (created) { t.grouping = true; try { t.groupId = await this.a.group(id); } finally { t.grouping = false; } }
    return t;
  }
  async begin(signal) {
    this.signal = signal; this.sites.clear(); this.pendingSites.clear();
    if (this.active !== null && this.tabs.has(this.active)) {
      const tab = await this.a.get(this.active);
      this.sites.add(safeURL(tab.url,this.settings.base).origin);
    }
    for (const [id] of this.tabs) { await this.own(id); this.invalidate(id); await this.showOverlay(id,true); }
  }
  async showOverlay(id, show) {
    try { await this.a.inject(id,show); return; } catch { /* Host permission is deliberately absent for arbitrary sites. */ }
    try {
      const {frameTree} = await this.a.send(id,'Page.getFrameTree');
      const {executionContextId} = await this.a.send(id,'Page.createIsolatedWorld',{frameId:frameTree.frame.id,worldName:'Rapid isolated overlay'});
      const binding = this.tabs.get(id)?.binding;
      if (!binding) return;
      await this.a.send(id,'Runtime.addBinding',{name:binding,executionContextId});
      await this.a.send(id,'Runtime.evaluate',{contextId:executionContextId,expression:`(${overlay.toString()})(${JSON.stringify(show)},${JSON.stringify(binding)})`});
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
    if (method === 'Page.frameNavigated') {
      this.invalidate(id);
      if (!p.frame.parentId) {
        if (t.bootstrapping && p.frame.url === 'about:blank') return;
        try { safeURL(p.frame.url,this.settings.base); } catch { await this.takeover(id,'navigated to a blocked URL'); return; }
      }
    }
    if (method === 'Page.navigatedWithinDocument') this.invalidate(id);
    if (method === 'Page.loadEventFired' && this.signal && !this.signal.aborted) await this.showOverlay(id,true);
    if (method === 'Fetch.requestPaused') {
      try {
        if (!this.signal || this.signal.aborted) throw new Error('Task stopped');
        await this.site(p.request.url,'Navigate document');
        this.check();
        await this.send(id,'Fetch.continueRequest',{requestId:p.requestId});
      } catch (e) {
        try { await this.a.send(id,'Fetch.failRequest',{requestId:p.requestId,errorReason:'BlockedByClient'}); } catch {}
        this.notices.push(`Navigation blocked: ${e.message}`);
      }
    }
  }
  async current(args) {
    const id = args.tabId ?? this.active;
    if (!this.tabs.has(id)) throw new Error('unowned_tab: use open_tab or ask the user to share a tab');
    if (!this.tabs.get(id).attached) throw new Error('tab_detached: ask the user to keep working');
    const tab = await this.a.get(id);
    await this.site(tab.url,'Operate tab');
    this.active = id; return {id,tab,t:this.tabs.get(id)};
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
      return {formName:accessibleName(form),submitter:submitters.map(accessibleName).join(' '),method:form?.method || '',name:[accessibleName(e),label].filter(Boolean).join(' ').slice(0,1000),type:e.type || e.getAttribute('type') || '',autocomplete:e.getAttribute('autocomplete') || '',id:[e.id,e.getAttribute('name')].join(' '),form:form?.innerText.slice(0,2000) || '',href:e.closest('a')?.href || '',action:form?.action || '',connected:e.isConnected};
    });
  }
  async gateTarget(id, target, action, {fill = false, submit = false, always = false} = {}) {
    const info = await this.inspect(id,target);
    if (!info.connected) throw new Error('stale_ref: call read_page');
    const classification = classifyTarget({...info,name:`${target.name} ${info.name}`});
    if (fill && classification.sensitive) throw new Error('blocked_sensitive_field: ask the user to enter passwords or payment details');
    if (always || (submit && !classification.search) || ((!fill || submit) && classification.consequential)) {
      const tab = await this.a.get(id);
      if (!await this.approve({action,target:target.name || info.name || 'Focused element',site:new URL(tab.url).origin})) throw new Error('denied: action not executed');
      this.check();
      if (target.epoch !== this.tabs.get(id)?.epoch || JSON.stringify(info) !== JSON.stringify(await this.inspect(id,target))) throw new Error('stale_ref: target changed during approval; call read_page');
    }
    if (!fill && info.href) await this.site(info.href,'Follow link');
    if (submit && info.action) await this.site(info.action,'Submit form');
    return info;
  }
  async click(id,target) {
    await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId});
    const {model} = await this.send(id,'DOM.getBoxModel',{backendNodeId:target.backendDOMNodeId});
    const q = model.content, x = (q[0]+q[2]+q[4]+q[6])/4, y = (q[1]+q[3]+q[5]+q[7])/4;
    const hit = await this.send(id,'DOM.getNodeForLocation',{x:Math.round(x),y:Math.round(y),includeUserAgentShadowDOM:true});
    if (hit.backendNodeId !== target.backendDOMNodeId) {
      const {object} = await this.send(id,'DOM.resolveNode',{backendNodeId:hit.backendNodeId});
      const r = await this.send(id,'Runtime.callFunctionOn',{objectId:target.objectId,functionDeclaration:'function(other) { return this === other || this.contains(other); }',arguments:[{objectId:object.objectId}],returnByValue:true});
      if (!r.result.value) throw new Error('obscured_target: hit test failed; call read_page');
    }
    if (target.epoch !== this.tabs.get(id)?.epoch) throw new Error('stale_ref: call read_page');
    await this.send(id,'Input.dispatchMouseEvent',{type:'mousePressed',x,y,button:'left',clickCount:1});
    await this.send(id,'Input.dispatchMouseEvent',{type:'mouseReleased',x,y,button:'left',clickCount:1});
  }
  async key(id,key) {
    const codes = {Enter:13,Tab:9,Escape:27,Backspace:8,Delete:46,Space:32,ArrowLeft:37,ArrowUp:38,ArrowRight:39,ArrowDown:40,Home:36,End:35,PageUp:33,PageDown:34};
    const p = {key:key === 'Space' ? ' ' : key,code:key === 'Space' ? 'Space' : key,windowsVirtualKeyCode:codes[key]};
    await this.send(id,'Input.dispatchKeyEvent',{type:'keyDown',...p,...(key === 'Enter' ? {text:'\r'} : key === 'Space' ? {text:' '} : {})});
    await this.send(id,'Input.dispatchKeyEvent',{type:'keyUp',...p});
  }
  async execute(name,args,signal) {
    this.signal = signal; this.check();
    if (name === 'list_tabs') {
      const tabs = await Promise.all([...this.tabs.keys()].map(async id => { const t = await this.a.get(id); return {tabId:id,title:t.title,url:t.url,attached:this.tabs.get(id).attached}; }));
      return {tabs};
    }
    if (name === 'wait') { await delay(args.ms,signal); return {text:`Waited ${args.ms}ms`,...(this.tabs.has(this.active) ? await this.meta(this.active) : {})}; }
    if (name === 'open_tab') {
      const url = await this.site(args.url,'Open tab');
      const tab = await this.a.create();
      this.check(); await this.own(tab.id,{created:true});
      await this.send(tab.id,'Page.navigate',{url});
      this.tabs.get(tab.id).bootstrapping = false;
      await this.showOverlay(tab.id,true);
      return {text:'Opened tab',tabId:tab.id,...await this.meta(tab.id)};
    }
    const {id,tab,t} = await this.current(args);
    if (name === 'read_page') {
      this.invalidate(id);
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
      const outline = outlineAX(nodes,{url:tab.url,title:tab.title,start:this.nextRef});
      t.refs = outline.refs; this.nextRef = outline.next;
      const forms = new Map();
      for (const [ref, target] of outline.refs) {
        if (!['button','textbox','searchbox','combobox'].includes(target.role)) continue;
        try {
          const resolved = await this.resolve(id,ref);
          const form = await this.call(id,resolved.objectId,function () {
            const f = this.form || this.closest('form');
            if (!f) return null;
            return {index:[...document.forms].indexOf(f),name:f.getAttribute('aria-label') || f.getAttribute('name') || '',method:f.method};
          });
          if (form) {
            const key = `${form.index}: ${JSON.stringify(form.name)} (${form.method})`;
            forms.set(key,[...(forms.get(key) || []),`[${ref}]`]);
          }
        } catch { this.check(); }
      }
      if (epoch !== t.epoch) throw new Error('stale_ref: page changed while reading forms; call read_page again');
      const formText = [...forms].map(([form,refs]) => `Form ${form}: ${refs.join(' ')}`).join('\n');
      return {text:`${formText ? `Form controls (use refs in outline):\n${formText}\n` : ''}${outline.text}`};
    }
    if (name === 'screenshot') {
      if (!this.settings.vision) return {note:'Vision is off. Use read_page.',...await this.meta(id)};
      const {data} = await this.send(id,'Page.captureScreenshot',{format:'jpeg',quality:65,captureBeyondViewport:false});
      return {text:'Screenshot captured',image:data,...await this.meta(id)};
    }
    if (name === 'navigate') {
      this.invalidate(id);
      if (args.url) await this.send(id,'Page.navigate',{url:await this.site(args.url)});
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
      if (name === 'click') await this.click(id,target);
      else {
        await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId});
        await this.send(id,'DOM.focus',{backendNodeId:target.backendDOMNodeId});
        // CDP editing command is platform independent (unlike Ctrl/Command+A).
        await this.send(id,'Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',commands:['selectAll']});
        await this.send(id,'Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA'});
        await this.send(id,'Input.insertText',{text:args.text});
        if (args.submit) {
          // Recheck after input handlers may have changed the form.
          await this.gateTarget(id,target,'Submit form',{fill:true,submit:true});
          const info = await this.inspect(id,target);
          if (classifyTarget(info).sensitive) throw new Error('blocked_sensitive_field');
          await this.key(id,'Enter');
        }
      }
    } else if (name === 'press') {
      const r = await this.send(id,'Runtime.evaluate',{expression:'document.activeElement',returnByValue:false});
      const target = {objectId:r.result.objectId,epoch:t.epoch,name:'Focused element'};
      // Enter submits the focused form; classify that effect just like fill+submit.
      await this.gateTarget(id,target,`Press ${args.key}`,{submit:args.key === 'Enter',always:args.key !== 'Enter'});
      await this.key(id,args.key);
    } else if (name === 'scroll') {
      if (args.ref) { const target = await this.resolve(id,args.ref); await this.send(id,'DOM.scrollIntoViewIfNeeded',{backendNodeId:target.backendDOMNodeId}); }
      else await this.send(id,'Input.dispatchMouseEvent',{type:'mouseWheel',x:100,y:200,deltaX:args.direction === 'right' ? 600 : args.direction === 'left' ? -600 : 0,deltaY:args.direction === 'down' ? 600 : args.direction === 'up' ? -600 : 0});
    }
    return {text:`${name} completed`,...await this.meta(id)};
  }
  async meta(id) { const tab = await this.a.get(id); return {url:tab.url,title:tab.title}; }
}
