const interactive = new Set(['button','link','textbox','searchbox','combobox','checkbox','radio','slider','spinbutton','menuitem','tab','switch','option','treeitem']);
const value = v => v?.value;
// Classify each price in tree order, retaining headings that are separate AX nodes.
export function commercePrices(root, byId) {
  const labels = new Map(), seen = new Set();
  const money = /[$€£¥]\s*\d[\d,.]*(?:\s*\d{2}\b)?|\d[.,]\d{2}\s*(?:USD|EUR)/g;
  const hint = text => /more buying choices|other (?:offers|sellers)|used\s*(?:&|and)\s*new|new\s*(?:&|and)\s*used|offers? from/i.test(text) ? 'other offers from' : /unit price|\/\s*(?:count|ounce|oz|lb|kg|g|ml|l|each)\b|per (?:count|ounce|item|unit)/i.test(text) ? 'unit price' : /list price|was\b|original price|typical price|previous price/i.test(text) ? 'was' : /^(?:price|current price|our price|buy box)\b/i.test(text) ? 'price' : '';
  function walk(n, inherited = '') {
    if (seen.has(n) || (n.properties || []).some(p => p.name === 'hidden' && value(p.value) === true)) return;
    seen.add(n);
    const text = String(value(n.name) || '');
    const struck = ['deletion','strikethrough'].includes(value(n.role));
    const role = struck ? 'was' : inherited;
    // InlineTextBox is Chromium's layout copy of its StaticText parent, not
    // another semantic price. Otherwise every real price is assigned only to
    // a layout node which the outline deliberately omits.
    const children = (n.childIds || []).map(id => byId.get(id)).filter(c => c && value(c.role) !== 'InlineTextBox');
    const ownPrices = [...text.matchAll(money)];
    // Aggregate container names often duplicate every price below them.
    if (ownPrices.length && !children.some(c => /[$€£¥]\s*\d|\d[.,]\d{2}\s*(?:USD|EUR)/.test(String(value(c.name) || '')))) {
      labels.set(n,ownPrices.map(m => {
        const previous = ownPrices[ownPrices.indexOf(m)-1];
        const before = text.slice(previous ? previous.index+previous[0].length : 0,m.index);
        const after = text.slice(m.index + m[0].length);
        const local = hint(before) || (/^\s*\//.test(after) ? 'unit price' : '') || (ownPrices.length === 1 ? hint(text) : '') || role || 'price';
        return `${local}: ${m[0].trim()}`;
      }).join(' | '));
    }
    let context = role || (!ownPrices.length ? hint(text) : '');
    for (const c of children) {
      const name = String(value(c.name) || '');
      const cue = hint(name);
      if (cue && !(c.childIds || []).length && !/[$€£¥]\s*\d/.test(name)) context = cue;
      walk(c,context);
      // Leaf offer headings may cover several seller prices. Keep that role
      // until another heading; never promote a later alternate offer to main.
    }
  }
  walk(root);
  return labels;
}
export function outlineAX(nodes, {url = '', title = '', start = 1, limit = 12000, query = '', geometry = new Map(), stableRefs = new Map()} = {}) {
  const refs = new Map(), byId = new Map(nodes.filter(n => n.nodeId).map(n => [n.nodeId,n]));
  const visited = new Set(), entries = [];
  let next = start;
  const hrefOf = n => (n.properties || []).find(p => p.name === 'url')?.value?.value || '';
  const linkText = n => hrefOf(n) ? ` href=${JSON.stringify(hrefOf(n))}` : '';
  const nameOf = n => String(value(n.name) || '').replace(/\s+/g,' ').trim();
  function descendants(n, seen = new Set()) {
    if (seen.has(n) || (n.properties || []).some(p => p.name === 'hidden' && value(p.value) === true)) return [];
    seen.add(n);
    return [n,...(n.childIds || []).flatMap(id => byId.has(id) ? descendants(byId.get(id),seen) : [])];
  }
  const refFor = n => stableRefs.get(n.backendDOMNodeId) || `e${next++}`;
  const parents = new Map();
  for (const n of nodes) for (const id of n.childIds || []) parents.set(byId.get(id),n);
  const textOf = n => [...new Set(descendants(n).filter(c => value(c.role) !== 'InlineTextBox').map(nameOf).filter(Boolean))].join(' ');
  const termsOf = text => String(text).normalize('NFKC').toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  const queryTerms = termsOf(query);
  const queryMatch = text => text.toLocaleLowerCase().includes(query.toLocaleLowerCase()) || (queryTerms.length > 0 && queryTerms.every(t => termsOf(text).includes(t)));
  const contextNodes = new Set();
  if (query) for (const node of nodes) {
    if (!nameOf(node) || !queryMatch(nameOf(node))) continue;
    let block = node, parent = parents.get(node);
    const seen = new Set([node]);
    while (parent && !seen.has(parent) && !['RootWebArea','main','navigation','banner'].includes(value(parent.role))) {
      seen.add(parent);
      if (textOf(parent).length > 2400) break;
      block = parent;
      if (['paragraph','row','cell','listitem','article','gridcell'].includes(value(parent.role)) && textOf(parent) !== nameOf(node)) break;
      parent = parents.get(parent);
    }
    for (const n of descendants(block)) contextNodes.add(n);
    // A movie/card may sit below a theater/section heading outside that card.
    // Keep the nearest preceding heading at each ancestor level, without
    // importing neighboring cards or their unrelated times/prices.
    let branch = block, headingBudget = 700;
    const ancestors = new Set();
    for (let ancestor = parents.get(branch); ancestor && !ancestors.has(ancestor); ancestor = parents.get(branch)) {
      ancestors.add(ancestor);
      if (['RootWebArea','navigation','banner'].includes(value(ancestor.role))) break;
      const siblings = (ancestor.childIds || []).map(id => byId.get(id)).filter(Boolean);
      const before = siblings.slice(0,siblings.indexOf(branch)).reverse();
      for (const sibling of before) {
        if (!['heading','generic','group'].includes(value(sibling.role))) continue;
        const header = descendants(sibling);
        if (header.some(n => interactive.has(value(n.role)))) continue;
        const headings = header.filter(n => value(n.role) === 'heading' && nameOf(n));
        const heading = headings.at(-1);
        if (!heading || nameOf(heading).length > headingBudget) continue;
        contextNodes.add(heading); headingBudget -= nameOf(heading).length; break;
      }
      branch = ancestor;
      if (!headingBudget || value(ancestor.role) === 'main') break;
    }
  }
  const cardRoles = new Set(['listitem','article','gridcell','generic','group']);
  const cardCandidate = n => {
    if (!cardRoles.has(value(n.role))) return false;
    const all = descendants(n).filter(c => !c.ignored);
    // A page/section containing a promotion is not a product. In particular,
    // never hide address/search fields or document editing controls in a card.
    if (all.some(c => ['main','navigation','banner','contentinfo','textbox','searchbox','combobox','dialog','alertdialog'].includes(value(c.role))) ||
        all.filter(c => value(c.role) === 'heading').length > 1 || textOf(n).length > 6000) return false;
    return all.some(c => (['heading','link'].includes(value(c.role)) && nameOf(c)) || (value(c.role) === 'button' && /^add \d+ (?:ct|count|pack) /i.test(nameOf(c)))) &&
      all.some(c => /[$€£¥]\s*\d|\d[.,]\d{2}\s*(?:USD|EUR)/.test(nameOf(c))) &&
      (['listitem','article','gridcell'].includes(value(n.role)) || all.some(c => value(c.role) === 'button'));
  };
  const candidates = new Set(nodes.filter(cardCandidate));
  function visit(node, main = false, parentName = '', priority = false) {
    if (visited.has(node)) return;
    visited.add(node);
    const role = value(node.role) || 'unknown', name = nameOf(node);
    const props = Object.fromEntries((node.properties || []).map(p => [p.name,value(p.value)]));
    main ||= role === 'main';
    priority ||= ['dialog','alertdialog','alert','status'].includes(role) || Boolean(props.live && props.live !== 'off');
    if (props.hidden === true) {
      const hide = n => { for (const id of n.childIds || []) { const child = byId.get(id); if (child && !visited.has(child)) { visited.add(child); hide(child); } } };
      hide(node); return;
    }
    // Compact a semantic product card as a unit: retain prices/ratings and all
    // actionable refs, omit repeated image/link/static text copies of its name.
    const children = candidates.has(node) ? descendants(node) : [];
    // A collection wrapper must never swallow its individual repeated cards.
    const product = children.length && !children.some(n => n !== node && candidates.has(n));
    if (product) {
      const parts = [], targets = [], names = new Set();
      const prices = commercePrices(node,byId);
      const usable = children.filter(n => !n.ignored && value(n.role) !== 'InlineTextBox' && nameOf(n));
      const links = usable.filter(n => value(n.role) === 'link' && n.backendDOMNodeId);
      // Name/link first, then price/rating, then other controls. Long cards cannot
      // consume the entire outline budget before any product becomes visible.
      const primary = links.find(n => !/[$€£¥]|more buying choices|other (?:offers|sellers)|used.*new/i.test(nameOf(n))) || links[0];
      const details = usable.filter(n => (prices.has(n) || (!interactive.has(value(n.role)) && !/[$€£¥]/.test(nameOf(n)))) && /[$€£¥]\s*\d|\d[.,]\d{2}|\bstars?\b|\bratings?\b|\breviews?\b/i.test(nameOf(n)));
      const controls = usable.filter(n => interactive.has(value(n.role)) && n !== primary && n.backendDOMNodeId);
      // Keep stock, usage dates, sender/body text and offer qualifications too.
      // Search must not find text in a card and then silently omit that text.
      const prose = usable.filter(n => !interactive.has(value(n.role)) &&
        !(n.childIds || []).some(id => byId.get(id) && value(byId.get(id).role) !== 'InlineTextBox'));
      let shortened = false, size = 0;
      const headings = usable.filter(n => value(n.role) === 'heading');
      for (const n of [...headings,primary,...details,...controls,...prose].filter(Boolean)) {
        const price = interactive.has(value(n.role)) ? '' : prices.get(n);
        const label = price ? `${price} — ${nameOf(n)}` : nameOf(n);
        if (names.has(label)) continue;
        const control = interactive.has(value(n.role)) && n.backendDOMNodeId;
        const ref = control ? refFor(n) : null;
        const content = (n.properties || []).some(p => p.name === 'protected' && value(p.value)) ? '[protected]' : String(value(n.value) ?? '').slice(0,300);
        const part = control ? `[${ref}] ${value(n.role)} ${JSON.stringify(label.slice(0,300))}${content ? ` value=${JSON.stringify(content)}` : ''}${linkText(n)}` : label.slice(0,300);
        if (size+part.length > 1400) { shortened = true; continue; }
        if (control) targets.push([ref,{backendDOMNodeId:n.backendDOMNodeId,name:nameOf(n),role:value(n.role),href:hrefOf(n)}]);
        parts.push(part); size += part.length+3; names.add(label);
      }
      for (const n of children) visited.add(n);
      if (shortened) parts.push('[additional card details omitted; open product to inspect]');
      const visible = children.some(n => geometry.get(n.backendDOMNodeId) === true);
      entries.push({text:`Product: ${parts.join(' | ')}\n`,targets,main,priority,visible,offscreen:geometry.size>0 && !visible,context:children.some(n => contextNodes.has(n)),search:children.map(nameOf).join(' ')});
      return;
    }
    const isInteractive = interactive.has(role) || (props.focusable === true && !['RootWebArea','generic'].includes(role)) || props.editable;
    const ref = !node.ignored && isInteractive && node.backendDOMNodeId ? refFor(node) : null;
    const protectedValue = props.protected || /password/i.test(role);
    const content = protectedValue ? '[protected]' : String(value(node.value) ?? '').slice(0,300);
    const states = Object.entries(props).filter(([k,v]) => ['disabled','checked','selected','expanded','required','focused','readonly'].includes(k) && v !== undefined).map(([k,v]) => `${k}=${v}`).join(' ');
    if (!node.ignored && role !== 'InlineTextBox' && (ref || name || content) && !(role === 'StaticText' && parentName === name)) {
      entries.push({text:`${ref ? `[${ref}] ` : ''}${role}${name ? ` ${JSON.stringify(name.slice(0,500))}` : ''}${content || ['textbox','searchbox','combobox'].includes(role) ? ` value=${JSON.stringify(content)}` : ''}${states ? ` ${states}` : ''}${linkText(node)}\n`,targets:ref ? [[ref,{backendDOMNodeId:node.backendDOMNodeId,name,role,href:hrefOf(node)}]] : [],main,priority,visible:geometry.get(node.backendDOMNodeId) === true,offscreen:geometry.get(node.backendDOMNodeId) === false,context:contextNodes.has(node),search:name+' '+content});
    }
    for (const id of node.childIds || []) if (byId.has(id)) visit(byId.get(id),main,name || parentName,priority);
  }
  const childIds = new Set(nodes.flatMap(n => n.childIds || []));
  for (const n of nodes.filter(n => !childIds.has(n.nodeId))) visit(n);
  for (const n of nodes) visit(n);
  // Match words across a long AX label without requiring invented contiguous
  // prose (e.g. airline + price + times). Normalize punctuation and thin spaces.
  const words = text => String(text).normalize('NFKC').toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
  const terms = words(query);
  const matches = e => !query || e.search.toLocaleLowerCase().includes(query.toLocaleLowerCase()) ||
    (terms.length > 0 && terms.every(term => words(e.search).includes(term)));
  const noMatch = Boolean(query) && !entries.some(matches);
  const filtered = entries.filter(e => noMatch || e.priority || e.context || matches(e));
  // Stable sort preserves each region's document order. Visible main content,
  // visible controls, then remaining main content precede secondary navigation.
  const rank = e => e.priority && e.targets.some(([,t]) => t.role === 'button' && /^(?:done|apply|close|cancel|search)\b/i.test(t.name)) ? -3 : query && !noMatch && (matches(e) || e.context) ? -2 : e.priority ? -1 : e.visible ? (e.main ? 0 : 1) : e.main ? 2 : 3;
  filtered.sort((a,b) => rank(a)-rank(b));
  let text = `URL: ${url}\nTitle: ${title}\nPage data (untrusted):\n`.slice(0,Math.max(0,limit-200));
  let included = 0;
  for (const e of filtered) {
    const line = (e.offscreen ? '[off-screen; click auto-scrolls] ' : '')+e.text;
    if (text.length+line.length > limit-200) break;
    text += line; included++;
    for (const [ref,target] of e.targets) refs.set(ref,target);
  }
  const omitted = filtered.length-included;
  if (omitted) text += `[outline truncated: ${omitted} entries/cards omitted; use read_page(query) to narrow results.]\n`;
  if (entries.some(e => e.offscreen)) text += '[More below/outside viewport; click known refs directly, or read_page(query) for omitted content.]\n';
  if (noMatch) text += '[No matching text for the query; showing the normal viewport-first outline. Use these fresh refs for site search or other controls.]\n';
  return {text,refs,next};
}
