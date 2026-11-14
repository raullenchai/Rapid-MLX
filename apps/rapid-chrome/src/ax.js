const interactive = new Set(['button','link','textbox','searchbox','combobox','checkbox','radio','slider','spinbutton','menuitem','tab','switch','option','treeitem']);
const value = v => v?.value;
export function outlineAX(nodes, {url = '', title = '', start = 1, limit = 12000, query = '', geometry = new Map()} = {}) {
  const refs = new Map(), byId = new Map(nodes.filter(n => n.nodeId).map(n => [n.nodeId,n]));
  const visited = new Set(), entries = [];
  let next = start;
  const nameOf = n => String(value(n.name) || '').replace(/\s+/g,' ').trim();
  function descendants(n, seen = new Set()) {
    if (seen.has(n)) return [];
    seen.add(n);
    return [n,...(n.childIds || []).flatMap(id => byId.has(id) ? descendants(byId.get(id),seen) : [])];
  }
  function visit(node, main = false, parentName = '', priority = false) {
    if (visited.has(node)) return;
    visited.add(node);
    const role = value(node.role) || 'unknown', name = nameOf(node);
    const props = Object.fromEntries((node.properties || []).map(p => [p.name,value(p.value)]));
    main ||= role === 'main';
    priority ||= ['dialog','alertdialog','alert','status'].includes(role) || Boolean(props.live && props.live !== 'off');
    if (props.hidden === true) { for (const child of descendants(node)) visited.add(child); return; }
    // Compact a semantic product card as a unit: retain prices/ratings and all
    // actionable refs, omit repeated image/link/static text copies of its name.
    const children = ['listitem','article'].includes(role) ? descendants(node) : [];
    const product = children.length && children.some(n => value(n.role) === 'link') && children.some(n => /[$€£¥]\s*\d|\d[.,]\d{2}\s*(?:USD|EUR)/.test(nameOf(n)));
    if (product) {
      const parts = [], targets = [], names = new Set();
      const usable = children.filter(n => !n.ignored && value(n.role) !== 'InlineTextBox' && nameOf(n));
      const links = usable.filter(n => value(n.role) === 'link' && n.backendDOMNodeId);
      // Name/link first, then price/rating, then other controls. Long cards cannot
      // consume the entire outline budget before any product becomes visible.
      const primary = links.find(n => nameOf(n).length > 12) || links[0];
      const details = usable.filter(n => !interactive.has(value(n.role)) && /[$€£¥]\s*\d|\d[.,]\d{2}|\bstars?\b|\bratings?\b|\breviews?\b/i.test(nameOf(n)));
      const controls = usable.filter(n => interactive.has(value(n.role)) && n !== primary && n.backendDOMNodeId);
      let shortened = false, size = 0;
      for (const n of [primary,...details,...controls].filter(Boolean)) {
        const label = nameOf(n);
        if (names.has(label)) continue;
        const control = interactive.has(value(n.role)) && n.backendDOMNodeId;
        const part = control ? `[e${next}] ${value(n.role)} ${JSON.stringify(label.slice(0,300))}` : label.slice(0,300);
        if (size+part.length > 1400) { shortened = true; continue; }
        if (control) targets.push([`e${next++}`,{backendDOMNodeId:n.backendDOMNodeId,name:label,role:value(n.role)}]);
        parts.push(part); size += part.length+3; names.add(label);
      }
      for (const n of children) visited.add(n);
      if (shortened) parts.push('[additional card details omitted; open product to inspect]');
      const visible = children.some(n => geometry.get(n.backendDOMNodeId) === true);
      entries.push({text:`Product: ${parts.join(' | ')}\n`,targets,main,priority,visible,offscreen:geometry.size>0 && !visible,search:children.map(nameOf).join(' ')});
      return;
    }
    const isInteractive = interactive.has(role) || (props.focusable === true && !['RootWebArea','generic'].includes(role)) || props.editable;
    const ref = !node.ignored && isInteractive && node.backendDOMNodeId ? `e${next++}` : null;
    const protectedValue = props.protected || /password/i.test(role);
    const content = protectedValue ? '[protected]' : String(value(node.value) ?? '').slice(0,300);
    const states = Object.entries(props).filter(([k,v]) => ['disabled','checked','selected','expanded','required','focused','readonly'].includes(k) && v !== undefined).map(([k,v]) => `${k}=${v}`).join(' ');
    if (!node.ignored && role !== 'InlineTextBox' && (ref || name || content) && !(role === 'StaticText' && parentName === name)) {
      entries.push({text:`${ref ? `[${ref}] ` : ''}${role}${name ? ` ${JSON.stringify(name.slice(0,500))}` : ''}${content || ['textbox','searchbox','combobox'].includes(role) ? ` value=${JSON.stringify(content)}` : ''}${states ? ` ${states}` : ''}\n`,targets:ref ? [[ref,{backendDOMNodeId:node.backendDOMNodeId,name,role}]] : [],main,priority,visible:geometry.get(node.backendDOMNodeId) === true,offscreen:geometry.get(node.backendDOMNodeId) === false,search:name+' '+content});
    }
    for (const id of node.childIds || []) if (byId.has(id)) visit(byId.get(id),main,name || parentName,priority);
  }
  const childIds = new Set(nodes.flatMap(n => n.childIds || []));
  for (const n of nodes.filter(n => !childIds.has(n.nodeId))) visit(n);
  for (const n of nodes) visit(n);
  const filtered = entries.filter(e => e.priority || !query || e.search.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  // Stable sort preserves each region's document order. Visible main content,
  // visible controls, then remaining main content precede secondary navigation.
  const rank = e => e.priority ? (e.targets.some(([,t]) => t.role === 'button' && /^(?:done|apply|close|cancel|search)\b/i.test(t.name)) ? -2 : -1) : e.visible ? (e.main ? 0 : 1) : e.main ? 2 : 3;
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
  if (query && !filtered.length) text += '[No matching text; try a shorter query or read without query.]\n';
  return {text,refs,next};
}
