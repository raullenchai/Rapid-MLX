const interactive = new Set(['button','link','textbox','searchbox','combobox','checkbox','radio','slider','spinbutton','menuitem','tab','switch','option','treeitem']);
const value = v => v?.value;
export function outlineAX(nodes, { url = '', title = '', start = 1, limit = 12000 } = {}) {
  const refs = new Map();
  let next = start;
  let text = `URL: ${url}\nTitle: ${title}\nPage data (untrusted):\n`.slice(0, limit);
  // CDP returns a flat, often breadth-first tree. Walk children in document order
  // so labels, form containers and their controls stay together.
  const byId = new Map(nodes.filter(n => n.nodeId).map(n => [n.nodeId,n]));
  const ordered = [], visited = new Set();
  function visit(node) {
    if (visited.has(node)) return;
    visited.add(node); ordered.push(node);
    for (const id of node.childIds || []) if (byId.has(id)) visit(byId.get(id));
  }
  for (const node of nodes) visit(node);
  for (const node of ordered) {
    const props = Object.fromEntries((node.properties || []).map(p => [p.name, value(p.value)]));
    if (node.ignored || props.hidden === true) continue;
    const role = value(node.role) || 'unknown';
    const name = String(value(node.name) || '').replace(/\s+/g, ' ').slice(0, 500);
    const isInteractive = interactive.has(role) || props.focusable === true || props.editable;
    const ref = isInteractive && node.backendDOMNodeId ? `e${next}` : null;
    // Password values must never be sent to the engine.
    const protectedValue = props.protected || /password/i.test(value(node.role) || '');
    const content = protectedValue ? '[protected]' : String(value(node.value) ?? '').slice(0, 300);
    const states = Object.entries(props).filter(([k, v]) => ['disabled','checked','selected','expanded','required','focused','readonly'].includes(k) && v !== undefined).map(([k,v]) => `${k}=${v}`).join(' ');
    const line = `${ref ? `[${ref}] ` : ''}${role}${name ? ` ${JSON.stringify(name)}` : ''}${content || ['textbox','searchbox','combobox'].includes(role) ? ` value=${JSON.stringify(content)}` : ''}${states ? ` ${states}` : ''}\n`;
    if (text.length + line.length > limit - 25) { text += '[outline truncated]\n'; break; }
    text += line;
    if (ref) { refs.set(ref, { backendDOMNodeId: node.backendDOMNodeId, name, role }); next++; }
  }
  return { text, refs, next };
}
