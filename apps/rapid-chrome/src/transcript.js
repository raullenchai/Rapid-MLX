// Bound both DOM work and text layout independently of task/history length.
export const VISIBLE_STEPS = 20;
export const TEXT_LIMIT = 6000;
export const clip = (value, limit = TEXT_LIMIT) => {
  const text = String(value ?? '');
  return text.length > limit ? `${text.slice(0,limit)}\n[…]` : text;
};
export function renderSteps(document, container, steps, {label, result, status, older}) {
  const open = new Set([...container.querySelectorAll('details[open]')].map(e => e.dataset.id));
  const node = (tag, text) => { const e = document.createElement(tag); e.textContent = text; return e; };
  const rows = steps.slice(-VISIBLE_STEPS).map(step => {
    const d = document.createElement('details'); d.dataset.id = step.id;
    const args = clip(JSON.stringify(step.args),1000), output = clip(result(step.result));
    d.append(node('summary',`${label(step.name)} · ${status(step.status)} · ${clip(args,160)} · ${clip(output,200)}`));
    const expand = () => {
      if (d.open && d.children.length === 1) d.append(node('pre',args),node('p',output));
    };
    d.ontoggle = expand;
    d.open = open.has(step.id); expand();
    return d;
  });
  if (steps.length > VISIBLE_STEPS) rows.unshift(node('p',older(steps.length - VISIBLE_STEPS)));
  container.replaceChildren(...rows);
}
