const string = { type: 'string' };
const tab = { tabId: { type: 'integer', description: 'Owned tab ID; omit to use the current owned tab.' } };
const spec = (name, description, properties = {}, required = []) => ({ type:'function', function: { name, description, parameters: { type:'object', properties, required, additionalProperties:false } } });
export const TOOLS = [
  spec('list_tabs', 'List only tabs you own or the user explicitly shared.'),
  spec('open_tab', 'Open an HTTP(S) URL in the Rapid group.', {url:string}, ['url']),
  spec('navigate', 'Navigate an owned tab; choose exactly one URL or history action.', {...tab,url:string,action:{type:'string',enum:['back','forward','reload']}}),
  spec('read_page', 'First tool to call. Read current controls, values and page text. Use only refs from this result; older refs are stale. Call again after each action.', tab),
  spec('click', 'Click a button or link by its current ref. After filling a form, read_page then click its submit button. Sensitive actions require approval.', {...tab,ref:string}, ['ref']),
  spec('fill', 'Fill a textbox by ref with text. Then read_page and click the form button to submit. Password/card fields are blocked. Leave submit false.', {...tab,ref:string,text:string,submit:{type:'boolean'}}, ['ref','text']),
  spec('press', 'Press a single key: Enter, Tab, Escape, Backspace, Delete, Space, ArrowUp/Down/Left/Right, Home, End, PageUp/Down. Activation/deletion requires approval.', {...tab,key:{type:'string',enum:['Enter','Tab','Escape','Backspace','Delete','Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End','PageUp','PageDown']}}, ['key']),
  spec('scroll', 'Scroll up/down/left/right or an observed ref into view.', {...tab,direction:{type:'string',enum:['up','down','left','right']},ref:string}),
  spec('screenshot', 'Capture JPEG only when vision is enabled.', tab),
  spec('wait', 'Wait at most 5000ms.', {ms:{type:'integer',minimum:0,maximum:5000}}, ['ms'])
];
export function validateTool(name, args) {
  const schema = TOOLS.find(t => t.function.name === name)?.function.parameters;
  if (!schema || !args || typeof args !== 'object' || Array.isArray(args)) throw new Error('invalid_tool: unknown tool or arguments');
  for (const key of schema.required) if (!(key in args)) throw new Error(`invalid_arguments: missing ${key}`);
  for (const [key,v] of Object.entries(args)) {
    const p = schema.properties[key];
    if (!p || (p.type === 'integer' ? !Number.isInteger(v) : typeof v !== p.type) ||
        (p.enum && !p.enum.includes(v)) || (p.minimum !== undefined && v < p.minimum) || (p.maximum !== undefined && v > p.maximum)) throw new Error(`invalid_arguments: ${key}`);
  }
  if (name === 'navigate' && Boolean(args.url) === Boolean(args.action)) throw new Error('invalid_arguments: choose url OR action');
  if (name === 'scroll' && Boolean(args.ref) === Boolean(args.direction)) throw new Error('invalid_arguments: choose ref OR direction');
}
