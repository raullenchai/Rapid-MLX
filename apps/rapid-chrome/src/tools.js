const string = { type: 'string' };
const tab = { tabId: { type: 'integer', description: 'Owned tab ID; omit to use the current owned tab.' } };
const spec = (name, description, properties = {}, required = []) => ({ type:'function', function: { name, description, parameters: { type:'object', properties, required, additionalProperties:false } } });
export const TOOLS = [
  spec('list_tabs', 'List only tabs you own or the user explicitly shared.'),
  spec('open_tab', 'Open a user-provided URL, an observed outline href, or a site root/search URL (new sites require approval). Never invent product/venue URLs.', {url:string}, ['url']),
  spec('navigate', 'Navigate to a user URL, observed href, or approved site root/search; never invent deep URLs. Choose exactly one URL or history action.', {...tab,url:string,action:{type:'string',enum:['back','forward','reload']}}),
  spec('read_page', 'First tool to call. Read current controls, values and page text. Use only refs from this result; older refs are stale. Viewport and main content first; use query to find products/text across the whole page instead of scrolling loops.', {...tab,query:{type:'string',description:'Case-insensitive text filter; includes matching product cards and their controls.'}}),
  spec('click', 'Click a button or link by its current ref. Returns effects and a fresh outline; use its refs for the next action. Automatically scrolls the target into view. Sensitive actions require approval.', {...tab,ref:string}, ['ref']),
  spec('fill', 'Fill a textbox by ref with text. For autocomplete use text=airport code/city and choose=exact option label (or unique words) to fill and select in one step. Returns effects and a fresh outline; use its refs directly. Password/card fields are blocked. Leave submit false.', {...tab,ref:string,text:string,choose:{type:'string',description:'Select a uniquely matching autocomplete option after typing; e.g. JFK. Does not submit the form.'},submit:{type:'boolean'}}, ['ref','text']),
  spec('press', 'Press a single key: Enter, Tab, Escape, Backspace, Delete, Space, ArrowUp/Down/Left/Right, Home, End, PageUp/Down. Enter uses the same consequential form classifier as clicks. Navigation keys need no approval.', {...tab,key:{type:'string',enum:['Enter','Tab','Escape','Backspace','Delete','Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End','PageUp','PageDown']}}, ['key']),
  spec('scroll', 'Scroll up/down/left/right or an observed ref into view. Returns a fresh outline; use its refs directly.', {...tab,direction:{type:'string',enum:['up','down','left','right']},ref:string}),
  spec('screenshot', 'Capture JPEG only when vision is enabled.', tab),
  spec('wait', 'Wait at most 5000ms and return a fresh outline; use its refs directly.', {ms:{type:'integer',minimum:0,maximum:5000}}, ['ms'])
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
  if (name === 'fill' && args.choose !== undefined && (!/[\p{L}\p{N}]/u.test(args.choose) || args.submit)) throw new Error('invalid_arguments: choose must be nonempty and cannot combine with submit');
  if (name === 'navigate' && Boolean(args.url) === Boolean(args.action)) throw new Error('invalid_arguments: choose url OR action');
  if (name === 'scroll' && Boolean(args.ref) === Boolean(args.direction)) throw new Error('invalid_arguments: choose ref OR direction');
}
