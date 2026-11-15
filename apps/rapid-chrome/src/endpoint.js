// Endpoint URLs are origins: paths, credentials and query strings are rejected.
export function endpoint(value) {
  let u;
  try { u = new URL(value); } catch { throw new Error('invalidEndpoint'); }
  if (!['http:','https:'].includes(u.protocol) || u.username || u.password || u.search || u.hash || u.pathname !== '/') throw new Error('invalidEndpoint');
  if (u.protocol === 'http:' && !privateHost(u.hostname)) throw new Error('insecureEndpoint');
  return u.origin;
}
export function privateHost(host) {
  if (host === 'localhost' || host === '[::1]') return true;
  const p = host.split('.').map(Number);
  if (p.length !== 4 || p.some(n => !Number.isInteger(n) || n < 0 || n > 255)) return false;
  return p[0] === 127 || p[0] === 10 || (p[0] === 172 && p[1] >= 16 && p[1] <= 31) ||
    (p[0] === 192 && p[1] === 168) || (p[0] === 100 && p[1] >= 64 && p[1] <= 127);
}
export const originPermission = base => `${endpoint(base)}/*`;
// The Desktop default is PortSweep.defaultPort in the native app. Permission
// patterns cannot constrain ports; both probes share this one loopback grant.
export const LOCAL_ENDPOINTS = ['http://127.0.0.1:8000','http://127.0.0.1:7659'];
export function reasoningOptions(effort) {
  return ['low','high'].includes(effort) ? {reasoning_effort:effort,chat_template_kwargs:{enable_thinking:true}} : {chat_template_kwargs:{enable_thinking:false}};
}
export function redact(value, key = '') {
  if (Array.isArray(value)) return value.map(v => redact(v,key));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,/authorization|api.?key/i.test(k) ? '[REDACTED]' : redact(v,key)]));
  if (typeof value === 'string') return value.replace(/\b(?:Bearer\s+)[^\s"',;}]+/gi,'Bearer [REDACTED]').replace(/(authorization\s*[:=]\s*)[^\r\n]+/gi,'$1[REDACTED]').split(key || '\u0000').join(key ? '[REDACTED]' : '\u0000');
  return value;
}
export async function testConnection(settings, {fetchImpl = fetch, permissions = chrome.permissions, probe = true, signal, onModels = () => {}} = {}) {
  const base = endpoint(settings.base);
  if (!await permissions.contains({origins:[originPermission(base)]})) throw new Error('permissionMissing');
  async function request(path, body) {
    let response;
    try {
      response = await fetchImpl(`${base}${path}`, {method:body ? 'POST' : 'GET',headers:{'Content-Type':'application/json',...(settings.apiKey ? {Authorization:`Bearer ${settings.apiKey}`} : {})},credentials:'omit',redirect:'error',signal:signal ? AbortSignal.any([signal,AbortSignal.timeout(15000)]) : AbortSignal.timeout(15000),...(body ? {body:JSON.stringify(body)} : {})});
    } catch { throw new Error('unreachable'); }
    if (response.status === 401) throw new Error('unauthorized');
    if (response.status === 403) throw new Error('forbidden');
    if (!response.ok) throw new Error('endpointHTTP', {cause:response.status});
    try { return await response.json(); } catch { throw new Error('invalidResponse'); }
  }
  const data = await request('/v1/models');
  if (!data || !Array.isArray(data.data)) throw new Error('invalidResponse');
  const models = (Array.isArray(data.data) ? data.data : []).map(m => m?.id).filter(id => typeof id === 'string' && id);
  if (!models.length) throw new Error('noModels');
  const model = models.includes(settings.model) ? settings.model : models[0];
  await onModels({models,model,base});
  if (probe) {
    const body = await request('/v1/chat/completions',{model,messages:[{role:'user',content:'Call rapid_connection_test with ok=true.'}],tools:[{type:'function',function:{name:'rapid_connection_test',description:'Connection test; no browser action.',parameters:{type:'object',properties:{ok:{type:'boolean'}},required:['ok'],additionalProperties:false}}}],tool_choice:{type:'function',function:{name:'rapid_connection_test'}},stream:false,...reasoningOptions(settings.reasoningEffort)});
    const calls = body?.choices?.[0]?.message?.tool_calls;
    const call = Array.isArray(calls) ? calls.find(c => c?.function?.name === 'rapid_connection_test') : undefined;
    let valid = false;
    try { valid = JSON.parse(call?.function?.arguments).ok === true; } catch {}
    if (!valid) throw new Error('noTools');
  }
  return {models,model,base};
}
