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
export function reasoningOptions(effort, local = true) {
  return {...(['low','high'].includes(effort) ? {reasoning_effort:effort} : {}),...(local ? {chat_template_kwargs:{enable_thinking:['low','high'].includes(effort)}} : {})};
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
    const timeout = AbortSignal.timeout(15000);
    const requestSignal = signal ? AbortSignal.any([signal,timeout]) : timeout;
    const failure = fallback => new Error(body && !signal?.aborted && requestSignal.reason?.name === 'TimeoutError' ? 'connectionTimeout' : fallback);
    let response;
    try {
      response = await fetchImpl(`${base}${path}`, {method:body ? 'POST' : 'GET',headers:{'Content-Type':'application/json',...(settings.apiKey ? {Authorization:`Bearer ${settings.apiKey}`} : {})},credentials:'omit',redirect:'error',signal:requestSignal,...(body ? {body:JSON.stringify(body)} : {})});
    } catch { throw failure('unreachable'); }
    if (response.status === 401) throw new Error('unauthorized');
    if (response.status === 403) throw new Error('forbidden');
    if (!response.ok) throw new Error('endpointHTTP', {cause:response.status});
    try { return await response.json(); } catch { throw failure('invalidResponse'); }
  }
  const data = await request('/v1/models');
  if (!data || !Array.isArray(data.data)) throw new Error('invalidResponse');
  const models = (Array.isArray(data.data) ? data.data : []).map(m => m?.id).filter(id => typeof id === 'string' && id);
  if (!models.length) throw new Error('noModels');
  const model = models.includes(settings.model) ? settings.model : models[0];
  await onModels({models,model,base});
  let warning;
  if (probe) {
    try {
      const body = await request('/v1/chat/completions',{model,messages:[{role:'user',content:'Call rapid_connection_test with ok=true.'}],tools:[{type:'function',function:{name:'rapid_connection_test',description:'Connection test; no browser action.',parameters:{type:'object',properties:{ok:{type:'boolean'}},required:['ok'],additionalProperties:false}}}],tool_choice:{type:'function',function:{name:'rapid_connection_test'}},stream:false,...requestOptions(settings)});
      const calls = body?.choices?.[0]?.message?.tool_calls;
      const call = Array.isArray(calls) ? calls.find(c => c?.function?.name === 'rapid_connection_test') : undefined;
      let valid = false;
      try { valid = JSON.parse(call?.function?.arguments).ok === true; } catch {}
      if (!valid) throw new Error('noTools');
    } catch (e) {
      if (e.message !== 'connectionTimeout') throw e;
      warning = 'toolProbeTimeout';
    }
  }
  return {models,model,base,...(warning ? {warning} : {})};
}

export const rapidLocal = settings => settings.endpointMode === 'local' || (settings.endpointMode == null && LOCAL_ENDPOINTS.includes(settings.base));
export const pageProfile = settings => ['compact','standard'].includes(settings.pageDetail) ? settings.pageDetail : rapidLocal(settings) ? 'compact' : 'standard';
export function requestOptions(settings, final = false, expanded = false) {
  const local = rapidLocal(settings);
  return {...reasoningOptions(settings.reasoningEffort,local),
    ...(local ? {temperature:settings.temperature ?? 0.2,max_tokens:final || expanded ? 2048 : 1024} : settings.temperature != null ? {temperature:settings.temperature} : {})};
}
// Read only a bounded prefix, including when the server streams a huge error.
export async function endpointError(response, key = '') {
  let raw = '', unavailable;
  try {
    if (response.body?.getReader) {
      const reader = response.body.getReader(), decoder = new TextDecoder();
      let bytes = 0;
      try { while (bytes < 8192) {
        const {value,done} = await reader.read();
        if (done) break;
        const chunk = value.subarray(0,8192-bytes); bytes += chunk.length;
        raw += decoder.decode(chunk,{stream:true});
      } } finally { await reader.cancel(); }
    } else raw = (await response.text()).slice(0,8192);
  } catch (e) { unavailable = e.message; }
  const body = redact(raw,key).replace(/((?:token|secret|password|api[_-]?key)["']?\s*[:=]\s*["']?)[^\s,"'}]+/gi,'$1[REDACTED]');
  let data = {};
  try { data = JSON.parse(body); } catch {}
  const source = data.error && typeof data.error === 'object' ? data.error : data;
  const message = typeof source.message === 'string' ? source.message : typeof source.detail === 'string' ? source.detail : source.detail ? JSON.stringify(source.detail) : body;
  const type = /context[_ -](?:length|window|overflow)|maximum context|too many tokens|prompt.{0,30}too long/i.test([source.code,source.type,message].join(' ')) ? 'context_overflow' : source.type || source.code || (response.status === 502 ? 'bad_gateway' : 'unknown_http_error');
  return {status:response.status,type,code:source.code,message,detail:source.detail,body,bodyUnavailable:unavailable,requestId:response.headers?.get?.('x-request-id') || response.headers?.get?.('request-id') || source.request_id || data.request_id};
}
