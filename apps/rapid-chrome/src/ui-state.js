// Full snapshots carry a worker epoch and monotonic sequence; reconnect is idempotent.
export function acceptsSnapshot(previous, next) {
  return !previous || !next.epoch || previous.epoch !== next.epoch || next.sequence > previous.sequence;
}
export class ScrollFollow {
  constructor() { this.follow = true; this.unread = 0; this.seen = new Set(); }
  scroll({scrollHeight,scrollTop,clientHeight}, selected = false) {
    this.follow = !selected && scrollHeight - scrollTop - clientHeight <= 40;
    if (this.follow) this.unread = 0;
  }
  update(messages, selected = false) {
    if (selected) this.follow = false;
    const ids = messages.map((m,i) => m.id || `${m.at || ''}:${m.role}:${i}`);
    const added = ids.filter(id => !this.seen.has(id)).length;
    if (this.seen.size && !this.follow) this.unread += added;
    this.seen = new Set(ids);
    return this.follow;
  }
  bottom() { this.follow = true; this.unread = 0; }
}
export function errorType(value = '') {
  const text = String(value);
  if (/^(?:cors|origin_refused)$|CORS (?:blocked|refused)|origin (?:blocked|refused|not allowed)/i.test(text)) return 'cors';
  if (/noTools|no.tool.support|does not support.*tool|model did not inspect/i.test(text)) return 'noTools';
  if (/step.cap/i.test(text)) return 'stepCap';
  if (/unreachable|not reachable|unavailable|failed to fetch|connectionTimeout|ECONNREFUSED/i.test(text)) return 'unreachable';
  return 'unknown';
}
export function runSnapshot(state) {
  return {version:1,savedAt:Date.now(),sequence:state.sequence,run:state.run,messages:state.messages.slice(-60).map(m => ({...m,content:String(m.content).slice(0,12000)})),steps:state.steps.slice(-100).map(s => ({...s,result:String(s.result || '').slice(0,6000)})),status:state.status,error:state.error,running:state.running,paused:state.paused,pauseReason:state.pauseReason,resultTab:state.resultTab};
}
export function restoreRun(saved) {
  if (saved?.version !== 1) return {};
  return {...saved,steps:(saved.steps || []).map(step => saved.running && step.status === 'running' ? {...step,status:'Stopped'} : step),approvals:[],running:false,paused:false,pauseReason:null,
    ...(saved.running ? {status:'Stopped',error:null,steps:(saved.steps || []).map(step => step.status === 'running' ? {...step,status:'stopped'} : step),run:{...saved.run,endedAt:saved.savedAt || Date.now(),interrupted:true}} : {})};
}
export function activityLabels(steps, t, running, seconds = 0) {
  const sites = new Set(steps.map(step => { try { return new URL(step.args?.url || step.url).hostname; } catch { return null; } }).filter(Boolean)).size;
  const count = steps.length;
  const stepText = t(count === 1 ? 'oneStep' : 'manySteps',[String(count)]);
  return running ? t('browsedSummary',[t(sites === 1 ? 'oneSite' : 'manySites',[String(sites)]),stepText]) : t('workedSummary',[String(seconds),stepText]);
}
export function quietSteps(steps) {
  return steps.filter((step,i) => {
    const prev = steps[i-1];
    return !(prev && step.name === 'read_page' && prev.name === step.name && prev.url === step.url && prev.status === step.status && JSON.stringify(prev.args) === JSON.stringify(step.args));
  });
}
