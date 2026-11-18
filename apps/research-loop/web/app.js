'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors = {improved:'#176345',worse:'#a33432',supported:'#176345',refuted:'#a33432',crashed:'#a33432','no effect':'#795919',baseline:'#526476',testing:'#285e89'};
const state = {slug:null, stage:'discuss', data:null, run:null, rounds:new Map(), selected:0, best:0, follow:true, replay:false, speed:8, source:null, files:[], artifacts:[], watermark:0, projects:[], templates:[], domain:'', chatBusy:false, chatDraft:null, generation:0, loadingRun:null};
let noticeTimer, paintPending = false, artifactKey = '', artifactGeneration = 0;
const artifactText = {code:null,diff:null};
function notice(message='', duration=8000) {
  $('notice').textContent = message; clearTimeout(noticeTimer);
  if (message) noticeTimer = setTimeout(() => { $('notice').textContent = ''; }, duration);
}
async function api(path, data, method='POST') {
  const response = await fetch(path, data === undefined ? {} : {method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed');
  return value;
}
function projectAPI(suffix='') { return `/api/projects/${encodeURIComponent(state.slug)}${suffix}`; }
function fileURL(name, download=false) { return projectAPI(`/runs/${encodeURIComponent(state.run.id)}/files/${encodeURIComponent(name)}`) + (download ? '?download=1' : ''); }
function metricLabel(task=state.data?.task) { return task?.metric_label || task?.metric?.replaceAll('_',' ') || 'Metric'; }
function format(value, task=state.data?.task, signed=false) {
  if (!Number.isFinite(value)) return '—';
  const percent = task?.unit === '%';
  const number = (percent ? value*100 : value).toLocaleString(undefined, {maximumFractionDigits:percent ? 1 : 4, minimumFractionDigits:percent ? 1 : 0});
  return `${signed && value>0 ? '+' : ''}${number}${task?.unit ? (percent ? '%' : ' '+task.unit) : ''}`;
}
function changeLabel(r) { return (r.change || r.hypothesis || (r.round === 0 ? 'Original settings' : 'Change pending')).replaceAll('->','→'); }
function experimentVerdict(verdict) { return ({supported:'improved',refuted:'worse','not run':'crashed'})[verdict] || verdict; }
function color(verdict) { return colors[verdict] || colors.testing; }
function statusClass(verdict) { return (verdict || 'testing').replaceAll(' ','-'); }
function directionMark(task) {
  const down=task.direction==='min';
  return `<svg class="direction-mark" viewBox="0 0 16 16" aria-label="${down?'Lower':'Higher'} is better" role="img"><path d="${down?'M8 2v12m-5-5 5 5 5-5':'M8 14V2M3 7l5-5 5 5'}"/></svg>`;
}
function activityTime(timestamp, now=Date.now()) {
  const elapsed=Math.max(0,(now-timestamp*1000)/60000);
  if(elapsed<1)return 'Just now';
  if(elapsed<60)return `${Math.floor(elapsed)} min ago`;
  if(elapsed<24*60)return `${Math.floor(elapsed/60)} hr ago`;
  const date=new Date(timestamp*1000);
  return date.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',...(date.getFullYear()!==new Date(now).getFullYear()?{year:'numeric'}:{})});
}
function orderedRuns(runs) { return [...runs].sort((a,b)=>(b.created ?? b.updated ?? 0)-(a.created ?? a.updated ?? 0)); }
function defaultRun(runs) {
  const ordered=orderedRuns(runs);
  return ordered.find(r=>r.active) || ordered.find(r=>r.experiments>0 || Number.isFinite(r.best)) || ordered[0];
}
async function resolveRunTimes(runs) {
  // Older running servers omit creation time for legacy runs. Read their first
  // event so the browser can order them correctly without restarting the server.
  await Promise.all(runs.filter(r=>!Number.isFinite(r.created)).map(async r=>{
    try {
      const detail=await api(projectAPI(`/runs/${encodeURIComponent(r.id)}`));
      r.created=detail.run.created ?? detail.events.find(e=>Number.isFinite(e.ts))?.ts ?? r.updated ?? 0;
    } catch {r.created=r.updated ?? 0;}
  }));
  return runs;
}
function getRound(number) {
  if (!state.rounds.has(number)) state.rounds.set(number, {round:number,parent:0,curves:[],seeds:{},diffs:[],errors:[],verdict:'testing'});
  return state.rounds.get(number);
}
function ledgerRows() { return [...state.rounds.values()].filter(r => r.round===0 ? Number.isFinite(r.value) : Boolean(r.hypothesis || r.verdict !== 'testing')).sort((a,b)=>a.round-b.round); }
function apply(event, paint=true) {
  if(event.seq && event.seq<=state.watermark)return;
  if(event.seq)state.watermark=event.seq;
  if(Array.isArray(event.hypotheses))state.data.hypotheses=event.hypotheses;
  const r = getRound(event.round ?? 0);
  if(event.error) {
    r.errorSeeds ??= {};
    const key=errorSummary(event.error), seeds=r.errorSeeds[key] ??= [];
    const active=Object.entries(r.seeds).filter(([,v])=>v==='running').map(([seed])=>seed);
    for(const seed of event.seed===undefined?active:[String(event.seed)])if(!seeds.includes(seed))seeds.push(seed);
  }
  switch(event.event) {
    case 'artifact': if(!state.artifacts.some(a=>a?.id===event.artifact?.id && a?.revision===event.artifact?.revision))state.artifacts.push(event.artifact); break;
    case 'experiment_record': r.record=event.record; break;
    case 'result_record': r.resultRecord=event.record; break;
    case 'research_ended': state.run.evidenceEnded=true;if(paint && !state.replay)refreshFiles(); break;
    case 'baseline': if(paint && !state.replay)refreshFiles(); Object.assign(r,{change:'Original settings',hypothesis:'Baseline',value:event.result?.[event.metric || state.data.task.metric],verdict:'baseline',result:event.result}); break;
    case 'idea':
      Object.assign(r,{parent:state.best},event); if(state.follow) state.selected=r.round;
      if(!state.replay && ['human','you'].includes(event.source) && state.data.ideas) {
        const i=state.data.ideas.findIndex(idea=>idea.hypothesis===event.hypothesis && (!idea.ts || event.ts>=idea.ts));
        if(i>=0){state.data.ideas.splice(i,1);renderQueue();}
      }
      break;
    case 'code': case 'fix': r.diffs.push({kind:event.event,diff:event.diff || ''}); r.verdict='testing'; break;
    case 'run_start': r.seeds[event.seed]='running'; if(state.follow) state.selected=r.round; break;
    case 'curve': r.curves.push(event.point ? {...event,...event.point} : event);if(paint && state.legacy && !state.replay && Date.now()-(state.legacyRefresh || 0)>1000){state.legacyRefresh=Date.now();refreshFiles();} break;
    case 'run_end': if(paint && !state.replay)refreshFiles(); r.seeds[event.seed]='complete'; r.seedResults ??= []; r.seedResults.push({seed:event.seed,...event.result}); break;
    case 'run_error': r.errors.push(event.error || 'Experiment failed'); markCrashed(r,event.seed); break;
    case 'edit_rejected': r.errors.push('Edit rejected: '+event.reason); break;
    case 'hypothesis_error': case 'idea_error': r.errors.push(event.error); notice(`Experiment ${event.round}: ${event.error}`); break;
    case 'result': Object.assign(r,event,{verdict:experimentVerdict(event.verdict)}); if(event.reason && !r.errors.includes(event.reason))r.errors.push(event.reason); if(event.error && r.errors.at(-1)!==event.error)r.errors.push(event.error); if(event.verdict==='crashed')markCrashed(r,event.seed); r.value ??= event.result?.[state.data.task.metric]; if(experimentVerdict(event.verdict)==='improved') state.best=r.round; if(paint && !state.replay)refreshFiles(); break;
    case 'goal_reached': notice('Goal reached. The measured result is recorded.'); break;
    case 'done': {
      const completedLive=paint && state.run.active && !state.replay;
      state.run.active=false; state.run.finished=true; state.data.runs.forEach(run=>{if(run.id===state.run.id)run.active=false;}); state.source?.close();
      if(completedLive)notice(event.reason || 'Run complete. Explore the ledger and artifacts in Results.',6000);
      if(paint) refreshFiles(); break;
    }
  }
  if(paint && ['baseline','result'].includes(event.event))r.arrived=Date.now();
  if(paint) scheduleRender();
}
function markCrashed(r, seed) {
  r.verdict='crashed';
  if(seed!==undefined)r.seeds[seed]='crashed';
  for(const [key,status] of Object.entries(r.seeds))if(status==='running')r.seeds[key]='crashed';
}
function errorSummary(error) {
  const lines=String(error || '').trim().split('\n').map(s=>s.trim()).filter(Boolean);
  return [...lines].reverse().find(line=>/^[\w.]*(?:Error|Exception|Interrupt|Exit)(?::|$)/.test(line)) || lines.at(-1) || 'Experiment failed';
}
function errorMarkup(error,scope='') {
  return `<p class="error error-summary" title="${esc(errorSummary(error))}">${esc(errorSummary(error))}${scope?' · '+esc(scope):''}</p><details class="error-details"><summary>Show error</summary><pre tabindex="0" aria-label="Experiment error">${esc(error)}</pre></details>`;
}
function groupedErrors(r) {
  const groups=new Map();
  for(const error of r.errors) {
    const key=errorSummary(error), previous=groups.get(key);
    if(!previous || String(error).length>String(previous).length)groups.set(key,error);
  }
  return [...groups].map(([key,error])=>{
    const seeds=r.errorSeeds?.[key] || [], all=state.data.task.seeds.every(s=>seeds.includes(String(s)));
    return {error,scope:seeds.length?(all && seeds.length>1?'on all seeds':'on seed'+(seeds.length>1?'s ':' ')+seeds.join(', ')):''};
  });
}
function domainChip(domain) { return domain ? `<span class="domain-chip">${esc(domain)}</span>` : ''; }
function scheduleRender() {
  if(paintPending) return;
  paintPending=true;
  (typeof requestAnimationFrame==='function' ? requestAnimationFrame : setTimeout)(() => {paintPending=false; renderRun();});
}
function parseRoute(hash=location.hash) {
  const [path,query=''] = (hash || '#/').slice(1).split('?');
  if(path==='/' || !path) return {home:true};
  const match=path.match(/^\/p\/([a-z0-9]+(?:-[a-z0-9]+)*)(?:\/(discuss|experiments|results))?\/?$/);
  if(!match) return {invalid:true};
  const params=new URLSearchParams(query);
  const replay=params.get('replay')==='1';
  const speed=Number(params.get('speed') || 8);
  return {slug:match[1],stage:match[2] || 'discuss',run:params.get('run'),round:params.has('round') ? Math.max(0,Number(params.get('round')) || 0) : null,replay,speed:Number.isFinite(speed) && speed>0 && speed<=1000 ? speed : 8};
}
function stageURL(stage, overrides={}) {
  const params=new URLSearchParams();
  const run=overrides.run ?? state.run?.id;
  if(run) params.set('run',run);
  const replay=overrides.replay ?? state.replay;
  if(replay) {params.set('replay','1'); params.set('speed',String(overrides.speed ?? state.speed));}
  if(overrides.round !== undefined) params.set('round',String(overrides.round));
  return `#/p/${state.slug}/${stage}${params.size ? '?'+params : ''}`;
}
function markdown(text) {
  // Escape first. No HTML, links, images, or executable attributes are accepted.
  const lines=esc(text).split('\n'); let html='', para=[], list=null, code=null;
  const inline=s=>s.replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/\[([^\]]+)\]\(([^\s)]+)\)/g,(all,label,url)=> /^(?:https?:\/\/|\/(?![\/\\]))/i.test(url) && !/[\\\x00-\x20]/.test(url) ? `<a href="${url}" rel="noopener noreferrer">${label}</a>` : label);
  const flush=()=>{if(para.length){html+='<p>'+inline(para.join(' '))+'</p>';para=[];} if(list){html+=`</${list}>`;list=null;}};
  for(const line of lines) {
    if(line.startsWith('```')) {flush(); if(code!==null){html+='<pre><code>'+code.join('\n')+'</code></pre>';code=null;}else code=[]; continue;}
    if(code!==null){code.push(line);continue;}
    const item=line.match(/^\s*(?:([-*])|\d+\.)\s+(.*)$/);
    if(item){if(para.length){html+='<p>'+inline(para.join(' '))+'</p>';para=[];}const type=item[1] ? 'ul' : 'ol';if(list!==type){if(list)html+=`</${list}>`;list=type;html+=`<${type}>`;}html+='<li>'+inline(item[2])+'</li>';}
    else if(!line.trim()) flush();
    else {if(list){html+=`</${list}>`;list=null;}para.push(line.replace(/^#{1,6}\s+/,''));}
  }
  flush(); if(code!==null)html+='<pre><code>'+code.join('\n')+'</code></pre>';
  return html;
}
function renderMath(el) {
  // KaTeX is self-hosted under /vendor/katex; code and pre blocks are left alone.
  if(!el || typeof window === 'undefined' || typeof window.renderMathInElement !== 'function') return;
  try { window.renderMathInElement(el, {delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false},{left:'$',right:'$',display:false}], throwOnError:false, strict:'ignore'}); } catch (_) {}
}
function starterPrompts(task=state.data?.task) {
  return [`Why does the baseline fail on ${metricLabel(task).toLowerCase()}?`, 'What should we try first?', 'What would refute our best hypothesis?'];
}
function goalDescription(program='') {
  const lines=program.split('\n').map(line=>line.trim()).filter(Boolean);
  return lines.find(line=>!/^#{1,6}\s/.test(line)) || lines[0]?.replace(/^#{1,6}\s+/, '') || '';
}
function projectHistory(runs, task) {
  let best=null;
  const progress=[];
  for(const run of [...runs].sort((a,b)=>(a.created || 0)-(b.created || 0))) {
    for(const e of run.events || []) {
      if(!['baseline','result'].includes(e.event))continue;
      const candidate=e.event==='baseline' ? e.result?.[e.metric || task.metric] : experimentVerdict(e.verdict)==='improved' ? e.value ?? e.result?.[task.metric] : null;
      if(Number.isFinite(candidate))best=best===null ? candidate : task.direction==='min' ? Math.min(best,candidate) : Math.max(best,candidate);
      if(best!==null)progress.push(best);
    }
  }
  return progress;
}
function sparkline(values=[], task={}) {
  const points=values.filter(Number.isFinite);
  if(!points.length)return '<span class="spark-empty muted">Awaiting baseline</span>';
  const low=Math.min(...points), high=Math.max(...points), range=high-low;
  const x=i=>5+(points.length===1 ? 100 : i/(points.length-1)*100), y=v=>range ? 31-(v-low)/range*24 : 19;
  let path=`M${x(0)},${y(points[0])}`;
  for(let i=1;i<points.length;i++)path+=` H${x(i)} V${y(points[i])}`;
  const label=`Running best, ${task.direction==='min'?'lower':'higher'} is better: ${points.map(v=>format(v,task)).join(', ')}`;
  return `<svg class="sparkline" viewBox="0 0 110 40" role="img" aria-label="${esc(label)}"><title>${esc(label)}</title><path d="M5 35H105" class="spark-rule"/><path d="${path}" class="spark-path"/><circle cx="${x(points.length-1)}" cy="${y(points.at(-1))}" r="3"/></svg>`;
}
async function projectSummary(project) {
  if(project.status && project.status!=='Running' && Array.isArray(project.progress) && typeof project.description==='string')return;
  // Active and older servers need live event counts rather than a finished-run snapshot.
  const path=`/api/projects/${encodeURIComponent(project.slug)}`;
  const detail=await api(path);
  project.task=project.task || detail.task;
  project.description=goalDescription(detail.program);
  project.status=detail.runs.some(r=>r.active)?'Running':detail.runs.length?'Finished':'No runs';
  const runs=await Promise.all(detail.runs.map(async r=>{
    const data=await api(`${path}/runs/${encodeURIComponent(r.id)}`);
    return {...r,...data.run,events:data.events,created:r.created ?? data.run.created ?? data.events.find(e=>Number.isFinite(e.ts))?.ts ?? r.updated ?? 0};
  }));
  project.experiments=runs.reduce((sum,r)=>sum+r.events.filter(e=>e.event==='result').length,0);
  project.hypotheses=detail.hypotheses?.length ?? new Set(runs.flatMap(r=>r.events.filter(e=>e.event==='idea').map(e=>e.hypothesis_id || e.hypothesis || e.change))).size;
  project.progress=projectHistory(runs,project.task);
  project.best=project.progress.at(-1) ?? null;
  project.updated=Math.max(project.updated || 0,...runs.map(r=>r.events.at(-1)?.ts ?? r.updated ?? 0));
}
function renderProjects() {
  const focusedDomain=document.activeElement?.dataset?.domain;
  const domains=[...new Set(state.projects.map(p=>p.task?.domain).filter(Boolean))].sort();
  if(!domains.includes(state.domain))state.domain='';
  $('domain-filters').hidden=!domains.length;
  $('domain-filters').innerHTML=['',...domains].map(d=>`<button type="button" class="secondary domain-filter" data-domain="${esc(d)}" aria-pressed="${state.domain===d}">${esc(d || 'All domains')}</button>`).join('');
  const projects=state.projects.filter(p=>!state.domain || p.task?.domain===state.domain).sort((a,b)=>b.updated-a.updated);
  $('project-list').innerHTML=projects.length ? projects.map(p=>`<article class="project-row"><div class="project-intro"><a href="#/p/${esc(p.slug)}">${esc(p.title)}</a>${domainChip(p.task?.domain)}<p class="project-description muted" title="${esc(p.description || '')}">${esc(p.description || 'Open the project to review its research goal.')}</p></div><div class="project-template"><span class="muted">Template</span><span>${esc(p.template)}</span><div><span class="project-status ${p.status==='Running'?'is-running':''}">${esc(p.status || 'Status unavailable')}</span></div></div><div><span class="muted">Research</span><span class="project-number project-counts">${p.hypotheses ?? 0} hypotheses · ${p.experiments} experiments</span></div><div class="project-metric"><div><span class="muted">Best ${esc(metricLabel(p.task))} ${directionMark(p.task)}</span><strong>${esc(format(p.best,p.task))}</strong></div>${sparkline(p.progress,p.task)}</div><div><span class="muted">Last activity</span><time datetime="${new Date(p.updated*1000).toISOString()}">${esc(activityTime(p.updated))}</time></div></article>`).join('') : '<div class="empty project-empty"><h3>Your next question starts here</h3><p>Create a project from a template to keep your goal, hypotheses, experiments, and evidence together.</p><button type="button" class="secondary" id="first-project">Create your first project</button></div>';
  if(!projects.length)$('first-project').onclick=()=>$('new-project').onclick();
  if(focusedDomain!==undefined)[...$('domain-filters').querySelectorAll('[data-domain]')].find(button=>button.dataset.domain===focusedDomain)?.focus({preventScroll:true});
}
$('domain-filters').onclick=event=>{const button=event.target.closest('[data-domain]');if(button){state.domain=button.dataset.domain;renderProjects();}};
function renderTemplateCards() {
  $('template-cards').innerHTML=state.templates.map(t=>`<label class="template-card"><input type="radio" name="template-choice" value="${esc(t.slug)}" ${t.slug===$('template').value?'checked':''}><span><strong>${esc(t.title)}</strong><span class="muted">${esc(metricLabel(t))}${t.unit?' · '+esc(t.unit):''} ${directionMark(t)} · ${t.direction==='min'?'Lower':'Higher'} is better</span><span class="muted">${esc(t.editable)}</span></span></label>`).join('');
}
function deltaClass(value, task=state.data?.task) {
  return !Number.isFinite(value) || value===0 ? '' : (task.direction==='min' ? value<0 : value>0) ? 'supported' : 'refuted';
}
function tokenizeCode(source) {
  // Tokenize raw source once; every emitted token is escaped at the render boundary.
  const pattern=/("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\[\s\S]|[^"\\\n])*"?|'(?:\\[\s\S]|[^'\\\n])*'?|#[^\n]*|\b(?:0[xX][\da-fA-F_]+|\d+(?:_\d+)*(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+)\b|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b)/g;
  const tokens=[];let offset=0;
  for(const m of source.matchAll(pattern)) {
    if(m.index>offset)tokens.push({text:source.slice(offset,m.index),kind:''});
    const text=m[0], kind=text[0]==='#'?'comment':/^['"]/.test(text)?'string':/^[\d.]/.test(text)?'number':'keyword';
    tokens.push({text,kind});offset=m.index+text.length;
  }
  if(offset<source.length)tokens.push({text:source.slice(offset),kind:''});
  return tokens;
}
function codeMarkup(source) {
  const lines=[''];
  for(const token of tokenizeCode(source)) {
    token.text.split('\n').forEach((part,i)=>{if(i)lines.push('');lines[lines.length-1]+=token.kind?`<span class="syntax-${token.kind}">${esc(part)}</span>`:esc(part);});
  }
  return lines.map((line,i)=>`<span class="code-line"><span class="line-number" aria-hidden="true">${i+1}</span><span>${line || ' '}</span></span>`).join('\n');
}
function diffMarkup(source) {
  let added=0,removed=0,inHunk=false;
  const html=source.split('\n').map(line=>{
    const hunk=line.startsWith('@@');if(hunk)inHunk=true;
    const header=!inHunk && /^(?:\+\+\+|---)\s/.test(line);
    const kind=header || hunk ? 'diff-meta' : line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : '';
    if(kind==='add')added++;if(kind==='del')removed++;
    return `<span class="${kind}">${esc(line) || ' '}</span>`;
  }).join('\n');
  return {html,added,removed};
}
async function loadHome() {
  const [projects,templates]=await Promise.all([api('/api/projects'),api('/api/templates')]);
  state.projects=projects.projects; state.templates=templates.templates;
  $('template').innerHTML=state.templates.map(t=>`<option value="${esc(t.slug)}">${esc(t.title)}</option>`).join('');
  if(!$('template').value)$('template').value=state.templates[0]?.slug || '';
  $('create-project').disabled=!state.templates.length;
  templateNote();renderTemplateCards();
  const summaries=await Promise.allSettled(state.projects.map(projectSummary));
  if(summaries.some(r=>r.status==='rejected'))notice('Some project summaries could not load. Open a project to retry.');
  renderProjects();
}
function templateNote() {
  const t=state.templates.find(t=>t.slug===$('template').value) || state.templates[0];
  $('template-note').textContent=t ? `${metricLabel(t)}${t.unit ? ' ('+t.unit+')' : ''} · ${t.direction==='min'?'lower':'higher'} is better · editable file: ${t.editable}` : 'No templates found. Add a template folder to get started.';
}
async function route() {
  const target=parseRoute();
  if(target.invalid){notice('Page not found. Choose a project.');location.hash='#/';return;}
  const oldSlug=state.slug;
  const generation=++state.generation;
  if(target.home){state.source?.close();state.source=null;state.slug=null;state.loadingRun=null;$('chat-panel').hidden=true;$('home').hidden=false;$('project').hidden=true;$('title').textContent='Projects';$('project-domain').hidden=true;document.title='Projects · Research Loop';try{await loadHome();}catch(e){notice(e.message);$('project-list').textContent='Projects could not load. Reload the page to retry.';}return;}
  $('home').hidden=true;$('project').hidden=false;
  const changed=target.slug!==oldSlug;
  if(changed){state.programDirty=false;state.programEditing=false;state.chatDraft=null;state.source?.close();state.source=null;state.run=null;state.data=null;state.rounds=new Map();state.files=[];state.artifacts=[];state.watermark=0;state.protocol=null;state.loadingRun=null;artifactKey='';artifactGeneration++;$('program').value='';$('title').textContent='Loading project…';}
  state.slug=target.slug;state.stage=target.stage;
  try {
    if(changed || !state.data) {
      const data=await api(projectAPI()); if(generation!==state.generation)return;
      await resolveRunTimes(data.runs);if(generation!==state.generation)return;
      state.data=data;
      $('project-domain').textContent=data.task.domain || '';$('project-domain').hidden=!data.task.domain;
      $('title').textContent=data.project.title; $('title').title=data.project.title;
      document.title=data.project.title+' · Research Loop';
      $('program').value=data.program;
      $('round-count').value=data.rounds;
      $('model').innerHTML=data.models.map(m=>`<option value="${esc(m)}">${esc(modelName(m))}</option>`).join('');
      $('model').value=data.models.includes(data.project.model) ? data.project.model : data.default_model;
      $('task-summary').innerHTML=`<strong>${esc(metricLabel())}</strong> · ${esc(data.task.direction==='min'?'lower is better':'higher is better')}<br>Noise threshold: ${format(data.task.noise)}<br>Seeds: ${esc(data.task.seeds.join(', '))}<br>Editable file: ${esc(data.task.editable)}${Number.isFinite(data.task.goal) ? '<br>Goal: '+format(data.task.goal) : ''}`;
      renderQueue();renderProgram();renderRunPicker();
    } else if(!state.chatBusy) {
      // A discussion saved in another tab must appear in either discussion host.
      const data=await api(projectAPI());if(generation!==state.generation)return;
      await resolveRunTimes(data.runs);if(generation!==state.generation)return;
      state.data.hypotheses=data.hypotheses;state.data.ideas=data.ideas;
      state.data.discussion=data.discussion;
      state.data.runs=data.runs;renderRunPicker();
    }
    const runId=target.run || defaultRun(state.data.runs)?.id;
    const newRun=runId!==state.run?.id || state.replay!==target.replay || state.speed!==target.speed;
    state.replay=target.replay; state.speed=target.speed;
    if(newRun || (runId && state.loadingRun)) await loadRun(runId,generation,target.round);
    if(generation!==state.generation)return;
    if(!runId){state.run=null;state.rounds=new Map();}
    if(target.round!==null){state.selected=target.round;state.follow=false;}
    showStage(); renderRun();
  } catch(e){if(generation===state.generation){notice(e.message);if(!state.data){$('project').hidden=true;$('home').hidden=false;$('project-list').innerHTML='<p class="empty">Project could not load. <a href="#/">Return to Projects</a>.</p>';}}}
}
function modelName(key) { return key==='q35' ? '35B · q35' : key==='q27' ? '27B · q27' : key; }
function showStage() {
  for(const name of ['discuss','experiments','results']) {
    $(name).hidden=state.stage!==name;
    $(`${name}-link`).href=stageURL(name);
    if(state.stage===name)$(`${name}-link`).setAttribute('aria-current','page');else $(`${name}-link`).removeAttribute('aria-current');
  }
  $('run-toolbar').hidden=state.stage==='discuss';
  $('chat-panel').hidden=state.stage==='results';
  (state.stage==='experiments' ? $('rail-discussion') : $('discussion-host')).append($('chat-panel'));
  // Scroll after mounting in the visible stage; hidden hosts have no scroll height.
  if(state.stage!=='results')renderConversation();
}
function renderRunPicker() {
  $('runs').innerHTML=orderedRuns(state.data.runs).map(r=>`<option value="${esc(r.id)}">${esc(r.id)}${r.active?' · running':''}</option>`).join('') || '<option value="">No runs yet</option>';
  if(state.run)$('runs').value=state.run.id;
}
async function loadRun(id,generation,selected=null) {
  state.source?.close();state.source=null;state.rounds=new Map();state.selected=selected??0;state.follow=selected===null;state.best=0;state.files=[];state.artifacts=[];state.watermark=0;artifactKey='';artifactGeneration++;
  if(!id){state.run=null;return;}
  state.loadingRun=id;
  const detail=await api(projectAPI(`/runs/${encodeURIComponent(id)}`));
  if(generation!==state.generation)return;
  state.loadingRun=null;state.run=detail.run;state.files=detail.files;state.artifacts=state.replay?[]:(detail.artifacts || []);state.protocol=detail.protocol;state.legacy=detail.legacy;$('run-report').hidden=true;$('run-report').innerHTML='';for(const host of [$('experiment-artifacts'),$('result-artifacts'),$('run-artifacts')])host.dataset.key='';renderRunPicker();
  if(!state.replay && (detail.protocol===1 || !state.run.active))for(const event of detail.events)apply(event,false);
  state.watermark=state.replay?0:(detail.watermark || state.watermark);
  if(state.run.active || state.replay) {
    const suffix=state.replay ? `?replay=1&speed=${state.speed}` : `?after=${state.watermark}`;
    const source=new EventSource(projectAPI(`/runs/${encodeURIComponent(id)}/events${suffix}`));state.source=source;
    source.onmessage=message=>{if(state.source!==source)return;apply(JSON.parse(message.data));};
    source.addEventListener('closed',()=>{if(state.source!==source)return;source.close();state.run.active=false;state.data.runs.forEach(run=>{if(run.id===state.run.id)run.active=false;});renderRun();notice('Run closed. Inspect loop.log if it ended before a verdict.');});
    source.onerror=()=>{if(state.source===source && state.run.active)notice('Event connection interrupted. Reconnecting…');};
  }
}
async function refreshFiles() {
  if(!state.run)return;
  const slug=state.slug, id=state.run.id;
  try {
    const data=await api(projectAPI(`/runs/${encodeURIComponent(id)}`));
    if(slug!==state.slug || id!==state.run?.id)return;
    state.files=data.files;
    state.artifacts=[...new Map([...(state.artifacts || []),...(data.artifacts || [])].map((a,i)=>[(a?.id || 'invalid-'+i)+':'+a?.revision,a])).values()];
    const run=state.data.runs.find(r=>r.id===id);if(run)Object.assign(run,data.run,{active:state.run.finished ? false : data.run.active});
    const project=await api(projectAPI());
    if(slug!==state.slug || id!==state.run?.id)return;
    state.data.hypotheses=project.hypotheses;state.data.ideas=project.ideas;
    renderRunPicker();renderQueue(); renderResults(); controls();
  }catch(e){notice(e.message);}
}
function hypothesisStatus(experiments, task=state.data.task) {
  const verdicts=experiments.map(e=>experimentVerdict(e.verdict));
  if(verdicts.includes('improved') && verdicts.includes('worse'))return 'inconclusive';
  if(verdicts.includes('improved'))return 'supported';
  const finished=verdicts.filter(v=>['worse','no effect','crashed'].includes(v));
  if(finished.length>=(task.experiments_per_hypothesis ?? 2))return finished.includes('worse')?'refuted':'inconclusive';
  return experiments.length?'testing':'untested';
}
function legacyHypothesisId(statement) {
  let hash=14695981039346656037n;
  for(const byte of new TextEncoder().encode(statement))hash=BigInt.asUintN(64,(hash^BigInt(byte))*1099511628211n);
  return 'legacy-'+hash.toString(16).padStart(16,'0');
}
function hypothesisGroups() {
  const groups=new Map((state.data?.hypotheses || []).map(h=>[h.id,{...h,experiments:(h.experiments || []).filter(e=>e.run!==state.run?.id)}]));
  for(const r of ledgerRows().filter(r=>r.round>0)) {
    const statement=r.hypothesis || changeLabel(r);
    let id=r.hypothesis_id || [...groups.values()].find(h=>h.statement===statement)?.id;
    if(!id)id=legacyHypothesisId(statement);
    if(!groups.has(id))groups.set(id,{id,statement,why:r.why || '',prediction:r.prediction || '',source:r.source==='human'?'you':r.source || 'model',experiments:[]});
    groups.get(id).experiments.push({...r,verdict:r.event==='result'?r.verdict:'testing',run:state.run?.id,hypothesis_id:id});
  }
  // Older servers still expose a project queue, including the loop already in memory.
  for(const [i,entry] of (state.data?.ideas || []).entries())if(![...groups.values()].some(h=>h.statement===entry.hypothesis))groups.set('pending-'+i,{id:'pending-'+i,statement:entry.hypothesis,source:'you',experiments:[]});
  return [...groups.values()].map((h,i)=>({...h,label:h.label || 'H'+(i+1),status:state.protocol===1 && h.status ? h.status : hypothesisStatus(h.experiments)}));
}
function hypothesisFor(r) { return hypothesisGroups().find(h=>h.experiments.some(e=>e.round===r.round && e.run===state.run?.id)); }
function hypothesisColor(h) {
  const palette=['#285e89','#705187','#87601f','#176345','#984c46','#376b70'];
  const index=Array.from(h?.id || '').reduce((n,c)=>n+c.charCodeAt(0),0);
  return palette[index%palette.length];
}
function hypothesisCaption(r) {
  const h=hypothesisFor(r);if(!h)return 'Baseline';
  const index=h.experiments.findIndex(e=>e.round===r.round && e.run===state.run?.id)+1;
  const limit=state.data.task.experiments_per_hypothesis ?? 2, total=h.id.startsWith('legacy-')?Math.max(limit,h.experiments.length):limit;
  return `${h.label || h.id} · ${h.statement} · experiment ${index} of ${total}`;
}
function countLabel(n,word){return `${n} ${word}${n===1?'':'s'}`;}
function hypothesisRule() { return `Complete improvements support a claim; conflicting complete results are inconclusive. After ${state.data.task.experiments_per_hypothesis ?? 2} experiments or when the run ends: worse evidence refutes; no effect, missing or crashed evidence alone is inconclusive.`; }
function pill(status) { return `<span class="verdict-pill ${statusClass(status)}">${esc(status)}</span>`; }
function changeMarkup(r) {
  const label=changeLabel(r), code=/\b(?:for|return|def|import)\b|\b[A-Z][A-Z_]+\b|(?:np|scipy)\.|[=(){}]|\w+_\w+/.test(label);
  return `<span class="ledger-change${code?' code-change':''}" title="${esc(label)}">${esc(label)}</span>`;
}
function experimentLink(e) {
  return `<a href="${stageURL('results',{run:e.run,round:e.round,replay:e.run===state.run?.id && state.replay})}" title="${esc(changeLabel(e))}">Experiment ${e.round} · ${changeMarkup(e)}</a>`;
}
function renderQueue() {
  const claims=hypothesisGroups();
  $('queue-count').textContent=String(claims.length);
  $('queue').innerHTML=claims.length ? claims.map(h=>`<li><strong>${esc(h.label || h.id)}</strong> ${pill(h.status)}<span class="hypothesis-statement">${esc(h.statement)}</span><small class="muted">${countLabel(h.experiments.length,'experiment')} · ${esc(h.source==='discussion'?'discussion':h.source==='you'?'you':'model')}</small></li>`).join('') : '<li class="queue-empty muted">A hypothesis is a claim; each will be tested by one or more experiments.</li>';
  $('hypothesis-rule').textContent=hypothesisRule();
  const openGroups=[...$('experiment-groups').querySelectorAll('details[open]')].map(d=>d.dataset.group), focusedExperiment=document.activeElement?.dataset?.experiment;
  $('experiment-groups').innerHTML=claims.length ? claims.map(h=>`<details class="hypothesis-group" data-group="${esc(h.id)}"${h.experiments.some(e=>e.round===state.selected && e.run===state.run?.id)?' open':''}><summary><span class="hypothesis-id" style="color:${hypothesisColor(h)}">${esc(h.label || h.id)}</span> ${pill(h.status)}<span class="hypothesis-statement">${esc(h.statement)}</span><small class="muted">${countLabel(h.experiments.length,'experiment')}</small></summary><ol>${h.experiments.map(e=>`<li${e.round===state.selected && e.run===state.run?.id?' class="is-selected"':''}>${e.run===state.run?.id?`<button type="button" class="text-button" data-experiment="${e.round}" aria-pressed="${e.round===state.selected}">Experiment ${e.round} · ${changeMarkup(e)}</button>`:experimentLink(e)} ${pill(e.verdict || 'testing')}</li>`).join('') || '<li class="muted">Ready for its first experiment.</li>'}</ol></details>`).join('') : '<p class="muted">Hypotheses and their experiments appear here as testing begins.</p>';
  for(const d of $('experiment-groups').querySelectorAll('details'))if(openGroups.includes(d.dataset.group))d.open=true;
  if(focusedExperiment!==undefined)$('experiment-groups').querySelector(`[data-experiment="${focusedExperiment}"]`)?.focus({preventScroll:true});
}
function renderConversation() {
  const messages=[...(state.data?.discussion || [])]; if(state.chatDraft)messages.push(state.chatDraft);
  $('conversation').innerHTML=messages.length ? messages.map((m,i)=>`<article class="chat ${m.role==='user'?'user':'assistant'}${m.pending?' streaming':''}"><header>${m.role==='user' ? 'You' : esc(modelName(m.model || $('model').value))}${m.role==='assistant'?'<span class="model-label">Local model</span>':''}</header><div class="message-content">${markdown(m.content || 'Thinking…')}</div>${m.pending?'<span class="streaming-caret" aria-label="Responding"></span>':''}${m.role==='assistant' && !m.pending ? `<button class="secondary queue-action" type="button" data-queue="${i}">Add as hypothesis</button>` : ''}</article>`).join('') : `<div class="discussion-empty"><h3>What do you want to test?</h3><p>Explore a few approaches with your local model. Add a testable hypothesis when you’re ready to measure it.</p><div class="starter-prompts">${starterPrompts().map((p,i)=>`<button type="button" class="secondary" data-prompt="${i}">${esc(p)}</button>`).join('')}</div><p class="muted">Your discussion stays with this project.</p></div>`; renderMath($('conversation'));
  $('conversation').scrollTop=messages.length ? $('conversation').scrollHeight : 0;
  controls();
}
function renderProgram() {
  $('program-view').innerHTML=markdown($('program').value); renderMath($('program-view'));
  $('program-view').hidden=Boolean(state.programEditing);
  $('program').hidden=!state.programEditing;
  $('save-program').hidden=!state.programEditing;
  $('edit-program').textContent=state.programEditing?'Preview':'Edit';
  $('edit-program').setAttribute('aria-expanded',String(Boolean(state.programEditing)));
}
function controls() {
  const active=Boolean(state.run?.active), anyActive=state.data?.runs.some(r=>r.active) || active;
  $('start').disabled=state.replay || anyActive || !state.data;
  $('round-count').disabled=state.replay || anyActive;
  $('pause').disabled=state.replay || !active || state.run?.paused;
  $('resume').disabled=state.replay || !active || !state.run?.paused;
  $('stop').disabled=state.replay || !active;
  for(const id of ['run-actions','run-model','run-control-note'])$(id).hidden=!active || state.replay;
  $('send').disabled=state.replay || state.chatBusy || !state.data;
  $('queue-message').disabled=state.replay || !state.data;
  $('add-hypothesis').disabled=state.replay || !state.data;
  for(const id of ['hypothesis-statement','hypothesis-why','hypothesis-prediction'])$(id).readOnly=state.replay;
  $('model').disabled=state.chatBusy || state.replay;
  $('save-program').disabled=state.replay || anyActive;
  $('edit-program').disabled=state.replay || anyActive;
  $('program').readOnly=state.replay || anyActive;
  $('goal-state').textContent=anyActive ? 'Goal fixed for the active run' : state.replay ? 'Replay is read only' : state.programDirty ? 'Unsaved changes' : '';
  $('replay-run').disabled=!state.run || (active && !state.replay);
  $('replay-run').textContent=state.replay ? 'Exit replay' : 'Replay';
  $('replay').hidden=!state.replay;
  $('replay').textContent=`Replay · ${state.speed}× speed`;
  const latest=getRound(Math.max(0,...state.rounds.keys())), seed=Object.entries(latest.seeds).find(([,v])=>v==='running')?.[0];
  $('run-state').textContent=state.replay ? 'Replaying recorded events' : active ? state.run.paused ? 'Pause requested' : `Running · ${latest.round ? 'experiment '+latest.round+' of '+(state.run.rounds || $('round-count').value || 10) : 'baseline'}${latest.round && hypothesisFor(latest)?' · '+(hypothesisFor(latest).label || hypothesisFor(latest).id):''}${seed!==undefined?' · seed '+seed:''}` : state.run ? 'Run finished' : 'Ready for experiments';
  $('run-state').className=active && !state.run.paused && !state.replay ? 'is-live' : '';
  $('start-summary').textContent=`Baseline + ${$('round-count').value || 10} experiments · seeds ${state.data?.task.seeds.join(', ') || '—'} · fixed budget`;
  $('run-model').textContent=state.run ? modelName(state.run.model)+' · '+state.run.id : 'Start from Discuss when your goal is ready.';
  for(const button of $('conversation').querySelectorAll('[data-queue]'))button.disabled=state.replay;
}
function renderRun() {
  if(!state.data)return;
  controls();renderQueue();
  drawProgress();renderCurrent();renderResults();
}
function progressData(rows=ledgerRows(), task=state.data.task) {
  let best=null;
  return rows.filter(r=>Number.isFinite(r.value) || r.verdict==='crashed').map(r=>{const kept=Number.isFinite(r.value) && (r.round===0 || r.verdict==='improved');if(kept)best=r.value;return {...r,kept,runningBest:best,direction:task.direction};});
}
function progressScale(values, task) {
  if(task.unit==='%')return {low:0,high:1,step:.25,ticks:[0,.25,.5,.75,1]};
  let low=Math.min(...values), high=Math.max(...values);
  const padding=(high-low)*.08 || Math.abs(high)*.05 || 1;
  low-=padding;high+=padding;
  const rough=(high-low)/4, power=10**Math.floor(Math.log10(rough));
  const step=([1,2,5,10].find(n=>n*power>=rough) || 10)*power;
  const first=Math.floor(low/step),last=Math.ceil(high/step);
  low=first*step;high=last*step;
  const ticks=Array.from({length:last-first+1},(_,i)=>(first+i)*step);
  return {low,high,step,ticks};
}
function progressTick(value, task, step) {
  const percent=task.unit==='%';
  const digits=percent?0:Math.max(0,Math.min(12,-Math.floor(Math.log10(step))));
  return (percent?value*100:value).toLocaleString(undefined,{maximumFractionDigits:digits})+(task.unit?(percent?'%':' '+task.unit):'');
}
function drawProgress() {
  const svg=$('progress'), points=progressData();
  const focused=document.activeElement?.closest?.('[data-round]')?.dataset?.round;
  const width=Math.max(svg.parentElement?.clientWidth || svg.clientWidth || 1000,points.length*72), height=Math.max(235,svg.clientHeight || 370);
  svg.style.minWidth=width+'px';svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const task=state.data.task;
  $('progress-subtitle').textContent=`${metricLabel()} · ${task.direction==='min'?'lower':'higher'} is better · ${countLabel(ledgerRows().filter(r=>r.round>0 && r.verdict!=='testing').length,'experiment')}`;
  $('noise-caption').textContent='Noise band ±'+format(task.noise);
  if(!points.length){svg.innerHTML=`<text x="${width/2}" y="${height/2}" text-anchor="middle" fill="#526476" font-size="17">${state.run ? 'Measuring the baseline…' : 'Start experiments to trace progress, one change at a time.'}</text>`;$('best-caption').textContent='Kept changes advance the running best.';return;}
  const kept=points.filter(p=>p.kept), values=kept.flatMap(p=>[p.value,p.runningBest]).filter(Number.isFinite);
  if(Number.isFinite(task.goal))values.push(task.goal);
  if(Number.isFinite(task.reference?.value))values.push(task.reference.value);
  const noise=Number.isFinite(task.noise)?Math.max(0,task.noise):0;
  for(const p of kept)values.push(p.value-noise,p.value+noise);
  const {low,high,step,ticks}=progressScale(values.length?values:[0,1],task);
  const tickLabels=ticks.map(v=>progressTick(v,task,step));
  const left=Math.max(66,...tickLabels.map(t=>t.length*7+18)),right=width-42,top=22,bottom=height-48;
  const maximum=Math.max(1,...ledgerRows().map(r=>r.round));
  const clamp=v=>Math.min(high,Math.max(low,v));
  const x=n=>left+n/maximum*(right-left),y=v=>bottom-(clamp(v)-low)/(high-low)*(bottom-top);
  let html='';
  for(const [i,v] of ticks.entries())html+=`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7ec"/><text data-y-tick="${v}" x="${left-12}" y="${y(v)+4}" text-anchor="end" font-size="12" fill="#526476">${esc(tickLabels[i])}</text>`;
  for(let i=0;i<=Math.min(maximum,8);i++){const n=Math.round(maximum*i/Math.min(maximum,8));html+=`<text x="${x(n)}" y="${height-26}" text-anchor="middle" font-size="12" fill="#526476">${n}</text>`;}
  if(kept.length){
    const stepPath=offset=>{let path=`M${x(kept[0].round)},${y(kept[0].value+offset)}`;for(const p of kept.slice(1))path+=` H${x(p.round)} V${y(p.value+offset)}`;return path+` H${x(maximum)}`;};
    if(noise){
      let band=stepPath(noise)+` L${x(maximum)},${y(kept.at(-1).value-noise)}`;
      for(let i=kept.length-1;i>0;i--)band+=` H${x(kept[i].round)} V${y(kept[i-1].value-noise)}`;
      band+=` H${x(kept[0].round)} Z`;
      html+=`<path data-noise-band="true" d="${band}" fill="#176345" fill-opacity="0.07"><title>Noise threshold ±${esc(format(noise))} around the running best; not a confidence interval</title></path>`;
    }
    html+=`<path data-running-best="true" d="${stepPath(0)}" fill="none" stroke="#176345" stroke-width="2.5"/>`;
  }
  for(const guide of [{value:task.goal,label:'Goal'},task.reference].filter(g=>Number.isFinite(g?.value))) {
    const label=`${guide.label || 'Reference'} · ${format(guide.value)}`, gy=y(guide.value);
    const labelY=gy<top+18 ? gy+16 : gy-8;
    html+=`<g ${guide===task.reference?'data-reference="true"':'data-goal="true"'}><line x1="${left}" x2="${right}" y1="${gy}" y2="${gy}" stroke="#697684" stroke-dasharray="5 5"/><text x="${right-6}" y="${labelY}" text-anchor="end" fill="#526476" font-size="12" paint-order="stroke" stroke="white" stroke-width="4" stroke-linejoin="round"><title>${esc(label)}</title>${esc(label)}</text></g>`;
  }
  const labelBoxes=[];
  for(const p of points){
    const label=changeLabel(p), hypothesisLabel=p.round ? ((hypothesisFor(p)?.label || hypothesisFor(p)?.id)+' · '+(hypothesisFor(p)?.statement || label)) : label, short=hypothesisLabel.length>28 ? hypothesisLabel.slice(0,27)+'…' : hypothesisLabel;
    const px=x(p.round), measured=Number.isFinite(p.value), py=measured?y(p.value):bottom-6;
    const text=p.round===0?'Baseline':short, textWidth=text.length*7;
    const flip=px+8+textWidth>width-8;
    let labelY=py-14,angle=-25*(flip?-1:1);
    if(p.kept) {
      const labelX=px+(flip?-8:8), dx=textWidth*Math.cos(25*Math.PI/180)*(flip?-1:1);
      const preferAbove=labelBoxes.length%2===0;
      const candidates=[];
      for(let lane=0;lane<5;lane++)for(const above of [preferAbove,!preferAbove]) {
        const ly=py+(above?-14-lane*30:22+lane*30), dy=textWidth*Math.sin(25*Math.PI/180)*(above?-1:1);
        const box={left:Math.min(labelX,labelX+dx)-3,right:Math.max(labelX,labelX+dx)+3,top:Math.min(ly,ly+dy)-12,bottom:Math.max(ly,ly+dy)+4};
        if(box.top<10 || box.bottom>bottom || box.left<0 || box.right>width)continue;
        const overlaps=labelBoxes.filter(b=>box.left<b.right && box.right>b.left && box.top<b.bottom && box.bottom>b.top).length;
        candidates.push({ly,above,box,overlaps});
      }
      const chosen=candidates.find(c=>!c.overlaps) || candidates.sort((a,b)=>a.overlaps-b.overlaps)[0];
      if(chosen){labelY=chosen.ly;angle=(chosen.above?-25:25)*(flip?-1:1);labelBoxes.push(chosen.box);}
    }
    const fill=p.round===0?'#176345':hypothesisColor(hypothesisFor(p));
    const outside=measured && (p.value<low?'low':p.value>high?'high':'');
    const value=measured?format(p.value)+(outside?'; outside plotted range, clamped at '+outside+' edge':''):'No measurement; marker at bottom edge';
    const caret=outside ? `<path data-clamped="${outside}" d="M${px-5},${py+(outside==='high'?7:-7)} l5,${outside==='high'?-7:7} l5,${outside==='high'?7:-7}" fill="none" stroke="${p.verdict==='worse'?'#a33432':color(p.verdict)}" stroke-width="2"/>` : '';
    const age=p.arrived===undefined?Infinity:Date.now()-p.arrived;
    const live=state.run?.active && !state.run.paused && !state.replay && p.round===points.at(-1).round;
    html+=`<g id="point-${p.round}" role="button" tabindex="0" class="progress-dot" aria-label="Inspect experiment ${p.round}: ${esc(hypothesisCaption(p))}, ${esc(label)}, ${esc(value)}, ${esc(p.verdict)}" data-round="${p.round}"><title>${esc(hypothesisCaption(p))} · ${esc(label)} · ${esc(value)} · ${esc(p.verdict)}</title><circle cx="${px}" cy="${py}" r="13" fill="transparent"/>${live?`<circle class="live-point" style="animation-delay:-${Date.now()%1800}ms" cx="${px}" cy="${py}" r="11"/>`:''}${state.selected===p.round ? `<circle cx="${px}" cy="${py}" r="9" fill="none" stroke="#285e89" stroke-width="2"/>` : ''}${caret || (p.verdict==='crashed'?`<path data-crashed="true" d="M${px-4},${py-4} l8,8 m-8,0 l8,-8" fill="none" stroke="#a33432" stroke-width="2"/>`:`<circle class="${age<200?'new-dot':''}" style="animation-delay:-${age<200?age:0}ms" cx="${px}" cy="${py}" r="${p.kept?5:3.5}" fill="${fill}" stroke="${p.kept?'#154f39':p.verdict==='worse'?'#93625f':'#687b89'}" stroke-width="1.5"/>`)}${p.kept ? `${Math.abs(labelY-py)>25?`<path class="label-leader" d="M${px},${py+(labelY>py?7:-7)} V${labelY+(labelY>py?-10:8)}"/>`:''}<text data-kept-label="true" transform="translate(${px+(flip?-8:8)} ${labelY}) rotate(${angle})" text-anchor="${flip?'end':'start'}" fill="${p.round?hypothesisColor(hypothesisFor(p)):'#176345'}" font-size="12"><title>${esc(label)}</title>${esc(text)}</text>` : ''}</g>`;
  }
  html+=`<text x="${right}" y="${height-5}" text-anchor="end" fill="#526476" font-size="12">Experiment #</text>`;
  svg.innerHTML=html;if(focused!==undefined)document.getElementById('point-'+focused)?.focus({preventScroll:true});svg.setAttribute('aria-label',`${metricLabel()}, ${task.direction==='min'?'lower':'higher'} is better. ${points.filter(p=>Number.isFinite(p.value)).length} measured points. Running best ${format(kept.at(-1)?.value)}. Unmeasured crashes are marked at the bottom edge.`);
  $('best-caption').textContent='Running best: '+format(kept.at(-1)?.value)+' · '+Math.max(0,kept.length-1)+' kept improvements';
}
function meanSeries(points, field, xKey=state.data?.task.curve?.x || 'step') {
  const groups=new Map();
  for(const p of points)if(Number.isFinite(p[xKey]) && Number.isFinite(p[field])) {
    if(!groups.has(p[xKey]))groups.set(p[xKey],[]);
    groups.get(p[xKey]).push(p[field]);
  }
  return [...groups].sort((a,b)=>a[0]-b[0]).map(([x,v])=>[x,v.reduce((a,b)=>a+b,0)/v.length]);
}
function drawCurve(svg,r) {
  const task=state.data.task, config=task.curve;
  const series=(config?.series || [{key:task.metric,label:metricLabel()}]).map((s,i)=>({...s,color:i===0 ? color(r.verdict) : ['#526476','#6b6179','#766446'][(i-1)%3],data:meanSeries(r.curves,s.key)}));
  const width=Math.max(280,svg.clientWidth || 430);let height=Math.max(130,svg.clientHeight || 180);
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const data=series.flatMap(s=>s.data);
  if(!data.length){
    const measured=r.seedResults || r.result?.per_seed || [];
    svg.setAttribute('aria-label',`Experiment ${r.round}: ${measured.length?'Per-seed measurements':'No curve measurements yet'}.`);
    svg.innerHTML=`<text x="${width/2}" y="${height/2-5}" text-anchor="middle" font-size="13" fill="#526476">${measured.length ? 'Per-seed measurements' : config ? 'Waiting for curve measurements' : 'This template reports a final metric per seed.'}</text>`+measured.map((s,i)=>`<text x="${width/2}" y="${height/2+18+i*17}" text-anchor="middle" fill="#526476" font-size="12">Seed ${esc(s.seed ?? task.seeds[i] ?? i)}: ${esc(format(s[task.metric]))}</text>`).join('');return;
  }
  const parent=state.rounds.get(r.parent), primary=series[0];
  const parentData=config && primary?.style!=='points' && parent && parent!==r ? meanSeries(parent.curves,primary.key) : [];
  const all=[...data,...parentData];
  const yScale=progressScale(all.map(d=>d[1]),task);
  const {low,high}=yScale;
  const xScale=progressScale([Math.min(0,...all.map(d=>d[0])),Math.max(...all.map(d=>d[0]))],{});
  const min=Math.min(0,...all.map(d=>d[0])), max=Math.max(...all.map(d=>d[0]));
  const curveTask=task;
  const yLabels=yScale.ticks.map(v=>progressTick(v,curveTask,yScale.step));
  const left=Math.max(64,...yLabels.map(t=>t.length*7+20));
  const right=Math.max(left+40,width-42);
  const keys=[...series,...(parentData.length?[{label:'Parent',color:'#697684',dashed:true}]:[])];
  let keyX=left,keyY=10,html='';
  for(const key of keys){
    const size=key.label.length*6+36;
    if(keyX>left && keyX+size>right){keyX=left;keyY+=17;}
    html+=key.style==='points' ? `<circle cx="${keyX+8}" cy="${keyY-3}" r="3" fill="${key.color}"/>` : `<line x1="${keyX}" x2="${keyX+16}" y1="${keyY-3}" y2="${keyY-3}" stroke="${key.color}" stroke-width="2" ${key.dashed?'stroke-dasharray="4 3"':''}/>`;
    html+=`<text x="${keyX+21}" y="${keyY}" font-size="11" fill="#526476">${esc(key.label)}</text>`;keyX+=size;
  }
  // Keep a positive plot area even when many configured legend keys wrap.
  height=Math.max(height,keyY+134);svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const top=keyY+14,bottom=height-40, x=v=>left+(v-min)/(max-min || 1)*(right-left), y=v=>bottom-(v-low)/(high-low)*(bottom-top);
  for(const [i,v] of yScale.ticks.entries())html+=`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7ec"/><text data-curve-y-tick="${v}" x="${left-12}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#526476">${esc(yLabels[i])}</text>`;
  const xTicks=xScale.ticks.filter(v=>v>=min && v<=max);
  for(const a of xTicks)html+=`<text data-curve-x-tick="${a}" x="${x(a)}" y="${height-24}" text-anchor="${a===min?'start':a===max?'end':'middle'}" fill="#526476" font-size="11">${esc(progressTick(a,{},xScale.step))}</text>`;
  if(parentData.length)html+=`<polyline points="${parentData.map(([a,b])=>`${x(a)},${y(b)}`).join(' ')}" fill="none" stroke="#697684" stroke-dasharray="5 4" stroke-width="1.5"/>`;
  for(const seriesItem of series) {
    if(seriesItem.style==='points' || seriesItem.data.length===1)for(const [a,b] of seriesItem.data)html+=`<circle data-series="${esc(seriesItem.key)}" cx="${x(a)}" cy="${y(b)}" r="3.5" fill="${seriesItem.color}"><title>${esc(seriesItem.label)}: ${esc(format(b,curveTask))} · ${esc(config?.x_label || 'Steps')} ${a}</title></circle>`;
    else if(seriesItem.data.length)html+=`<polyline data-series="${esc(seriesItem.key)}" points="${seriesItem.data.map(([a,b])=>`${x(a)},${y(b)}`).join(' ')}" fill="none" stroke="${seriesItem.color}" stroke-width="2.5"/>`;
  }
  html+=`<text x="${right}" y="${height-4}" text-anchor="end" font-size="11" fill="#526476">${esc(config?.x_label || 'Steps')}</text><title>Measurements averaged across seeds at each configured x value. Dots show measured points. Dashed line: parent comparison.</title>`;
  svg.innerHTML=html;svg.setAttribute('aria-label',`Experiment ${r.round}: ${series.map(s=>s.label).join(' and ')} over ${config?.x_label || 'steps'}.`);
}
function renderTitle(id, toggle, text) {
  const heading=$(id), button=$(toggle);
  if(heading.dataset.text!==text){heading.dataset.text=text;heading.dataset.expanded='false';}
  const expanded=heading.dataset.expanded==='true';
  heading.textContent=text;heading.title=text;heading.className=expanded?'expanded':'';
  button.hidden=!state.run || text.length<100;
  button.textContent=expanded?'Show less':'Show full change';
  button.setAttribute('aria-expanded',String(expanded));
}
function renderCurrent() {
  const r=getRound(state.selected);
  $('round-label').textContent=state.run ? `${r.round ? 'Experiment '+r.round : 'Baseline'}${state.follow ? ' · following latest' : ' · selected'}` : 'No experiments yet';
  renderTitle('change','expand-change',state.run ? changeLabel(r) : 'Turn a hypothesis into a measured result');
  $('hypothesis').textContent=r.round && hypothesisFor(r) ? hypothesisCaption(r) : 'Discuss your goal, add hypotheses, then start experiments.';$('hypothesis').title=r.hypothesis || '';
  $('seeds').innerHTML=Object.entries(r.seeds).map(([s,v])=>`<span class="seed-chip" data-status="${esc(v)}">Seed ${esc(s)} <span>${esc(v)}</span></span>`).join('');
  $('verdict').textContent=Number.isFinite(r.value) ? `${Number.isFinite(r.best_before) ? format(r.best_before)+' → ' : ''}${format(r.value)} · ${r.verdict}` : state.run ? r.verdict==='testing' ? 'Measuring…' : r.verdict : '';
  if(Number.isFinite(r.value) && Number.isFinite(r.best_before))$('verdict').textContent+=` · Δ ${format(r.value-r.best_before,state.data.task,true)}`;
  $('verdict').innerHTML=$('verdict').textContent.split(' · ').map(part=>`<span>${part.split(' → ').map(v=>`<span class="metric-value">${esc(v)}</span>`).join(' → ')}</span>`).join(' · ');
  $('verdict').className='verdict '+statusClass(r.verdict);
  $('follow').hidden=state.follow || !state.run;
  $('inspect-result').href=stageURL('results',{round:r.round});$('inspect-result').hidden=!state.run;
  const errors=groupedErrors(r), error=errors.at(-1)?.error || '', scope=errors.at(-1)?.scope || '';
  $('round-error').textContent=error ? errorSummary(error)+(scope?' · '+scope:'') : '';$('round-error').title=$('round-error').textContent;
  const errorKey=`${state.run?.id}/${r.round}/${error}`;
  if($('round-error-details').dataset.error!==errorKey){$('round-error-details').open=false;$('round-error-details').dataset.error=errorKey;}
  $('round-error-details').hidden=!error;$('round-traceback').textContent=error;
  $('curve-label').textContent=state.data.task.curve ? `${state.data.task.curve.x_label || 'Steps'} · selected experiment` : 'Selected experiment · per-seed measurements';
  drawCurve($('curve'),r);
  renderArtifactPanels().catch(e=>notice(e.message));
}
function experimentArtifactLinks(r) {
  const items=r.run===state.run?.id ? latestArtifacts().filter(a=>a.owner?.experiment===r.round) : [];
  return items.length ? `<div class="experiment-attachments">${items.map(a=>`<a href="${artifactURL(a,true)}">${esc(a.title || a.kind)}</a>`).join('')}</div>` : r.run!==state.run?.id ? `<a href="${esc(stageURL('results',{run:r.run,round:r.round}))}">Inspect artifacts</a>` : '';
}
function renderResults() {
  const rows=ledgerRows(), best=state.rounds.get(state.best), task=state.data.task;
  $('result-best').innerHTML=best && Number.isFinite(best.value) ? `Best ${esc(metricLabel().toLowerCase())}: <span class="metric-value">${esc(format(best.value))}</span>` : 'Results will appear here';
  $('result-location').textContent=state.run ? `projects/${state.slug}/runs/${state.run.id}/` : `projects/${state.slug}/ · start a run to collect reproducible artifacts`;
  const bestFile='best_'+task.editable;
  const available=state.files.includes(bestFile) || state.files.includes(`round${state.best}_${task.editable}`);
  $('downloads').innerHTML=state.run ? `<a class="download" href="${projectAPI(`/runs/${encodeURIComponent(state.run.id)}/bundle.zip`)}">Download all</a><a href="${projectAPI(`/runs/${encodeURIComponent(state.run.id)}/ledger.csv`)}">Ledger CSV</a>`+(available ? `<a class="download" href="${fileURL(bestFile,true)}">Download best ${esc(task.editable)}</a>` : '')+['events.jsonl','loop.log'].filter(f=>state.files.includes(f)).map(f=>`<a href="${fileURL(f,true)}">${f}</a>`).join('') : '';
  $('metric-heading').textContent='Best '+metricLabel().toLowerCase()+(task.unit?' ('+task.unit+')':'');
  const reference=task.reference, reached=Number.isFinite(best?.value) && Number.isFinite(reference?.value) && Math.abs(best.value-reference.value)<=(task.noise || 0);
  const reachedRound=reached ? rows.find(r=>(r.round===0 || r.verdict==='improved') && Number.isFinite(r.value) && Math.abs(r.value-reference.value)<=(task.noise || 0)) : null;
  $('result-reference').hidden=!Number.isFinite(reference?.value);
  $('result-reference').textContent=Number.isFinite(reference?.value) ? `${reference.label || 'Reference'} ${format(reference.value).replace(/^-/,'−')}${reachedRound ? reachedRound.round===0?' · reached at baseline':' · reached in round '+reachedRound.round : ''}` : '';
  $('ledger-note').textContent=hypothesisRule()+' Evidence spans project runs. Best result includes all measured experiments; Δ compares each experiment with its kept parent.';
  const focused=document.activeElement?.dataset?.ledger;
  const claims=hypothesisGroups();
  const experimentRows=experiments=>experiments.map(r=>`<tr class="${state.selected===r.round && r.run===state.run?.id?'selected-row ':''}${r.round===state.best && r.run===state.run?.id?'best-row':''}"><td>${r.run===state.run?.id?`<button type="button" data-ledger="${r.round}" aria-pressed="${state.selected===r.round}" title="${esc(changeLabel(r))}">${changeMarkup(r)}</button>`:experimentLink(r)}<small class="muted">Experiment ${r.round} · ${esc(r.run)}</small>${experimentArtifactLinks(r)}</td><td>${format(r.value ?? r.result?.[task.metric])}</td><td class="${deltaClass(r.value-r.best_before,task)}">${Number.isFinite(r.value) && Number.isFinite(r.best_before)?format(r.value-r.best_before,task,true):'—'}</td><td>${pill(experimentVerdict(r.verdict) || 'testing')}</td></tr>`).join('');
  const html=claims.map(h=>{
    const values=h.experiments.map(e=>e.value ?? e.result?.[task.metric]).filter(Number.isFinite);
    const bestValue=values.length?(task.direction==='min'?Math.min(...values):Math.max(...values)):null;
    return `<tr class="hypothesis-row"><td><details data-hypothesis="${esc(h.id)}"><summary><strong>${esc(h.label || h.id)}</strong> <span>${esc(h.statement)}</span></summary>${h.why?`<p><strong>Why:</strong> ${esc(h.why)}</p>`:''}${h.prediction?`<p><strong>Prediction:</strong> ${esc(h.prediction)}</p>`:''}<div class="nested-experiments"><table aria-label="Experiments for ${esc(h.label || h.id)}"><thead><tr><th>Change</th><th>${esc(metricLabel())}</th><th>Δ vs best</th><th>Experiment verdict</th></tr></thead><tbody>${experimentRows(h.experiments) || '<tr><td colspan="4" class="muted">No experiments yet.</td></tr>'}</tbody></table></div><div class="hypothesis-conclusion" data-hypothesis-report="${esc(h.id)}" hidden></div></details></td><td>${pill(h.status)}</td><td>${format(bestValue)}</td><td>${h.experiments.length}</td></tr>`;
  }).join('') || '<tr><td colspan="4" class="empty">No results yet. Add a hypothesis in Discuss, then start experiments.</td></tr>';
  if($('ledger').innerHTML!==html){
    $('result-artifacts').dataset.key='';
    const open=[...$('ledger').querySelectorAll('details[open]')].map(d=>d.dataset.hypothesis);
    $('ledger').innerHTML=html;
    for(const id of open)for(const d of $('ledger').querySelectorAll('details'))if(d.dataset.hypothesis===id)d.open=true;
    if(focused!==undefined)$('ledger').querySelector(`[data-ledger="${focused}"]`)?.focus({preventScroll:true});
  }
  $('learned').innerHTML=[['supported','Supported hypotheses'],['refuted','Refuted hypotheses'],['inconclusive','Inconclusive hypotheses']].map(([status,title])=>{const items=claims.filter(h=>h.status===status);return items.length ? `<h3 class="${statusClass(status)}">${title}</h3><ul>${items.map(h=>`<li>${esc(h.label || h.id)} · ${esc(h.statement)} · ${countLabel(h.experiments.length,'experiment')}</li>`).join('')}</ul>` : '';}).join('') || '<p class="muted">No conclusions yet. Open hypotheses stay untested or testing until the evidence meets the rule.</p>';
  const r=getRound(state.selected);
  renderTitle('detail-title','expand-detail',state.run ? `Experiment ${r.round} · ${changeLabel(r)}` : 'Select an experiment');
  const detailHTML=state.run ? `<p>${esc(r.hypothesis || 'The unchanged starting code.')}</p>${r.why?'<p><strong>Why:</strong> '+esc(r.why)+'</p>':''}${r.prediction?'<p><strong>Prediction:</strong> '+esc(r.prediction)+'</p>':''}<p><span class="verdict-pill ${statusClass(r.verdict)}">${esc(r.verdict)}</span>${Number.isFinite(r.value)?' <strong>'+format(r.value)+'</strong>':''}</p>${groupedErrors(r).map(e=>errorMarkup(e.error,e.scope)).join('')}` : '<p>Each row opens its hypothesis, evidence, code, and diff against the kept parent.</p>';
  if($('detail-text').innerHTML!==detailHTML)$('detail-text').innerHTML=detailHTML;
  drawCurve($('result-curve'),r);
  $('diff-label').textContent=`Diff vs parent experiment ${r.parent ?? 0} · ${task.editable}`;
  $('code-label').textContent=`Full code · round${r.round}_${task.editable}`;
  $('diff-disclosure').hidden=!state.run || r.round===0;
  $('code-disclosure').hidden=!state.run;
  if(state.stage==='results' && state.run)loadArtifacts(r);
  renderArtifactPanels().catch(e=>notice(e.message));
}
async function loadArtifacts(r) {
  const key=`${state.slug}/${state.run.id}/${r.round}/${state.files.length}`;
  if(artifactKey===key)return;artifactKey=key;
  const generation=++artifactGeneration;
  $('code').textContent='Loading experiment code…';$('diff').textContent='Loading diff…';
  artifactText.code=artifactText.diff=null;$('copy-code').disabled=$('copy-diff').disabled=true;$('diff-stats').textContent='';
  const codeFile=`round${r.round}_${state.data.task.editable}`;
  const tasks=[fetch(fileURL(codeFile)).then(async response=>{if(!response.ok)throw new Error('No code artifact: this change may not have been applied.');return response.text();})];
  if(r.round>0)tasks.push(api(projectAPI(`/runs/${encodeURIComponent(state.run.id)}/diff/${r.round}`)));
  const results=await Promise.allSettled(tasks);if(generation!==artifactGeneration)return;
  if(results[0].status==='fulfilled'){artifactText.code=results[0].value;$('code').innerHTML=codeMarkup(artifactText.code);$('copy-code').disabled=false;}
  else $('code').textContent=results[0].reason.message;
  if(r.round>0){const result=results[1];if(result.status==='fulfilled'){
    artifactText.diff=result.value.diff || '';const diff=diffMarkup(artifactText.diff);
    $('diff').innerHTML=artifactText.diff?diff.html:'No code difference from the parent.';
    $('diff-stats').textContent=`+${diff.added} −${diff.removed}`;$('copy-diff').disabled=false;
  }else $('diff').textContent='No reproducible diff is available. '+result.reason.message;}
}
async function addHypothesis(statement, source='you', why='', prediction='') {
  if(state.replay)return;
  const slug=state.slug;
  const result=await api(projectAPI('/hypotheses'),{statement,source,why,prediction});
  if(slug!==state.slug)return;
  state.data.hypotheses=result.hypotheses;renderRun();notice('Hypothesis added. Your untested hypotheses go first.');
}
$('hypothesis-form').onsubmit=async event=>{event.preventDefault();$('add-hypothesis').disabled=true;try{await addHypothesis($('hypothesis-statement').value,'you',$('hypothesis-why').value,$('hypothesis-prediction').value);$('hypothesis-form').reset();}catch(e){notice(e.message);}finally{controls();}};
$('experiment-groups').onclick=event=>{const target=event.target.closest('[data-experiment]');if(!target)return;state.selected=Number(target.dataset.experiment);state.follow=false;renderRun();};
$('new-project').onclick=()=>{$('new-form').hidden=false;$('new-title').focus();};
$('cancel-project').onclick=()=>{$('new-form').hidden=true;$('new-project').focus();};
$('template').onchange=templateNote;
$('template-cards').onchange=event=>{if(event.target.name!=='template-choice')return;$('template').value=event.target.value;templateNote();};
$('round-count').oninput=controls;
for(const kind of ['code','diff'])$(`copy-${kind}`).onclick=async()=>{
  if(artifactText[kind]===null)return;
  try{await navigator.clipboard.writeText(artifactText[kind]);notice(`${kind==='code'?'Code':'Diff'} copied.`);}catch{notice('Clipboard unavailable. Select the text in the viewer to copy it.');}
};
$('new-form').onsubmit=async event=>{event.preventDefault();$('create-project').disabled=true;try{const data=await api('/api/projects',{title:$('new-title').value,template:$('template').value,goal:$('new-goal').value});$('new-form').reset();$('new-form').hidden=true;location.hash=`#/p/${data.project.slug}`;}catch(e){notice(e.message);}finally{$('create-project').disabled=false;}};
$('save-program').onclick=async()=>{$('save-program').disabled=true;const slug=state.slug;try{const result=await api(projectAPI('/program'),{program:$('program').value},'PUT');if(slug===state.slug){state.data.program=result.program;state.programDirty=false;state.programEditing=false;renderProgram();notice('Research goal saved.');}}catch(e){notice(e.message);}finally{controls();}};
$('edit-program').onclick=()=>{state.programEditing=!state.programEditing;renderProgram();if(state.programEditing)$('program').focus();};
for(const [id,toggle] of [['change','expand-change'],['detail-title','expand-detail']])$(toggle).onclick=()=>{$(id).dataset.expanded=String($(id).dataset.expanded!=='true');renderTitle(id,toggle,$(id).textContent);};
$('program').oninput=()=>{state.programDirty=$('program').value!==state.data?.program;controls();};
$('queue-message').onclick=async()=>{try{await addHypothesis($('message').value);$('message').value='';}catch(e){notice(e.message);}};
$('conversation').onclick=async event=>{
  const prompt=event.target.closest('[data-prompt]');
  if(prompt){if(state.replay)return;$('message').value=starterPrompts()[Number(prompt.dataset.prompt)] || '';$('message').focus();return;}
  const button=event.target.closest('[data-queue]');if(!button)return;const message=state.data.discussion[Number(button.dataset.queue)];if(!message)return;button.disabled=true;try{await addHypothesis(message.content,'discussion');}catch(e){notice(e.message);}finally{button.disabled=state.replay;}
};
$('composer').onsubmit=async event=>{
  event.preventDefault();const message=$('message').value.trim();if(!message || state.chatBusy || state.replay)return;
  const slug=state.slug, model=$('model').value, path=projectAPI('/chat');
  state.chatBusy=true;state.data.discussion.push({role:'user',content:message,model});const draft={role:'assistant',content:'',model,pending:true};state.chatDraft=draft;$('message').value='';$('chat-state').textContent='Discussing with '+modelName(model)+'…';renderConversation();
  let text='';
  try{
    const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message,model})});
    if(!response.ok)throw new Error((await response.json()).error);
    const reader=response.body.getReader(),decoder=new TextDecoder();let pending='';
    while(true){const {done,value}=await reader.read();if(done)break;pending+=decoder.decode(value,{stream:true});const lines=pending.split('\n');pending=lines.pop();for(const line of lines){if(!line.startsWith('data:') || line.slice(5).trim()==='[DONE]')continue;const data=JSON.parse(line.slice(5));if(data.error)throw new Error(data.error.message || 'Model error');text+=data.choices?.[0]?.delta?.content || '';}
      if(slug===state.slug && state.chatDraft===draft){draft.content=text;renderConversation();}
    }
    if(!text)throw new Error('The model returned no text. Try sending another message.');
    if(slug===state.slug && state.chatDraft===draft)state.data.discussion.push({role:'assistant',content:text,model});
  }catch(e){notice(e.message);if(text && slug===state.slug && state.chatDraft===draft)state.data.discussion.push({role:'assistant',content:text,model});}
  finally{state.chatBusy=false;state.chatDraft=null;$('chat-state').textContent='';if(slug===state.slug)renderConversation();controls();}
};
$('start').onclick=async()=>{if(state.replay)return;$('start').disabled=true;try{if(state.programDirty){await api(projectAPI('/program'),{program:$('program').value},'PUT');state.data.program=$('program').value;state.programDirty=false;}const run=await api(projectAPI('/runs'),{model:$('model').value,rounds:Number($('round-count').value)});state.data.runs.unshift(run);renderRunPicker();location.hash=stageURL('experiments',{run:run.id,replay:false});}catch(e){notice(e.message);controls();}};
for(const action of ['pause','resume','stop'])$(action).onclick=async()=>{if(state.replay || !state.run?.active)return;try{await api(projectAPI(`/runs/${encodeURIComponent(state.run.id)}/${action}`),{});state.run.paused=action==='pause';controls();notice(action==='stop'?'Stop requested. The current step will finish first.':action==='pause'?'Pause requested. The current step will finish first.':'Resumed.');}catch(e){notice(e.message);}};
$('runs').onchange=()=>{location.hash=stageURL(state.stage,{run:$('runs').value,replay:false});};
$('replay-run').onclick=()=>{location.hash=stageURL('experiments',{replay:!state.replay,speed:8});};
$('follow').onclick=()=>{state.follow=true;state.selected=Math.max(0,...state.rounds.keys());renderRun();};
function selectProgress(event){const target=event.target.closest('[data-round]');if(!target)return;state.selected=Number(target.dataset.round);state.follow=false;renderRun();}
$('progress').onclick=selectProgress;
$('progress').onkeydown=event=>{if(event.key==='Enter' || event.key===' '){event.preventDefault();selectProgress(event);}};
$('ledger').onclick=event=>{const target=event.target.closest('[data-ledger]');if(!target)return;state.selected=Number(target.dataset.ledger);state.follow=false;location.hash=stageURL('results',{round:state.selected});renderRun();$('detail-title').focus({preventScroll:true});};
// Protocol-1 artifacts: all project output is rendered by kind and placement owner.
const artifactCache=new Map();
let artifactPanelGeneration=0;
const artifactTabs=new Map();
function artifactURL(a, download=false, run=a?.owner?.run || state.run?.id) {
  return projectAPI(`/runs/${encodeURIComponent(run)}/artifacts/${encodeURIComponent(a.id || '')}`)+`?revision=${encodeURIComponent(a.revision || 1)}`+(download?'&download=1':'');
}
function safeArtifact(a) {
  const kinds=['code','report','data','chart','image','log','file'],o=a?.owner;
  const integer=v=>Number.isInteger(v) && v>=0;
  const ownerValid=o && typeof o==='object' && typeof o.run==='string' && ['run','hypothesis','experiment','seed'].includes(o.scope) && Object.keys(o).every(k=>['scope','run','hypothesis_id','experiment','seed','attempt'].includes(k)) && (o.scope==='run'?Object.keys(o).length===2:o.scope==='hypothesis'?typeof o.hypothesis_id==='string' && !('experiment' in o) && !('seed' in o):integer(o.experiment) && (o.scope==='seed'?integer(o.seed):!('seed' in o)));
  const valid=a && a.protocol===1 && typeof a.id==='string' && /^[A-Za-z0-9_-]{1,120}$/.test(a.id) && Number.isInteger(a.revision) && a.revision>0 && typeof a.sha256==='string' && /^[a-f0-9]{64}$/.test(a.sha256) && Number.isFinite(a.bytes) && a.bytes>=0 && typeof a.path==='string' && !a.path.split('/').includes('..') && !a.path.startsWith('/') && typeof a.title==='string' && typeof a.mime==='string' && ['harness','script','model'].includes(a.producer) && ownerValid && Array.isArray(a.refs);
  return {...(a || {}),owner:ownerValid?o:{scope:'run',run:state.run?.id || ''},kind:valid && kinds.includes(a.kind)?a.kind:'file',warning:Boolean(a?.warning) || !valid || !kinds.includes(a?.kind)};
}

function latestArtifacts() {
  return [...new Map((state.artifacts || []).map((a,i)=>[a?.id || "invalid-"+i,a])).values()].map(safeArtifact);
}
async function artifactFetch(a, suffix='', run=a?.owner?.run || state.run?.id) {
  const url=artifactURL(a,false,run)+suffix, key=url+'/'+a.sha256;
  if(!artifactCache.has(key)) {
    const pending=fetch(url).then(async response=>{if(!response.ok)throw new Error('Artifact unavailable. Download the bundle to inspect its records.');return suffix.includes('rows=')?response.json():response.text();});
    artifactCache.set(key,pending);
    pending.catch(()=>artifactCache.delete(key));
    if(artifactCache.size>100)artifactCache.delete(artifactCache.keys().next().value);
  }
  return artifactCache.get(key);
}
function artifactLink(a) {
  return `<a href="${artifactURL(a,true)}">Download ${esc(a.title || 'file')}</a>${a.warning?'<small class="muted">Unrecognized artifact; download available.</small>':''}`;
}
function tableMarkup(rows, columns={}, total=rows.length) {
  const fields=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  return `<p class="muted">${Math.min(rows.length,500)} of ${total} rows</p><div class="table-scroll"><table class="artifact-table"><thead><tr>${fields.map(f=>`<th><button type="button" data-sort="${esc(f)}">${esc(f)}${columns[f]?.unit?' ('+esc(columns[f].unit)+')':''}</button>${columns[f]?.type?`<small class="muted">${esc(columns[f].type)}</small>`:''}</th>`).join('')}</tr></thead><tbody>${rows.slice(0,500).map(r=>`<tr>${fields.map(f=>`<td>${esc(r[f] ?? '—')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function chartMarkup(spec, rows) {
  // Reject everything outside rl-chart/1 before constructing SVG. All text is escaped.
  const validChannel=(v,color=false)=>v && typeof v==='object' && typeof v.field==='string' && Object.keys(v).every(k=>(color?['field']:['field','title','scale']).includes(k)) && (!v.scale || Object.keys(v.scale).length===1 && ['linear','log'].includes(v.scale.type));
  if(!spec || spec.$schema!=='rl-chart/1' || !['line','point','bar'].includes(spec.mark) || !spec.data || Object.keys(spec.data).join(',')!=='artifact' || typeof spec.data.artifact!=='string' || Object.keys(spec).some(k=>!['$schema','mark','data','encoding','layer','band','reference'].includes(k)))throw new Error('Unsupported chart specification. Download its source to inspect it.');
  const layers=spec.layer?.length ? spec.layer.map(l=>({...spec,...l})) : [spec];
  if(spec.layer && (!Array.isArray(spec.layer) || spec.layer.length>12 || spec.layer.some(l=>!l || Object.keys(l).some(k=>!['mark','encoding'].includes(k)))))throw new Error('Invalid chart layers');
  for(const layer of [spec,...layers]) {
    if(!['line','point','bar'].includes(layer.mark) || !layer.encoding?.x || !layer.encoding?.y || Object.entries(layer.encoding).some(([k,v])=>!['x','y','color'].includes(k) || !validChannel(v,k==='color')))throw new Error('Invalid chart encodings');
  }
  if(spec.band && (Object.keys(spec.band).sort().join(',')!=='high,low' || typeof spec.band.low!=='string' || typeof spec.band.high!=='string'))throw new Error('Invalid chart band');
  if(spec.reference && (Object.keys(spec.reference).sort().join(',')!=='label,value' || !Number.isFinite(spec.reference.value) || typeof spec.reference.label!=='string'))throw new Error('Invalid chart reference');
  const palette=['#285e89','#176345','#705187','#87601f','#984c46','#376b70'];
  const xlog=spec.encoding.x.scale?.type==='log', ylog=spec.encoding.y.scale?.type==='log';
  const numeric=v=>v!==null && v!=='' && Number.isFinite(Number(v));
  const tx=v=>xlog?Math.log10(v):v, ty=v=>ylog?Math.log10(v):v;
  const points=layers.flatMap(layer=>rows.filter(r=>numeric(r[layer.encoding.x.field]) && numeric(r[layer.encoding.y.field])).map(r=>[Number(r[layer.encoding.x.field]),Number(r[layer.encoding.y.field])])).filter(([a,b])=>(!xlog || a>0) && (!ylog || b>0));
  if(!points.length)return '<p class="muted">No measured points for this chart.</p>';
  const xs=points.map(p=>tx(p[0])),ys=points.map(p=>ty(p[1]));
  if(spec.band)for(const r of rows)for(const f of [spec.band.low,spec.band.high])if(numeric(r[f]) && (!ylog || Number(r[f])>0))ys.push(ty(Number(r[f])));
  if(spec.reference && (!ylog || spec.reference.value>0))ys.push(ty(spec.reference.value));
  let xmin=xs.reduce((a,b)=>Math.min(a,b),Infinity),xmax=xs.reduce((a,b)=>Math.max(a,b),-Infinity),ymin=ys.reduce((a,b)=>Math.min(a,b),Infinity),ymax=ys.reduce((a,b)=>Math.max(a,b),-Infinity);
  if(layers.some(l=>l.mark==='bar') && !ylog)ymin=Math.min(0,ymin);
  if(xmin===xmax){xmin-=.5;xmax+=.5;}if(ymin===ymax){ymin-=.5;ymax+=.5;}
  const x=v=>60+(tx(v)-xmin)/(xmax-xmin)*500,y=v=>230-(ty(v)-ymin)/(ymax-ymin)*180;
  let html='<svg class="artifact-chart" viewBox="0 0 600 280" role="img" aria-label="'+esc(spec.encoding.y.title || spec.encoding.y.field)+'"><title>'+esc(spec.encoding.y.title || spec.encoding.y.field)+'</title>';
  for(let i=0;i<=4;i++) {
    const a=xmin+(xmax-xmin)*i/4,b=ymin+(ymax-ymin)*i/4;
    html+=`<line x1="60" x2="560" y1="${230-i*45}" y2="${230-i*45}" stroke="#e1e7ec"/><text x="50" y="${234-i*45}" text-anchor="end" font-size="11" fill="#526476">${esc(Number((ylog?10**b:b).toPrecision(4)))}</text><text x="${60+i*125}" y="250" text-anchor="middle" font-size="11" fill="#526476">${esc(Number((xlog?10**a:a).toPrecision(4)))}</text>`;
  }
  let series=0;
  for(const layer of layers) {
    const {encoding,mark}=layer,groups=new Map();
    for(const r of rows) {
      const a=Number(r[encoding.x.field]),b=Number(r[encoding.y.field]);
      if(!numeric(r[encoding.x.field]) || !numeric(r[encoding.y.field]) || xlog && a<=0 || ylog && b<=0)continue;
      const key=encoding.color?String(r[encoding.color.field]):encoding.y.title || encoding.y.field;
      if(!groups.has(key))groups.set(key,[]);groups.get(key).push([a,b,r]);
    }
    for(const [label,values] of groups) {
      values.sort((a,b)=>a[0]-b[0]);const color=palette[series++%palette.length];
      html+=`<g><title>${esc(label)}</title>`;
      if(spec.band) {
        const valid=values.filter(([, ,r])=>numeric(r[spec.band.low]) && numeric(r[spec.band.high]) && (!ylog || Number(r[spec.band.low])>0 && Number(r[spec.band.high])>0));
        html+=`<polygon data-band="true" points="${[...valid.map(([a,,r])=>`${x(a)},${y(Number(r[spec.band.low]))}`),...valid.toReversed().map(([a,,r])=>`${x(a)},${y(Number(r[spec.band.high]))}`)].join(' ')}" fill="${color}" opacity=".12"/>`;
      }
      if(mark==='line' && values.length>1)html+=`<polyline data-mark="line" points="${values.map(([a,b])=>`${x(a)},${y(b)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
      else for(const [a,b] of values)html+=mark==='bar'?`<rect data-mark="bar" x="${x(a)-4}" y="${Math.min(y(b),y(ylog?10**ymin:0))}" width="8" height="${Math.max(1,Math.abs(y(b)-y(ylog?10**ymin:0)))}" fill="${color}"><title>${esc(label)}: ${b}</title></rect>`:`<circle data-mark="point" cx="${x(a)}" cy="${y(b)}" r="3" fill="${color}"><title>${esc(label)}: ${b}</title></circle>`;
      html+='</g>';
    }
  }
  if(spec.reference && (!ylog || spec.reference.value>0))html+=`<line x1="60" x2="560" y1="${y(spec.reference.value)}" y2="${y(spec.reference.value)}" stroke="#795919" stroke-dasharray="5 4"/><text x="560" y="${y(spec.reference.value)-5}" text-anchor="end" fill="#795919" font-size="11">${esc(spec.reference.label)}</text>`;
  return html+`<text x="560" y="275" text-anchor="end" fill="#526476" font-size="12">${esc(spec.encoding.x.title || spec.encoding.x.field)}</text><text x="60" y="20" fill="#526476" font-size="12">${esc(spec.encoding.y.title || spec.encoding.y.field)}${spec.encoding.color?' · Colour: '+esc(spec.encoding.color.field):''}</text></svg>`;
}
async function artifactMarkup(input) {
  const a=safeArtifact(input), download=artifactLink(a);
  if(a.kind==='file')return download;
  if(a.kind==='image') {
    if(!['image/png','image/jpeg','image/svg+xml'].includes(a.mime))return artifactLink({...a,warning:true});
    return `<img class="artifact-image" src="${artifactURL(a)}" alt="${esc(a.title)}" loading="lazy">${download}`;
  }
  if(a.kind==='data') {
    const data=await artifactFetch(a,'&rows=500');
    return tableMarkup(data.rows,data.columns,data.total)+download;
  }
  const text=await artifactFetch(a);
  if(a.kind==='report')return `<div class="markdown-body">${markdown(text)}</div>${download}`;
  if(a.kind==='code')return `<pre tabindex="0">${a.role==='diff'?diffMarkup(text).html:codeMarkup(text)}</pre><button type="button" class="secondary" data-artifact-copy="${esc(a.id)}">Copy source</button> ${download}`;
  if(a.kind==='log') {
    const lines=text.trimEnd().split('\n'),count=text?lines.length:0;
    const body=`<div class="log-controls"><label class="log-search">Search log<input type="search" data-log-search placeholder="Find text…"></label><label class="log-tail"><input type="checkbox" data-log-tail checked> Last 200 lines</label></div><pre tabindex="0" data-log-body>${esc(lines.slice(-200).join('\n'))}</pre>${download}`;
    return a.producer==='harness' && a.owner.scope==='run' ? `<details class="loop-log"><summary>Loop log · ${count} lines</summary>${body}</details>` : body;
  }
  if(a.kind==='chart') {
    const spec=JSON.parse(text),data=latestArtifacts().find(d=>d.id===spec.data?.artifact && d.kind==='data');
    if(!data || !a.refs.includes(data.id))throw new Error('Chart data reference is unavailable.');
    const raw=await artifactFetch(data);
    const rows=data.mime==='text/csv'?parseCSV(raw):raw.split('\n').filter(Boolean).map(line=>JSON.parse(line));
    return chartMarkup(spec,rows)+download;
  }
  return download;
}
function parseCSV(text) {
  const rows=[];let row=[],value='',quoted=false;
  for(let i=0;i<text.length;i++) {
    const c=text[i];
    if(c==='"'){if(quoted && text[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}
    else if(c===',' && !quoted){row.push(value);value='';}
    else if(c==='\n' && !quoted){row.push(value.replace(/\r$/,''));rows.push(row);row=[];value='';}
    else value+=c;
  }
  if(row.length || value){row.push(value.replace(/\r$/,''));rows.push(row);}
  const fields=rows.shift() || [];return rows.filter(r=>r.some(v=>v!=='')).map(r=>Object.fromEntries(fields.map((f,i)=>[f,r[i] ?? ''])));
}
function tabFor(a) { return ({chart:'Chart',image:'Chart',data:'Data',code:'Code',log:'Log',report:'Notes',file:'Notes'})[a.kind] || 'Notes'; }
async function renderArtifactPanels() {
  const all=latestArtifacts(), selected=all.filter(a=>a.owner?.experiment===state.selected && ['experiment','seed'].includes(a.owner.scope));
  const hypothesisReports=(state.data?.hypotheses || []).flatMap(h=>(h.reports || []).slice(-1)).map(safeArtifact);
  const current=getRound(state.selected), liveCurve=Boolean(state.data?.task.curve && current.curves.length && current.verdict==='testing');
  const key=`${state.slug}/${state.run?.id}/${state.selected}/${liveCurve}/${[...all,...hypothesisReports].map(a=>a.id+':'+a.sha256).join(',')}`;
  if(!all.some(a=>a.kind==='report' && a.role==='summary' && a.owner?.scope==='run')){$('run-report').hidden=true;$('run-report').innerHTML='';}
  const hosts=[$('experiment-artifacts'),$('result-artifacts'),$('run-artifacts')];
  if(hosts.every(h=>h.dataset.key===key)){
    if(liveCurve)for(const host of hosts){const svg=host.querySelector('[data-live-curve]');if(svg)drawCurve(svg,current);}
    return;
  }
  const generation=++artifactPanelGeneration;
  for(const host of hosts) {
    const runOwner=host.id==='run-artifacts',selection=runOwner?all.filter(a=>a.owner?.scope==='run' && !(a.kind==='report' && a.role==='summary')):selected,live=!runOwner && liveCurve;
    host.dataset.key=key;host.hidden=!selection.length && !live;
    if(host.hidden){host.innerHTML='';continue;}
    const tabs=['Chart','Data','Code','Log','Notes'].filter(t=>selection.some(a=>tabFor(a)===t) || t==='Chart' && live);
    let active=artifactTabs.get(host.id);if(!tabs.includes(active))active=tabs[0];
    host.innerHTML=(runOwner?'<h2>Run artifacts</h2>':'')+`<div class="artifact-tabs" role="tablist" aria-label="${runOwner?'Run':'Experiment'} artifacts">${tabs.map(t=>`<button type="button" role="tab" id="${host.id}-${t}" data-artifact-tab="${t}" aria-selected="${t===active}" aria-controls="${host.id}-panel" tabindex="${t===active?'0':'-1'}">${t}</button>`).join('')}</div><div id="${host.id}-panel" role="tabpanel" tabindex="0" aria-labelledby="${host.id}-${active}"><p class="muted">Loading artifacts…</p></div>`;
    const items=selection.filter(a=>tabFor(a)===active);
    const results=await Promise.allSettled(items.map(artifactMarkup));
    if(generation!==artifactPanelGeneration)return;
    const html=(active==='Chart' && live?'<section class="artifact-entry"><h3>Live measurements</h3><svg data-live-curve class="artifact-chart" role="img" aria-label="Live measurements"></svg></section>':'')+items.map((a,i)=>`<section class="artifact-entry" data-artifact-id="${esc(a.id)}"><div class="section-heading"><h3>${esc(a.title)}</h3><small class="muted">${esc(a.producer)}${a.owner.seed!==undefined?' · seed '+esc(a.owner.seed):''}</small></div>${results[i].status==='fulfilled'?results[i].value:`<p class="muted">${esc(results[i].reason.message)}</p>${artifactLink(a)}`}</section>`).join('');
    host.querySelector('[role="tabpanel"]').innerHTML=html;
    renderMath(host);
    const liveSVG=host.querySelector('[data-live-curve]');if(liveSVG)drawCurve(liveSVG,current);
    if(active==='Chart' && !runOwner) {
      const charts=items.filter(a=>a.kind==='chart'),roles=[...new Set(charts.map(a=>a.role))];
      const candidates=all.filter(a=>a.kind==='chart' && a.owner?.experiment!==state.selected && a.owner?.experiment!==undefined && roles.includes(a.role));
      if(candidates.length)host.querySelector('[role="tabpanel"]').insertAdjacentHTML('beforeend',`<label class="compare-picker">Compare same chart role<select data-compare><option value="">Choose experiment</option>${[...new Set(candidates.map(a=>a.owner.experiment))].map(n=>`<option value="${n}">Experiment ${n}</option>`).join('')}</select></label><div class="artifact-comparison"></div>`);
    }
  }
  // Run and hypothesis reports use the same report renderer.
  const reports=[...hypothesisReports,...all.filter(a=>a.kind==='report' && (a.owner?.scope==='hypothesis' || a.owner?.scope==='run' && a.role==='summary'))];
  for(const report of reports) {
    const target=report.owner.scope==='run'?$('run-report'):[...$('ledger').querySelectorAll('[data-hypothesis-report]')].find(el=>el.dataset.hypothesisReport===report.owner.hypothesis_id);
    if(!target)continue;
    try {const html=await artifactMarkup(report);if(generation!==artifactPanelGeneration)return;target.hidden=false;target.innerHTML=html;renderMath(target);}catch {target.textContent='Report unavailable. Try its download in the bundle.';}
  }
}
async function artifactInteraction(event) {
  const host=event.currentTarget;
  const tab=event.target.closest('[data-artifact-tab]');
  if(tab) {
    artifactTabs.set(host.id,tab.dataset.artifactTab);host.dataset.key='';
    await renderArtifactPanels();$(host.id+'-'+tab.dataset.artifactTab)?.focus();return;
  }
  const copy=event.target.closest('[data-artifact-copy]');
  if(copy)try {await navigator.clipboard.writeText(await artifactFetch(latestArtifacts().find(a=>a.id===copy.dataset.artifactCopy)));notice('Source copied.');}catch {notice('Clipboard unavailable. Select the source to copy it.');}
  const sort=event.target.closest('[data-sort]');
  if(sort) {
    const entry=sort.closest('[data-artifact-id]'),a=latestArtifacts().find(a=>a.id===entry.dataset.artifactId),data=await artifactFetch(a,'&rows=500'),field=sort.dataset.sort,descending=sort.dataset.direction!=='desc';
    const rows=[...data.rows].sort((a,b)=>{const av=a[field] ?? '',bv=b[field] ?? '';return (av!=='' && bv!=='' && Number.isFinite(Number(av)) && Number.isFinite(Number(bv))?Number(av)-Number(bv):String(av).localeCompare(String(bv)))*(descending?-1:1);});
    entry.innerHTML=tableMarkup(rows,data.columns,data.total)+artifactLink(a);
    const button=[...entry.querySelectorAll('[data-sort]')].find(b=>b.dataset.sort===field);button.dataset.direction=descending?'desc':'asc';button.closest('th').setAttribute('aria-sort',descending?'descending':'ascending');button.focus();
  }
}
async function artifactInput(event) {
  const host=event.currentTarget,entry=event.target.closest('[data-artifact-id]');
  if(entry && (event.target.matches('[data-log-search]') || event.target.matches('[data-log-tail]'))) {
    const a=latestArtifacts().find(a=>a.id===entry.dataset.artifactId),text=await artifactFetch(a),term=entry.querySelector('[data-log-search]').value.toLowerCase(),tail=entry.querySelector('[data-log-tail]').checked;
    const lines=text.split('\n').filter(line=>!term || line.toLowerCase().includes(term));
    entry.querySelector('[data-log-body]').textContent=(tail?lines.slice(-200):lines).join('\n');
  }
  if(event.target.matches('[data-compare]')) {
    const n=event.target.value,target=host.querySelector('.artifact-comparison');
    if(n===''){target.innerHTML='';return;}
    const all=latestArtifacts(),selected=all.filter(a=>a.owner?.experiment===state.selected && a.kind==='chart'),roles=[...new Set(selected.map(a=>a.role))];
    target.innerHTML='<p class="muted">Loading comparison…</p>';
    const generation=artifactPanelGeneration;
    const pairs=roles.map(role=>[selected.find(a=>a.role===role),all.find(a=>a.kind==='chart' && a.role===role && String(a.owner?.experiment)===n)]).filter(pair=>pair.every(Boolean));
    const html=await Promise.all(pairs.map(async ([left,right])=>`<div class="compare-pair"><section><h3>Experiment ${state.selected} · ${esc(left.role)}</h3>${await artifactMarkup(left)}</section><section><h3>Experiment ${esc(n)} · ${esc(right.role)}</h3>${await artifactMarkup(right)}</section></div>`));
    if(generation===artifactPanelGeneration)target.innerHTML=html.join('');
  }
}
for(const id of ['experiment-artifacts','result-artifacts','run-artifacts']) {
  $(id).onclick=event=>{artifactInteraction(event).catch(e=>notice(e.message));};
  $(id).oninput=event=>{artifactInput(event).catch(e=>notice(e.message));};
  $(id).onchange=event=>{if(event.target.matches('[data-compare]'))artifactInput(event).catch(e=>notice(e.message));};
  $(id).onkeydown=event=>{
    if(!event.target.matches('[role="tab"]') || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
    event.preventDefault();const tabs=[...$(id).querySelectorAll('[role="tab"]')],index=tabs.indexOf(event.target),next=event.key==='Home'?0:event.key==='End'?tabs.length-1:(index+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;tabs[next].click();
  };
}

window.addEventListener('hashchange',route);
window.addEventListener('resize',scheduleRender);
window.addEventListener('beforeunload',event=>{if(state.programDirty){event.preventDefault();event.returnValue='';}});
route();
