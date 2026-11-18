'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors = {supported:'#176345',refuted:'#a33432',crashed:'#a33432','no effect':'#795919',baseline:'#526476',testing:'#285e89'};
const state = {slug:null, stage:'discuss', data:null, run:null, rounds:new Map(), selected:0, best:0, follow:true, replay:false, speed:8, source:null, files:[], projects:[], templates:[], domain:'', chatBusy:false, chatDraft:null, generation:0, loadingRun:null};
let noticeTimer, paintPending = false, artifactKey = '', artifactGeneration = 0;
const artifactText = {code:null,diff:null};
function notice(message='') {
  $('notice').textContent = message; clearTimeout(noticeTimer);
  if (message) noticeTimer = setTimeout(() => { $('notice').textContent = ''; }, 8000);
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
  const r = getRound(event.round ?? 0);
  switch(event.event) {
    case 'baseline': if(paint && !state.replay)refreshFiles(); Object.assign(r,{change:'Original settings',hypothesis:'Baseline',value:event.result?.[event.metric || state.data.task.metric],verdict:'baseline',result:event.result}); break;
    case 'idea':
      Object.assign(r,{parent:state.best},event); if(state.follow) state.selected=r.round;
      if(!state.replay && ['human','you'].includes(event.source)) {
        const i=state.data.ideas.findIndex(idea=>idea.hypothesis===event.hypothesis && (!idea.ts || event.ts>=idea.ts));
        if(i>=0){state.data.ideas.splice(i,1);renderQueue();}
      }
      break;
    case 'code': case 'fix': r.diffs.push({kind:event.event,diff:event.diff || ''}); r.verdict='testing'; break;
    case 'run_start': r.seeds[event.seed]='running'; if(state.follow) state.selected=r.round; break;
    case 'curve': r.curves.push(event); break;
    case 'run_end': r.seeds[event.seed]='complete'; r.seedResults ??= []; r.seedResults.push({seed:event.seed,...event.result}); break;
    case 'run_error': r.errors.push(event.error || 'Experiment failed'); markCrashed(r,event.seed); break;
    case 'edit_rejected': r.errors.push('Edit rejected: '+event.reason); break;
    case 'idea_error': r.errors.push(event.error); notice(`Round ${event.round}: ${event.error}`); break;
    case 'result': Object.assign(r,event); if(event.error && r.errors.at(-1)!==event.error)r.errors.push(event.error); if(event.verdict==='crashed')markCrashed(r,event.seed); r.value ??= event.result?.[state.data.task.metric]; if(event.verdict==='supported') state.best=r.round; if(paint && !state.replay)refreshFiles(); break;
    case 'goal_reached': notice('Goal reached. The measured result is recorded.'); break;
    case 'done':
      state.run.active=false; state.run.finished=true; state.data.runs.forEach(run=>{if(run.id===state.run.id)run.active=false;}); state.source?.close(); notice(event.reason || 'Run complete. Explore the ledger and artifacts in Results.');
      if(paint) refreshFiles(); break;
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
function errorMarkup(error) {
  return `<p class="error error-summary" title="${esc(errorSummary(error))}">${esc(errorSummary(error))}</p><details class="error-details"><summary>Show error</summary><pre tabindex="0" aria-label="Experiment error">${esc(error)}</pre></details>`;
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
  const inline=s=>s.replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
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
  return [`Why does the baseline fail on ${metricLabel(task).toLowerCase()}?`, 'What should we try first?', 'What would refute our best idea?'];
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
      const candidate=e.event==='baseline' ? e.result?.[e.metric || task.metric] : e.verdict==='supported' ? e.value ?? e.result?.[task.metric] : null;
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
  $('project-list').innerHTML=projects.length ? projects.map(p=>`<article class="project-row"><div class="project-intro"><a href="#/p/${esc(p.slug)}">${esc(p.title)}</a>${domainChip(p.task?.domain)}<p class="project-description muted" title="${esc(p.description || '')}">${esc(p.description || 'Open the project to review its research goal.')}</p></div><div class="project-template"><span class="muted">Template</span><span>${esc(p.template)}</span><div><span class="project-status ${p.status==='Running'?'is-running':''}">${esc(p.status || 'Status unavailable')}</span></div></div><div><span class="muted">Experiments</span><span class="project-number">${p.experiments}</span></div><div class="project-metric"><div><span class="muted">Best ${esc(metricLabel(p.task))} ${directionMark(p.task)}</span><strong>${esc(format(p.best,p.task))}</strong></div>${sparkline(p.progress,p.task)}</div><div><span class="muted">Last activity</span><time datetime="${new Date(p.updated*1000).toISOString()}">${esc(activityTime(p.updated))}</time></div></article>`).join('') : '<div class="empty project-empty"><h3>Your next question starts here</h3><p>Create a project from a template to keep your goal, ideas, experiments, and evidence together.</p><button type="button" class="secondary" id="first-project">Create your first project</button></div>';
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
  if(changed){state.programDirty=false;state.programEditing=false;state.chatDraft=null;state.source?.close();state.source=null;state.run=null;state.data=null;state.rounds=new Map();state.files=[];state.loadingRun=null;artifactKey='';artifactGeneration++;$('program').value='';$('title').textContent='Loading project…';}
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
  state.source?.close();state.source=null;state.rounds=new Map();state.selected=selected??0;state.follow=selected===null;state.best=0;state.files=[];artifactKey='';artifactGeneration++;
  if(!id){state.run=null;return;}
  state.loadingRun=id;
  const detail=await api(projectAPI(`/runs/${encodeURIComponent(id)}`));
  if(generation!==state.generation)return;
  state.loadingRun=null;state.run=detail.run;state.files=detail.files;renderRunPicker();
  if(state.run.active || state.replay) {
    const suffix=state.replay ? `?replay=1&speed=${state.speed}` : '';
    const source=new EventSource(projectAPI(`/runs/${encodeURIComponent(id)}/events${suffix}`));state.source=source;
    source.onmessage=message=>{if(state.source!==source)return;apply(JSON.parse(message.data));};
    source.addEventListener('closed',()=>{if(state.source!==source)return;source.close();state.run.active=false;state.data.runs.forEach(run=>{if(run.id===state.run.id)run.active=false;});renderRun();notice('Run closed. Inspect loop.log if it ended before a verdict.');});
    source.onerror=()=>{if(state.source===source && state.run.active)notice('Event connection interrupted. Reconnecting…');};
  } else {for(const event of detail.events)apply(event,false);}
}
async function refreshFiles() {
  if(!state.run)return;
  const slug=state.slug, id=state.run.id;
  try {
    const data=await api(projectAPI(`/runs/${encodeURIComponent(id)}`));
    if(slug!==state.slug || id!==state.run?.id)return;
    state.files=data.files;
    const run=state.data.runs.find(r=>r.id===id);if(run)Object.assign(run,data.run,{active:state.run.finished ? false : data.run.active});
    renderRunPicker(); renderResults(); controls();
  }catch(e){notice(e.message);}
}
function renderQueue() {
  const ideas=state.data?.ideas || [];
  $('queue-count').textContent=String(ideas.length);
  $('queue').innerHTML=ideas.length ? ideas.map(i=>`<li><span>${esc(i.change || i.hypothesis)}</span></li>`).join('') : '<li class="queue-empty muted">No ideas queued yet. Discuss an approach, then choose “Queue this idea”, or queue your own text.</li>';
}
function renderConversation() {
  const messages=[...(state.data?.discussion || [])]; if(state.chatDraft)messages.push(state.chatDraft);
  $('conversation').innerHTML=messages.length ? messages.map((m,i)=>`<article class="chat ${m.role==='user'?'user':'assistant'}${m.pending?' streaming':''}"><header>${m.role==='user' ? 'You' : esc(modelName(m.model || $('model').value))}${m.role==='assistant'?'<span class="model-label">Local model</span>':''}</header><div class="message-content">${markdown(m.content || 'Thinking…')}</div>${m.pending?'<span class="streaming-caret" aria-label="Responding"></span>':''}${m.role==='assistant' && !m.pending ? `<button class="secondary queue-action" type="button" data-queue="${i}">Queue this idea</button>` : ''}</article>`).join('') : `<div class="discussion-empty"><h3>What do you want to test?</h3><p>Explore a few approaches with your local model. Queue a promising idea when you’re ready to measure it.</p><div class="starter-prompts">${starterPrompts().map((p,i)=>`<button type="button" class="secondary" data-prompt="${i}">${esc(p)}</button>`).join('')}</div><p class="muted">Your discussion stays with this project.</p></div>`; renderMath($('conversation'));
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
  $('send').disabled=state.replay || state.chatBusy || !state.data;
  $('queue-message').disabled=state.replay || !state.data;
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
  $('run-state').textContent=state.replay ? 'Replaying recorded events' : active ? state.run.paused ? 'Pause requested' : `Running · ${latest.round ? 'round '+latest.round+' of '+(state.run.rounds || $('round-count').value || 10) : 'baseline'}${seed!==undefined?' · seed '+seed:''}` : state.run ? 'Run finished' : 'Ready for experiments';
  $('run-state').className=active && !state.run.paused && !state.replay ? 'is-live' : '';
  $('start-summary').textContent=`Baseline + ${$('round-count').value || 10} rounds · seeds ${state.data?.task.seeds.join(', ') || '—'} · fixed budget`;
  $('run-model').textContent=state.run ? modelName(state.run.model)+' · '+state.run.id : 'Start from Discuss when your goal is ready.';
  for(const button of $('conversation').querySelectorAll('[data-queue]'))button.disabled=state.replay;
}
function renderRun() {
  if(!state.data)return;
  controls();
  drawProgress();renderCurrent();renderResults();
}
function progressData(rows=ledgerRows(), task=state.data.task) {
  let best=null;
  return rows.filter(r=>Number.isFinite(r.value) || r.verdict==='crashed').map(r=>{const kept=Number.isFinite(r.value) && (r.round===0 || r.verdict==='supported');if(kept)best=r.value;return {...r,kept,runningBest:best,direction:task.direction};});
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
  $('progress-subtitle').textContent=`${metricLabel()} · ${task.direction==='min'?'lower':'higher'} is better · ${ledgerRows().filter(r=>r.round>0 && r.verdict!=='testing').length} experiments`;
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
    const label=changeLabel(p), short=label.length>28 ? label.slice(0,27)+'…' : label;
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
    const fill=p.kept?'#176345':p.verdict==='refuted'?'#d9a19f':'#b3bdc5';
    const outside=measured && (p.value<low?'low':p.value>high?'high':'');
    const value=measured?format(p.value)+(outside?'; outside plotted range, clamped at '+outside+' edge':''):'No measurement; marker at bottom edge';
    const caret=outside ? `<path data-clamped="${outside}" d="M${px-5},${py+(outside==='high'?7:-7)} l5,${outside==='high'?-7:7} l5,${outside==='high'?7:-7}" fill="none" stroke="${p.verdict==='refuted'?'#a33432':color(p.verdict)}" stroke-width="2"/>` : '';
    const age=p.arrived===undefined?Infinity:Date.now()-p.arrived;
    const live=state.run?.active && !state.run.paused && !state.replay && p.round===points.at(-1).round;
    html+=`<g id="point-${p.round}" role="button" tabindex="0" class="progress-dot" aria-label="Inspect experiment ${p.round}: ${esc(label)}, ${esc(value)}, ${esc(p.verdict)}" data-round="${p.round}"><title>${esc(label)} · ${esc(value)} · ${esc(p.verdict)}</title><circle cx="${px}" cy="${py}" r="13" fill="transparent"/>${live?`<circle class="live-point" style="animation-delay:-${Date.now()%1800}ms" cx="${px}" cy="${py}" r="11"/>`:''}${state.selected===p.round ? `<circle cx="${px}" cy="${py}" r="9" fill="none" stroke="#285e89" stroke-width="2"/>` : ''}${caret || (p.verdict==='crashed'?`<path data-crashed="true" d="M${px-4},${py-4} l8,8 m-8,0 l8,-8" fill="none" stroke="#a33432" stroke-width="2"/>`:`<circle class="${age<200?'new-dot':''}" style="animation-delay:-${age<200?age:0}ms" cx="${px}" cy="${py}" r="${p.kept?5:3.5}" fill="${fill}" stroke="${p.kept?'#154f39':p.verdict==='refuted'?'#93625f':'#687b89'}" stroke-width="1.5"/>`)}${p.kept ? `${Math.abs(labelY-py)>25?`<path class="label-leader" d="M${px},${py+(labelY>py?7:-7)} V${labelY+(labelY>py?-10:8)}"/>`:''}<text data-kept-label="true" transform="translate(${px+(flip?-8:8)} ${labelY}) rotate(${angle})" text-anchor="${flip?'end':'start'}" fill="#176345" font-size="12"><title>${esc(label)}</title>${esc(text)}</text>` : ''}</g>`;
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
  let low=Math.min(...all.map(d=>d[1])), high=Math.max(...all.map(d=>d[1]));
  if(task.unit==='%'){low=Math.min(0,low);high=Math.max(1,high);}
  if(task.unit!=='%'){const padding=(high-low)*.06 || Math.abs(high)*.05 || 1;low-=padding;high+=padding;}
  const min=Math.min(0,...all.map(d=>d[0])),max=Math.max(...all.map(d=>d[0]));
  const curveTask=task;
  const yLabels=[low,(low+high)/2,high].map(v=>format(v,curveTask));
  const left=Math.max(45,...yLabels.map(t=>t.length*6+10)),right=Math.max(left+40,width-16);
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
  for(let i=0;i<=2;i++){
    const v=low+(high-low)*i/2, a=min+(max-min)*i/2;
    html+=`<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e7ec"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end" font-size="11" fill="#526476">${esc(yLabels[i])}</text><text x="${x(a)}" y="${height-24}" text-anchor="middle" fill="#526476" font-size="11">${a.toLocaleString(undefined,{maximumFractionDigits:2})}</text>`;
  }
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
  $('round-label').textContent=state.run ? `${r.round ? 'Round '+r.round : 'Baseline'}${state.follow ? ' · following latest' : ' · selected'}` : 'No experiments yet';
  renderTitle('change','expand-change',state.run ? changeLabel(r) : 'Turn a hypothesis into a measured result');
  $('hypothesis').textContent=r.hypothesis || 'Discuss your goal, queue ideas, then start experiments.';$('hypothesis').title=r.hypothesis || '';
  $('seeds').innerHTML=Object.entries(r.seeds).map(([s,v])=>`<span class="seed-chip" data-status="${esc(v)}">Seed ${esc(s)} <span>${esc(v)}</span></span>`).join('');
  $('verdict').textContent=Number.isFinite(r.value) ? `${Number.isFinite(r.best_before) ? format(r.best_before)+' → ' : ''}${format(r.value)} · ${r.verdict}` : state.run ? r.verdict==='testing' ? 'Measuring…' : r.verdict : '';
  if(Number.isFinite(r.value) && Number.isFinite(r.best_before))$('verdict').textContent+=` · Δ ${format(r.value-r.best_before,state.data.task,true)}`;
  $('verdict').innerHTML=$('verdict').textContent.split(' · ').map(part=>`<span>${part.split(' → ').map(v=>`<span class="metric-value">${esc(v)}</span>`).join(' → ')}</span>`).join(' · ');
  $('verdict').className='verdict '+statusClass(r.verdict);
  $('follow').hidden=state.follow || !state.run;
  $('inspect-result').href=stageURL('results',{round:r.round});$('inspect-result').hidden=!state.run;
  const error=r.errors.at(-1) || '';
  $('round-error').textContent=error ? errorSummary(error) : '';$('round-error').title=$('round-error').textContent;
  const errorKey=`${state.run?.id}/${r.round}/${error}`;
  if($('round-error-details').dataset.error!==errorKey){$('round-error-details').open=false;$('round-error-details').dataset.error=errorKey;}
  $('round-error-details').hidden=!error;$('round-traceback').textContent=error;
  $('curve-label').textContent=state.data.task.curve ? `${state.data.task.curve.x_label || 'Steps'} · selected round` : 'Selected round · per-seed measurements';
  drawCurve($('curve'),r);
}
function renderResults() {
  const rows=ledgerRows(), best=state.rounds.get(state.best), task=state.data.task;
  $('result-best').innerHTML=best && Number.isFinite(best.value) ? `Best ${esc(metricLabel().toLowerCase())}: <span class="metric-value">${esc(format(best.value))}</span>` : 'Results will appear here';
  $('result-location').textContent=state.run ? `projects/${state.slug}/runs/${state.run.id}/` : `projects/${state.slug}/ · start a run to collect reproducible artifacts`;
  const bestFile='best_'+task.editable;
  const available=state.files.includes(bestFile) || state.files.includes(`round${state.best}_${task.editable}`);
  $('downloads').innerHTML=state.run ? (available ? `<a class="download" href="${fileURL(bestFile,true)}">Download best ${esc(task.editable)}</a>` : '')+['events.jsonl','loop.log'].filter(f=>state.files.includes(f)).map(f=>`<a href="${fileURL(f,true)}">${f}</a>`).join('') : '';
  $('metric-heading').textContent=metricLabel()+(task.unit?' ('+task.unit+')':'');
  const reference=task.reference, reached=Number.isFinite(best?.value) && Number.isFinite(reference?.value) && Math.abs(best.value-reference.value)<=(task.noise || 0);
  const reachedRound=reached ? rows.find(r=>(r.round===0 || r.verdict==='supported') && Number.isFinite(r.value) && Math.abs(r.value-reference.value)<=(task.noise || 0)) : null;
  $('result-reference').hidden=!Number.isFinite(reference?.value);
  $('result-reference').textContent=Number.isFinite(reference?.value) ? `${reference.label || 'Reference'} ${format(reference.value).replace(/^-/,'−')}${reachedRound ? reachedRound.round===0?' · reached at baseline':' · reached in round '+reachedRound.round : ''}` : '';
  $('ledger-note').textContent=`${metricLabel()} · ${task.direction==='min'?'lower':'higher'} is better. Δ is the signed metric change from the kept best before that round.`;
  const focused=document.activeElement?.dataset?.ledger;
  const html=rows.map(r=>`<tr class="${state.selected===r.round?'selected-row ':''}${r.round===state.best?'best-row':''}"><td>${r.round}${r.round===state.best?'<span class="best-marker" title="Kept best" aria-label="Kept best"></span>':''}</td><td><button type="button" data-ledger="${r.round}" aria-pressed="${state.selected===r.round}" title="${esc(changeLabel(r))}"><span class="ledger-change">${esc(changeLabel(r))}</span></button></td><td>${format(r.value)}</td><td class="${deltaClass(r.value-r.best_before,task)}">${Number.isFinite(r.value) && Number.isFinite(r.best_before) ? format(r.value-r.best_before,task,true) : '—'}</td><td><span class="verdict-pill ${statusClass(r.verdict)}">${esc(r.verdict)}</span></td><td>${r.round===0 ? 'baseline' : r.source==='human' || r.source==='you' ? 'you' : esc(modelName(state.run?.model || 'model'))}</td></tr>`).join('') || '<tr><td colspan="6" class="empty">No results yet. Your baseline and every experiment will appear here, including failed attempts.</td></tr>';
  if($('ledger').innerHTML!==html){$('ledger').innerHTML=html;if(focused!==undefined)$('ledger').querySelector(`[data-ledger="${focused}"]`)?.focus({preventScroll:true});}
  $('learned').innerHTML=[['supported','Supported changes'],['refuted','Refuted changes'],['no effect','No measured effect'],['crashed','Failed attempts'],['not run','Unapplied changes']].map(([verdict,title])=>{const items=rows.filter(r=>r.verdict===verdict);return items.length ? `<h3 class="${statusClass(verdict)}">${title}</h3><ul>${items.map(r=>`<li>Round ${r.round}: ${esc(changeLabel(r))}${Number.isFinite(r.value)?' · '+format(r.value):''}</li>`).join('')}</ul>` : '';}).join('') || '<p class="muted">After experiments finish, their recorded verdicts will be grouped here. There are no conclusions yet.</p>';
  const r=getRound(state.selected);
  renderTitle('detail-title','expand-detail',state.run ? `Experiment ${r.round} · ${changeLabel(r)}` : 'Select an experiment');
  const detailHTML=state.run ? `<p>${esc(r.hypothesis || 'The unchanged starting code.')}</p>${r.why?'<p><strong>Why:</strong> '+esc(r.why)+'</p>':''}${r.prediction?'<p><strong>Prediction:</strong> '+esc(r.prediction)+'</p>':''}<p><span class="verdict-pill ${statusClass(r.verdict)}">${esc(r.verdict)}</span>${Number.isFinite(r.value)?' <strong>'+format(r.value)+'</strong>':''}</p>${r.errors.map(errorMarkup).join('')}` : '<p>Each row opens its hypothesis, evidence, code, and diff against the kept parent.</p>';
  if($('detail-text').innerHTML!==detailHTML)$('detail-text').innerHTML=detailHTML;
  drawCurve($('result-curve'),r);
  $('diff-label').textContent=`Diff vs parent experiment ${r.parent ?? 0} · ${task.editable}`;
  $('code-label').textContent=`Full code · round${r.round}_${task.editable}`;
  $('diff-disclosure').hidden=!state.run || r.round===0;
  $('code-disclosure').hidden=!state.run;
  if(state.stage==='results' && state.run)loadArtifacts(r);
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
async function queueIdea(hypothesis) {
  if(state.replay)return;
  const slug=state.slug;
  const result=await api(projectAPI('/ideas'),{hypothesis});
  if(slug!==state.slug)return;
  state.data.ideas=result.ideas;renderQueue();notice('Idea queued. Your ideas are tested before model proposals.');
}
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
$('queue-message').onclick=async()=>{try{await queueIdea($('message').value);$('message').value='';}catch(e){notice(e.message);}};
$('conversation').onclick=async event=>{
  const prompt=event.target.closest('[data-prompt]');
  if(prompt){if(state.replay)return;$('message').value=starterPrompts()[Number(prompt.dataset.prompt)] || '';$('message').focus();return;}
  const button=event.target.closest('[data-queue]');if(!button)return;const message=state.data.discussion[Number(button.dataset.queue)];if(!message)return;button.disabled=true;try{await queueIdea(message.content);}catch(e){notice(e.message);}finally{button.disabled=state.replay;}
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
window.addEventListener('hashchange',route);
window.addEventListener('resize',scheduleRender);
window.addEventListener('beforeunload',event=>{if(state.programDirty){event.preventDefault();event.returnValue='';}});
route();
