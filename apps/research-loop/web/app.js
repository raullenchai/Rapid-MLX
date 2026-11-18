const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const replay = params.get('replay') === '1';
const speed = Math.max(0.1, Math.min(1000, Number(params.get('speed')) || 8));
let runId = params.get('run') || '', source, selected = 0, follow = true, best = 0;
let rounds = new Map(), messages = [], active = false, paused = false, chatBusy = false;
let catalog = [], model = '';
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const statusClass = value => (value || '').replaceAll(' ', '-');
function notice(text = '') { $('notice').textContent = text; }
function round(n) {
  if (!rounds.has(n)) rounds.set(n, {round:n, parent:0, curves:[], seeds:{}, diffs:[], errors:[], verdict:'testing…'});
  return rounds.get(n);
}
async function api(path, data) {
  const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed');
  return value;
}
function controls() {
  for (const id of ['pause','resume','stop','test']) $(id).disabled = replay || !active || (chatBusy && id === 'test');
  $('send').disabled = replay || !runId || chatBusy;
  $('pause').disabled ||= paused;
  $('resume').disabled ||= !paused;
  $('start').disabled = replay;
  $('run-state').textContent = replay ? 'Replay' : active ? paused ? 'Pause requested · after current step' : 'Running' : runId ? 'Finished' : 'Ready';
}
function connect(id) {
  source?.close(); runId = id; rounds = new Map(); selected = best = 0; follow = true;
  messages = []; $('conversation').innerHTML = ''; $('cards').innerHTML = '';
  const meta = catalog.find(r => r.id === id);
  active = Boolean(meta?.active); paused = Boolean(meta?.paused);
  $('model').textContent = `${meta?.model || model} · running on this Mac`;
  const url = new URL(location); url.searchParams.set('run', id); history.replaceState({}, '', url);
  $('runs').value = id;
  const suffix = replay ? `?replay=1&speed=${speed}` : '';
  source = new EventSource(`/api/runs/${encodeURIComponent(id)}/events${suffix}`);
  source.onmessage = message => { apply(JSON.parse(message.data)); };
  source.addEventListener('closed', () => { source.close(); active = false; controls(); notice('Run ended. Check loop.log if it ended before a verdict.'); });
  source.onerror = () => notice('Event connection interrupted. Reconnecting…');
  render(); controls();
}
function apply(event) {
  notice();
  const r = round(event.round ?? 0);
  switch (event.event) {
    case 'baseline':
      r.hypothesis = 'Baseline'; r.change = 'Original settings'; r.value = event.result[event.metric || 'test_acc']; r.metric = event.metric || 'test_acc'; r.verdict = 'baseline';
      $('title').textContent = event.task || 'Research Loop';
      $('subtitle').textContent = 'Discuss an idea. Run it on this Mac. Let the evidence guide the next step.';
      break;
    case 'idea': Object.assign(r, {parent:best}, event); if (follow) selected = event.round; break;
    case 'code': case 'fix': r.verdict = 'testing…'; r.diffs.push({kind:event.event, diff:event.diff}); break;
    case 'edit_rejected': r.errors.push(`Edit rejected: ${event.reason}`); break;
    case 'run_start': r.seeds[event.seed] = 'running'; if (follow) selected = event.round; break;
    case 'curve': r.curves.push(event); break;
    case 'run_end': r.seeds[event.seed] = 'done'; break;
    case 'run_error': r.verdict = 'crashed'; r.errors.push(event.error || 'Experiment crashed'); r.hadCrash = true; break;
    case 'idea_error': notice(`The model could not propose round ${event.round}: ${event.error}`); break;
    case 'result': Object.assign(r, event); r.value ??= event.result?.[rounds.get(0)?.metric || 'test_acc']; if (event.verdict === 'supported') best = event.round; break;
    case 'goal_reached': notice('Goal reached. The result is recorded.'); break;
    case 'done': active = false; source.close(); controls(); notice(event.reason || 'Run complete. Select any round to explore the evidence.'); break;
  }
  render();
}
function render() {
  const r = round(selected);
  $('round-label').textContent = selected ? `Round ${selected}${follow ? ' · following live' : ' · selected'}` : 'Baseline';
  $('change').textContent = r.change || r.hypothesis || (runId ? 'Measuring the baseline…' : 'A small experiment, measured.');
  $('experiment-status').textContent = r.verdict;
  $('experiment-status').className = `badge ${statusClass(r.verdict)}`;
  $('seeds').innerHTML = Object.entries(r.seeds).map(([seed,state]) => `<span class="seed">Seed ${esc(seed)} · ${state === 'done' ? 'complete' : 'running…'}</span>`).join('');
  const metric = rounds.get(0)?.metric || 'test_acc';
  const accuracy = metric.endsWith('acc');
  const format = value => accuracy ? pct(value) : Number.isFinite(value) ? String(value) : '—';
  const label = metric.replaceAll('_', ' ');
  $('verdict').textContent = Number.isFinite(r.value) ? `${label} ${Number.isFinite(r.best_before) ? format(r.best_before) + ' → ' : ''}${format(r.value)} · ${r.verdict}` : r.verdict === 'testing…' ? (runId ? 'Testing the idea…' : 'Results will guide the next idea.') : r.verdict;
  $('verdict').className = statusClass(r.verdict);
  $('diff-label').textContent = r.diffs.some(d => d.kind === 'fix') ? 'crashed → fixed by the model' + (Object.values(r.seeds).includes('done') ? ' → ran' : '') : '';
  $('diff').innerHTML = r.diffs.length ? r.diffs.map(d => `<span>${d.kind === 'fix' ? 'Model’s fix' : 'Experiment edit'}</span>\n` + d.diff.split('\n').map(line => `<span class="${line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : ''}">${esc(line)}</span>`).join('\n')).join('\n\n') : 'The model’s exact edit will appear here.';
  $('errors').innerHTML = r.errors.map(e => `<details class="error"><summary>${esc(e.split('\n')[0].slice(0,160))}</summary><pre>${esc(e)}</pre></details>`).join('');
  const scroll = document.querySelector('.idea-scroll'), nearEnd = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
  $('cards').innerHTML = [...rounds.values()].filter(r => r.round && r.hypothesis).map(r => `<article class="card ${selected === r.round ? 'selected' : ''}" tabindex="0" data-round="${r.round}"><header><span>Round ${r.round}</span><span class="badge">${r.source === 'human' ? 'you' : 'model'}</span></header><h3>${esc(r.hypothesis)}</h3><p>${esc(r.why)}</p><p class="change">${esc(r.change)}</p><div class="outcome ${statusClass(r.verdict)}">${esc(r.verdict)}${Number.isFinite(r.value) ? ' · ' + format(r.value) : ''}</div></article>`).join('');
  $('idea-count').textContent = `${[...rounds.values()].filter(r => r.hypothesis && r.round).length} hypotheses`;
  if (nearEnd && follow) scroll.scrollTop = scroll.scrollHeight;
  drawChart(r); drawMap(format);
}
function series(points, field) {
  const groups = new Map();
  for (const p of points) { if (!groups.has(p.step)) groups.set(p.step, []); if (Number.isFinite(p[field])) groups.get(p.step).push(p[field]); }
  return [...groups].sort((a,b) => a[0]-b[0]).map(([step,values]) => [step, values.reduce((a,b)=>a+b,0)/values.length]);
}
function drawChart(r) {
  const comparison = rounds.get(r.parent ?? 0);
  const max = Math.max(5000, ...r.curves.map(p=>p.step), ...(comparison?.curves || []).map(p=>p.step));
  const x = step => 55 + step / max * 715, y = value => 240 - value * 215;
  let svg = [0,.25,.5,.75,1].map(v => `<line x1="55" x2="770" y1="${y(v)}" y2="${y(v)}" stroke="#eee"/><text x="43" y="${y(v)+5}" text-anchor="end" fill="#888" font-size="14">${v*100}%</text>`).join('');
  for (let i=0;i<=5;i++) svg += `<text x="${x(max*i/5)}" y="263" text-anchor="middle" fill="#888" font-size="14">${Math.round(max*i/5).toLocaleString()}</text>`;
  svg += '<text x="770" y="282" text-anchor="end" fill="#888" font-size="13">training steps · mean across seeds</text>';
  function line(points, field, color, width, dash='') {
    const data = series(points, field);
    if (data.length) svg += `<polyline points="${data.map(([a,b])=>`${x(a)},${y(b)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linejoin="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  }
  if (comparison && comparison !== r) line(comparison.curves, 'test_acc', '#bad7c6', 3, '5 5');
  line(r.curves, 'train_acc', '#aaa', 2);
  line(r.curves, 'test_acc', '#188257', 4);
  if (!r.curves.length) svg += '<text x="415" y="135" text-anchor="middle" fill="#aaa" font-size="17">Waiting for measurements</text>';
  $('chart').innerHTML = svg;
}
function drawMap(format) {
  const nodes = [...rounds.values()].filter(r=>r.hypothesis || r.curves.length).sort((a,b)=>a.round-b.round);
  const width = Math.max(1000, nodes.length * 225 + 100), positions = new Map();
  nodes.forEach((r,i) => positions.set(r.round, {x:110+i*225, y:r.round === 0 || r.verdict === 'supported' ? 48 : 88}));
  let svg = '';
  for (const r of nodes) {
    const p = positions.get(r.round), parent = positions.get(r.parent);
    if (r.round && parent) svg += `<path d="M${parent.x+17},${parent.y} C${p.x-100},${parent.y} ${p.x-100},${p.y} ${p.x-17},${p.y}" stroke="#d8d8d8" stroke-width="2" fill="none"/>`;
  }
  for (const r of nodes) {
    const p = positions.get(r.round), fill = ({supported:'#188257',refuted:'#bd5555',crashed:'#bd5555','no effect':'#e5e5e5',baseline:'#aaa'})[r.verdict] || '#bbb';
    const text = r.change || r.hypothesis || 'Baseline';
    svg += `<g tabindex="0" role="button" aria-label="Round ${r.round}: ${esc(text)}" data-round="${r.round}"><title>${esc(text)}</title><circle cx="${p.x}" cy="${p.y}" r="${best === r.round ? 22 : 18}" fill="white" stroke="${best === r.round ? '#181818' : selected === r.round ? '#888' : 'transparent'}" stroke-width="2"/><circle cx="${p.x}" cy="${p.y}" r="16" fill="${fill}"/><text x="${p.x}" y="${p.y+5}" text-anchor="middle" font-size="13" fill="${['supported','refuted','crashed'].includes(r.verdict) ? 'white' : '#333'}">${r.round}</text><text x="${p.x}" y="${p.y+39}" text-anchor="middle" fill="#333" font-size="14">${esc(text.length > 27 ? text.slice(0,26)+'…' : text)}</text><text x="${p.x}" y="${p.y+60}" text-anchor="middle" fill="#777" font-size="14">${format(r.value)}</text></g>`;
  }
  $('map').setAttribute('width', width); $('map').innerHTML = svg;
}
function inspect(event) {
  const node = event.target.closest('[data-round]');
  if (!node) return;
  selected = Number(node.dataset.round); follow = false; render();
}
for (const id of ['cards','map']) {
  $(id).onclick = inspect;
  $(id).onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(event); } };
}
async function submitIdea(hypothesis) {
  try { await api(`/api/runs/${runId}/idea`, {hypothesis}); notice('Idea queued for the next round.'); $('queue-status').textContent = 'Your idea is queued'; }
  catch (error) { notice(error.message); }
}
$('test').onclick = () => submitIdea($('message').value);
$('composer').onsubmit = async event => {
  event.preventDefault();
  const content = $('message').value.trim();
  if (!content || chatBusy || !runId || replay) return;
  messages.push({role:'user', content});
  const user = document.createElement('div'); user.className = 'chat user'; user.textContent = content; $('conversation').append(user);
  $('message').value = '';
  const reply = document.createElement('div'); reply.className = 'chat assistant'; reply.textContent = 'Thinking…'; $('conversation').append(reply); reply.scrollIntoView({block:'nearest'});
  chatBusy = true; controls(); let text = '';
  try {
    const response = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({run_id:runId, messages:messages.slice(-40)})});
    if (!response.ok) throw new Error((await response.json()).error);
    const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = '';
    while (true) {
      const {done,value} = await reader.read(); if (done) break;
      pending += decoder.decode(value, {stream:true});
      const lines = pending.split('\n'); pending = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ') || line.slice(6).trim() === '[DONE]') continue;
        const data = JSON.parse(line.slice(6));
        if (data.error) throw new Error(data.error.message || 'Model error');
        text += data.choices?.[0]?.delta?.content || ''; reply.textContent = text || 'Thinking…'; reply.scrollIntoView({block:'nearest'});
      }
    }
    if (!text) throw new Error('The model returned no reply. Try again.');
    messages.push({role:'assistant', content:text});
    const button = document.createElement('button'); button.className = 'secondary'; button.textContent = 'Test this idea'; button.disabled = replay || !active;
    button.onclick = () => submitIdea(text); reply.append(button);
  } catch (error) { reply.textContent = text || error.message; notice(error.message); }
  finally { chatBusy = false; controls(); }
};
$('start').onclick = async () => {
  $('start').disabled = true;
  try {
    const run = await api('/api/runs', {task:$('task').value, model});
    catalog.unshift({...run, active:true}); addOption(run); connect(run.id);
  } catch (error) { notice(error.message); controls(); }
};
for (const action of ['pause','resume','stop']) $(action).onclick = async () => {
  try {
    await api(`/api/runs/${runId}/${action}`, {}); paused = action === 'pause'; controls();
    notice(action === 'stop' ? 'Stop requested. The current step will finish first.' : action === 'pause' ? 'Pause requested. The current step will finish first.' : 'Resumed.');
  } catch (error) { notice(error.message); }
};
$('runs').onchange = () => {
  const url = new URL(location); url.searchParams.delete('replay'); url.searchParams.delete('speed');
  if ($('runs').value) url.searchParams.set('run', $('runs').value); else url.searchParams.delete('run');
  location.href = url;
};
function addOption(run) { const option = document.createElement('option'); option.value = run.id; option.textContent = `${run.task} · ${run.id}`; $('runs').append(option); }
async function init() {
  try {
    const response = await fetch('/api/runs'); if (!response.ok) throw new Error('Could not load runs');
    const data = await response.json(); catalog = data.runs; model = data.model;
    catalog.forEach(addOption);
    $('task').innerHTML = data.tasks.map(t=>`<option>${esc(t)}</option>`).join('');
    $('model').textContent = `${model} · running on this Mac`;
    if (replay) { $('replay').hidden = false; $('replay').textContent = `replay · ${speed}× speed`; }
    if (runId) connect(runId); else { render(); controls(); }
  } catch (error) { notice(error.message); }
}
init();
