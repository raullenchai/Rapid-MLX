const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const replay = params.get('replay') === '1';
const speed = Math.max(0.1, Math.min(1000, Number(params.get('speed')) || 8));
let runId = params.get('run') || '', source, selected = 0, follow = true, best = 0;
let rounds = new Map(), messages = [], active = false, paused = false, chatBusy = false;
let catalog = [], model = '';
let expandedRound = null, displayedRound = null;
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
const statusClass = value => value === 'testing…' ? 'testing' : (value || '').replaceAll(' ', '-');
const verdictColors = {supported:'#176345', refuted:'#a33432', crashed:'#a33432', 'no effect':'#795919', baseline:'#526476'};
const verdictColor = verdict => verdictColors[verdict] || '#285e89';
const changeLabel = value => String(value || '').replaceAll('->', '→');
let noticeTimer;
function notice(text = '') { $('notice').textContent = text; clearTimeout(noticeTimer); if (text) noticeTimer = setTimeout(() => { $('notice').textContent = ''; }, 6000); }
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
  expandedRound = displayedRound = null;
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
      $('title').title = $('title').textContent;
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
  if (displayedRound !== selected) {
    $('changes').open = false; $('hypothesis-details').open = false;
    displayedRound = selected;
  }
  $('round-label').textContent = selected ? `Round ${selected}${follow ? ' · following live' : ' · selected'}` : 'Baseline';
  $('change').textContent = changeLabel(r.change) || (selected ? 'Hypothesis proposed' : runId ? 'Measuring the baseline…' : 'A small experiment, measured.');
  $('change').title = $('change').textContent;
  $('hypothesis').textContent = r.hypothesis || 'Choose a round to inspect its evidence.';
  $('hypothesis').title = r.hypothesis || '';
  $('hypothesis-full').textContent = [r.hypothesis, r.why].filter(Boolean).join('\n\n');
  $('hypothesis-details').hidden = !r.hypothesis;
  $('follow').hidden = follow || !rounds.size;
  $('experiment-status').textContent = !selected && r.verdict === 'testing…' ? 'measuring…' : r.verdict;
  $('experiment-status').className = `badge ${statusClass(r.verdict)}`;
  $('seeds').innerHTML = Object.entries(r.seeds).map(([seed,state]) => `<span class="seed">Seed ${esc(seed)} · ${state === 'done' ? 'complete' : 'running…'}</span>`).join('');
  const metric = rounds.get(0)?.metric || 'test_acc';
  const accuracy = metric.endsWith('acc');
  const format = value => accuracy ? pct(value) : Number.isFinite(value) ? String(value) : '—';
  const label = metric === 'test_acc' ? 'Test accuracy' : metric.replaceAll('_', ' ');
  $('chart-title').textContent = accuracy ? 'Accuracy' : label;
  $('verdict').textContent = Number.isFinite(r.value) ? `${Number.isFinite(r.best_before) ? format(r.best_before) + ' → ' : ''}${format(r.value)} · ${r.verdict}` : r.verdict === 'testing…' ? (runId ? selected ? 'Testing the idea…' : 'Measuring the baseline…' : 'Results will guide the next idea.') : r.verdict;
  if (Number.isFinite(r.value)) {
    const values = `${Number.isFinite(r.best_before) ? format(r.best_before) + ' → ' : ''}${format(r.value)}`;
    $('verdict').innerHTML = `<span>${esc(values)}</span><span class="verdict-word"> · ${esc(r.verdict)}</span>`;
  }
  $('verdict').className = !runId && r.verdict === 'testing…' ? 'ready' : statusClass(r.verdict);
  $('chart-test-key').setAttribute('style', `border-color:${verdictColor(r.verdict)}`);
  $('diff-label').textContent = r.diffs.some(d => d.kind === 'fix') ? 'crashed → fixed by the model' + (Object.values(r.seeds).includes('done') ? ' → ran' : '') : '';
  const diffLines = r.diffs.flatMap(d => d.diff.split('\n'));
  const files = [...new Set(diffLines.filter(line => line.startsWith('+++ ')).map(line => line.slice(4).replace(/^b\//, '').split('\t')[0]))];
  const added = diffLines.filter(line => line.startsWith('+') && !line.startsWith('+++')).length;
  const removed = diffLines.filter(line => line.startsWith('-') && !line.startsWith('---')).length;
  $('diff-summary').textContent = r.diffs.length ? `${files.join(', ') || 'Experiment edit'} · +${added} −${removed}` : 'What changed · awaiting edit';
  $('diff-summary').title = $('diff-summary').textContent;
  $('diff').innerHTML = r.diffs.length ? r.diffs.map(d => `<span>${d.kind === 'fix' ? 'Model’s fix' : 'Experiment edit'}</span>\n` + d.diff.split('\n').map(line => `<span class="${line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : ''}">${esc(line)}</span>`).join('\n')).join('\n\n') : 'The model’s exact edit will appear here.';
  $('errors').innerHTML = r.errors.map(e => `<details class="error"><summary>${esc(e.split('\n')[0].slice(0,160))}</summary><pre>${esc(e)}</pre></details>`).join('');
  const scroll = $('rounds-panel'), nearEnd = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 90;
  const ideas = [...rounds.values()].filter(r => r.round && r.hypothesis);
  const cards = ideas.map(r => `<article class="card ${selected === r.round ? 'selected' : ''}"><button type="button" id="idea-${r.round}" class="idea-row" aria-expanded="${expandedRound === r.round}" aria-controls="idea-detail-${r.round}" aria-pressed="${selected === r.round}" aria-label="Inspect round ${r.round}: ${esc(changeLabel(r.change) || r.hypothesis)}, ${esc(r.verdict)}" data-round="${r.round}"><span class="round-number" title="Round ${r.round}">${r.round}</span><span class="idea-change" title="${esc(changeLabel(r.change) || r.hypothesis)}">${esc(changeLabel(r.change) || 'Hypothesis proposed')}</span><span class="outcome ${statusClass(r.verdict)}">${esc(r.verdict)}${Number.isFinite(r.value) ? ' · ' + format(r.value) : ''}</span></button><div id="idea-detail-${r.round}" class="idea-detail"${expandedRound === r.round ? '' : ' hidden'}><h3>${esc(r.hypothesis)}</h3><p>${esc(r.why)}</p><span class="badge">${r.source === 'human' ? 'you' : 'model'}</span></div></article>`).join('');
  if ($('cards').innerHTML !== cards) {
    const focused = document.activeElement?.closest?.('.idea-row')?.id;
    $('cards').innerHTML = cards || '<p class="welcome">New hypotheses will appear here, one round at a time. Discuss or propose an idea below.</p>';
    if (focused) $(focused)?.focus({preventScroll:true});
  }
  $('idea-count').textContent = `${ideas.length} hypotheses`;
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
  const width = $('chart').clientWidth || 1000, height = $('chart').clientHeight || 285;
  const right = width - 38, bottom = height - 53;
  const x = step => 62 + step / max * (right - 62), y = value => bottom - value * (bottom - 22);
  $('chart').setAttribute('viewBox', `0 0 ${width} ${height}`);
  const color = verdictColor(r.verdict);
  let svg = [0,.25,.5,.75,1].map(v => `<line x1="62" x2="${right}" y1="${y(v)}" y2="${y(v)}" stroke="#dce3e9"/><text x="48" y="${y(v)+5}" text-anchor="end" fill="#526476" font-size="15">${v*100}%</text>`).join('');
  for (let i=0;i<=5;i++) svg += `<text x="${x(max*i/5)}" y="${height-29}" text-anchor="middle" fill="#526476" font-size="14">${Math.round(max*i/5).toLocaleString()}</text>`;
  svg += `<text x="${right}" y="${height-5}" text-anchor="end" fill="#526476" font-size="13">training steps · mean across seeds</text>`;
  function line(points, field, stroke, width, dash='') {
    const data = series(points, field);
    if (data.length) svg += `<polyline points="${data.map(([a,b])=>`${x(a)},${y(b)}`).join(' ')}" fill="none" stroke="${stroke}" stroke-width="${width}" stroke-linejoin="round" stroke-linecap="round" ${dash ? `stroke-dasharray="${dash}"` : ''}/>`;
  }
  // Draw the comparison first; a solid test curve remains visible at a shared value.
  if (comparison && comparison !== r) line(comparison.curves, 'test_acc', '#697684', 2, '7 6');
  line(r.curves, 'train_acc', '#66798b', 2);
  line(r.curves, 'test_acc', color, 4);
  const test = series(r.curves, 'test_acc').at(-1);
  if (test) svg += `<circle cx="${x(test[0])}" cy="${y(test[1])}" r="5" fill="${color}" stroke="white" stroke-width="2"/>`;
  if (!r.curves.length) svg += `<text x="${(62+right)/2}" y="${height/2-10}" text-anchor="middle" fill="#526476" font-size="18">${!selected && runId ? 'Measuring the baseline…' : 'Waiting for measurements'}</text>`;
  $('chart').innerHTML = svg;
  $('chart').setAttribute('aria-label', `Round ${r.round}: train and test accuracy over training steps. ${Number.isFinite(r.value) ? `Test accuracy ${pct(r.value)}. ` : ''}${r.verdict}.`);
}
function drawMap(format) {
  const nodes = [...rounds.values()].filter(r=>r.hypothesis || r.curves.length).sort((a,b)=>a.round-b.round);
  const width = Math.max($('map').clientWidth || 1000, nodes.length * 150), positions = new Map();
  const step = width / Math.max(1, nodes.length);
  nodes.forEach((r,i) => positions.set(r.round, {x:step*(i+.5), y:r.round === 0 || r.verdict === 'supported' ? 23 : 41}));
  let svg = '';
  for (const r of nodes) {
    const p = positions.get(r.round), parent = positions.get(r.parent);
    if (r.round && parent) svg += `<path d="M${parent.x+18},${parent.y} C${p.x-step*.5},${parent.y} ${p.x-step*.5},${p.y} ${p.x-18},${p.y}" stroke="#9baab7" stroke-width="1.5" fill="none"/>`;
  }
  for (const r of nodes) {
    const p = positions.get(r.round), color = verdictColor(r.verdict);
    const text = changeLabel(r.change) || r.hypothesis || 'Baseline';
    svg += `<g tabindex="0" role="button" aria-pressed="${selected === r.round}" aria-label="Inspect ${r.round ? 'round '+r.round : 'baseline'}: ${esc(text)}, ${esc(r.verdict)}, ${format(r.value)}" data-round="${r.round}"><title>${esc(text)} · ${esc(r.verdict)}</title><circle cx="${p.x}" cy="${p.y}" r="22" fill="#eef2f5" stroke="${best === r.round ? '#181818' : 'transparent'}" stroke-width="2"/><circle cx="${p.x}" cy="${p.y}" r="18" fill="#eef2f5" stroke="${selected === r.round ? '#285e89' : 'transparent'}" stroke-width="2"/><circle class="node-fill" cx="${p.x}" cy="${p.y}" r="14" fill="${color}"/><text x="${p.x}" y="${p.y+4.5}" text-anchor="middle" font-size="12" font-weight="600" fill="white">${r.round}</text><text x="${p.x}" y="${p.y+36}" text-anchor="middle" fill="#172b3b" font-size="13">${esc(text.length > 17 ? text.slice(0,16)+'…' : text)}</text><text x="${p.x}" y="${p.y+51}" text-anchor="middle" fill="${color}" font-size="12">${format(r.value)} · ${esc(r.verdict)}</text></g>`;
  }
  if (!nodes.length) svg = '<text x="0" y="64" fill="#526476" font-size="16">Each experiment becomes a node. Choose one to inspect its evidence.</text>';
  $('map').setAttribute('viewBox', `0 0 ${width} 104`);
  $('map').setAttribute('style', `min-width:${width}px`);
  $('map').innerHTML = svg;
}
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(() => {
    drawChart(round(selected));
    const metric = rounds.get(0)?.metric || 'test_acc';
    drawMap(value => metric.endsWith('acc') ? pct(value) : Number.isFinite(value) ? String(value) : '—');
  }).observe($('chart'));
}
function inspect(event) {
  const node = event.target.closest('[data-round]');
  if (!node) return;
  selected = Number(node.dataset.round); follow = false; render();
}
$('follow').onclick = () => {
  selected = Math.max(0, ...rounds.keys()); follow = true; render();
};
$('cards').onclick = event => {
  const row = event.target.closest('[data-round]');
  if (!row) return;
  const n = Number(row.dataset.round);
  expandedRound = expandedRound === n ? null : n;
  inspect(event);
};
$('map').onclick = inspect;
$('map').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspect(event); } };
function showIdeaTab(name) {
  for (const tab of ['rounds', 'discussion']) {
    const chosen = tab === name;
    $(`${tab}-tab`).setAttribute('aria-selected', String(chosen));
    $(`${tab}-tab`).tabIndex = chosen ? 0 : -1;
    $(`${tab}-panel`).hidden = !chosen;
  }
}
for (const name of ['rounds', 'discussion']) {
  $(`${name}-tab`).onclick = () => showIdeaTab(name);
  $(`${name}-tab`).onkeydown = event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 'rounds' : event.key === 'End' ? 'discussion' : name === 'rounds' ? 'discussion' : 'rounds';
    showIdeaTab(next); $(`${next}-tab`).focus();
  };
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
  showIdeaTab('discussion');
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
