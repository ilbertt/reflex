"""JEPA decoder testbed — FastAPI + SSE SPA.

Runs the 23-task JEPA suite across a chosen subset of decoders and
surfaces, live, the metrics that matter for this meta:

  * pass / fail per task, per decoder, per tier
  * cosine margin stats (mean / p10 / p50 / p90) per decoder
  * overrides / interventions taken by tiebreak and exec_verify
  * winning beam score for beam decoder

Transport is SSE + POST — works through plain HTTP / SSH tunnels.

Usage:
    python scripts/jepa_testbed_server.py --ckpt reflex.pt \
        --device cuda --port 8766

Then open http://localhost:8766.  Over SSH:
    ssh -L 8766:localhost:8766 <host>
    # then point browser at http://localhost:8766
"""
import argparse
import asyncio
import json
import queue as pyqueue
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

from reflex.decoders import DECODERS, get as get_decoder
from reflex.demo import load as ckpt_load
from reflex.model import MAX_INSTR_TOKENS
from scripts.jepa_testbed import (
    SHORT, MEDIUM, LONG, DISPLAY_TIER, _check, _display_read,
)


TIER_MAP = {'short': SHORT, 'medium': MEDIUM, 'long': LONG, 'display': DISPLAY_TIER}

DEFAULT_KWARGS = {
    'pure':            {},
    'tiebreak':        {'margin_eps': 0.15, 'halt_margin_eps': 0.30,
                        'topk': 5, 'min_cycle': 2, 'max_interventions': 2},
    'exec_verify':     {'margin_eps': 0.15, 'topk': 4, 'min_cycle': 2,
                        'max_interventions': 3, 'lookahead_depth': 6,
                        'scoring': 'margin'},
    'beam':            {'beam_width': 4, 'branching': 3},
    'rd_consistency':  {'margin_eps': 0.15, 'halt_margin_eps': 0.35,
                        'topk': 5, 'min_cycle': 2, 'lookback': 8,
                        'max_interventions': 3},
}


# ───────────────────────────── state ─────────────────────────────

class State:
    def __init__(self):
        self.model = None
        self.tok = None
        self.cfg = None
        self.device = 'cuda'
        self.ckpt_path = 'reflex.pt'
        self.events: asyncio.Queue | None = None
        self.running: bool = False
        self.last_results: dict | None = None


STATE = State()


async def _emit(event: dict):
    if STATE.events is None:
        return
    await STATE.events.put(event)


def _emit_threadsafe(loop, event: dict):
    if STATE.events is None:
        return
    asyncio.run_coroutine_threadsafe(STATE.events.put(event), loop)


# ───────────────────────────── workers ─────────────────────────────

def _pass(tier: str, cpu, halted, err, row) -> tuple[bool, str]:
    if tier == 'display':
        _, _, want, _ = row
        got = _display_read(cpu, len(want))
        return bool(halted and not err and got == want), got
    _, _, kind, expected, _ = row
    return bool(halted and not err and _check(cpu, kind, expected)), ''


def _collect_decoder_metrics(meta: dict) -> dict:
    """Extract a small set of decoder-specific numbers for the UI."""
    out = {'decoder': meta.get('decoder', '?')}
    if meta.get('overrides') is not None:
        out['overrides'] = len(meta['overrides'])
        out['override_details'] = meta['overrides']
    if meta.get('interventions') is not None:
        out['interventions'] = len(meta['interventions'])
        out['intervention_details'] = meta['interventions']
    if meta.get('winner_score') is not None:
        out['winner_score'] = meta['winner_score']
        out['n_completed'] = meta.get('n_completed', 0)
    return out


def _run_single(decoder_name: str, row, tier_name: str, kwargs: dict) -> dict:
    fn = get_decoder(decoder_name)
    tag = row[0]; prompt = row[1]; max_cycles = row[-1]
    seed_memcpy = (tier_name != 'display')
    kw = {**DEFAULT_KWARGS.get(decoder_name, {}), **kwargs}
    kw['max_instr_tokens'] = STATE.cfg.get('max_instr_tokens', MAX_INSTR_TOKENS)
    kw['use_chat_template'] = bool(STATE.cfg.get('chat_template', True))
    kw['use_context_prefix'] = bool(STATE.cfg.get('context_prefix', False))
    t0 = time.time()
    try:
        cpu, emitted, halted, err, meta = fn(
            STATE.model, STATE.tok, prompt, STATE.device, max_cycles,
            seed_memcpy=seed_memcpy, **kw)
        passed, got = _pass(tier_name, cpu, halted, err, row)
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        cpu, emitted, halted, err = None, [], False, f'EXC: {e}'
        passed, got, meta = False, '', {'decoder': decoder_name, 'exc': tb}
    dt = round(time.time() - t0, 2)
    return {
        'tag': tag, 'prompt': prompt, 'passed': passed,
        'ops': len(emitted), 'halted': halted, 'err': err, 'got': got,
        'elapsed_s': dt,
        'decoder_meta': _collect_decoder_metrics(meta),
    }


async def run_suite(decoders: list[str], tiers: list[str],
                     kwargs_by_decoder: dict[str, dict]):
    """Orchestrator: iterates decoders × tiers × tasks, runs each in a
    worker thread so FastAPI's event loop stays responsive, and emits
    SSE events at every state change."""
    loop = asyncio.get_running_loop()
    grand = {'decoders': decoders, 'tiers': tiers, 'results': {}}
    STATE.running = True
    await _emit({'type': 'start', 'decoders': decoders, 'tiers': tiers})
    try:
        for decoder in decoders:
            kw = {**DEFAULT_KWARGS.get(decoder, {}),
                  **(kwargs_by_decoder.get(decoder) or {})}
            await _emit({'type': 'decoder_start',
                         'decoder': decoder, 'kwargs': kw})
            dec_res = {'kwargs': kw, 'tiers': {}}
            for tier in tiers:
                tasks = TIER_MAP[tier]
                await _emit({'type': 'tier_start', 'decoder': decoder,
                             'tier': tier, 'total': len(tasks)})
                tier_rows = []
                passes = 0
                for idx, row in enumerate(tasks):
                    await _emit({'type': 'task_start',
                                 'decoder': decoder, 'tier': tier,
                                 'tag': row[0], 'idx': idx,
                                 'total': len(tasks)})
                    res = await asyncio.to_thread(
                        _run_single, decoder, row, tier,
                        kwargs_by_decoder.get(decoder) or {})
                    passes += int(res['passed'])
                    tier_rows.append(res)
                    await _emit({'type': 'task_done',
                                 'decoder': decoder, 'tier': tier,
                                 'idx': idx, 'total': len(tasks),
                                 'result': res})
                dec_res['tiers'][tier] = {'rows': tier_rows,
                                           'passes': passes,
                                           'total': len(tasks)}
                await _emit({'type': 'tier_done',
                             'decoder': decoder, 'tier': tier,
                             'passes': passes, 'total': len(tasks)})
            grand['results'][decoder] = dec_res
            await _emit({'type': 'decoder_done', 'decoder': decoder,
                         'result': dec_res})
        STATE.last_results = grand
        await _emit({'type': 'complete', 'results': grand})
    except Exception as e:
        await _emit({'type': 'error', 'msg': f'{e}',
                     'trace': traceback.format_exc()})
    finally:
        STATE.running = False


# ───────────────────────────── FastAPI ─────────────────────────────

app = FastAPI()

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Reflex JEPA decoder testbed</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0f1014; color: #d9dce0; margin: 0; padding: 20px; }
  h1 { color: #f0f3f7; font-size: 20px; margin: 0 0 8px 0; font-weight: 500;}
  h2 { color: #aab3bd; font-size: 14px; margin: 24px 0 6px 0; font-weight: 500;
       letter-spacing: 0.05em; text-transform: uppercase; }
  .controls { background: #181a22; padding: 14px 16px; border-radius: 6px;
              margin-bottom: 20px; border: 1px solid #23262e; }
  .controls label { display: inline-block; margin-right: 16px; cursor: pointer; }
  .controls input[type=checkbox] { accent-color: #8ab4ff; }
  button { background: #2a4e8f; color: #fff; border: none; padding: 8px 18px;
           border-radius: 4px; cursor: pointer; font: inherit; font-weight: 500; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button:hover:not(:disabled) { background: #3760ac; }
  .grid { display: grid; gap: 2px; margin-top: 10px;
          border: 1px solid #23262e; background: #23262e; border-radius: 4px;
          overflow: hidden; }
  .grid .header, .grid .row { display: contents; }
  .cell { padding: 6px 10px; background: #141620; font-size: 13px;
          min-width: 140px; }
  .cell.task { color: #aab3bd; text-align: left; }
  .cell.tier-head { background: #1b1d28; color: #aab3bd;
                    font-size: 11px; text-transform: uppercase;
                    letter-spacing: 0.1em; padding: 10px; }
  .cell.decoder-head { background: #1b1d28; color: #8ab4ff;
                       font-size: 11px; text-transform: uppercase;
                       letter-spacing: 0.1em; padding: 10px; text-align: center; }
  .cell.pass { background: #1c2d1c; color: #86d286; text-align: center; }
  .cell.fail { background: #301b1b; color: #e07474; text-align: center; }
  .cell.pending { background: #1a1d26; color: #575c6b; text-align: center; }
  .cell.running { background: #1f2a3a; color: #8ab4ff; text-align: center; }
  .cell.pass:hover, .cell.fail:hover { outline: 1px solid #8ab4ff;
                                        cursor: pointer; }
  .small { font-size: 11px; color: #777e8c; }
  .summary { margin-top: 16px; display: flex; gap: 20px; flex-wrap: wrap; }
  .card { background: #181a22; padding: 10px 14px; border-radius: 4px;
          border: 1px solid #23262e; min-width: 180px; }
  .card .label { font-size: 10px; text-transform: uppercase; color: #777e8c;
                 letter-spacing: 0.1em; }
  .card .value { font-size: 20px; color: #f0f3f7; margin-top: 2px; }
  .log { background: #0a0b0e; padding: 10px; border-radius: 4px; height: 180px;
         overflow-y: auto; font-size: 12px; border: 1px solid #23262e; }
  .log .evt { color: #575c6b; margin: 1px 0; }
  .log .evt.task-done { color: #aab3bd; }
  .detail { background: #181a22; padding: 14px; border-radius: 4px;
            border: 1px solid #23262e; margin-top: 10px; display: none;
            font-size: 12px; }
  .detail.open { display: block; }
  .detail pre { white-space: pre-wrap; word-break: break-all; color: #aab3bd;
                margin: 6px 0; }
  .flag { display: inline-block; padding: 1px 5px; border-radius: 2px;
          font-size: 10px; margin-left: 4px; vertical-align: middle;
          background: #3a2a1f; color: #e0a074; }
</style>
</head>
<body>
<h1>Reflex — JEPA decoder testbed</h1>
<div class="small">Checkpoint: <span id="ckpt"></span> · Device: <span id="device"></span>
    · <a href="/CONFESSION.md" style="color:#8ab4ff" target="_blank">CONFESSION.md</a></div>

<div class="controls">
  <div style="margin-bottom:10px;"><strong style="color:#aab3bd">Decoders:</strong>
    <label><input type="checkbox" class="dec" value="pure" checked> pure</label>
    <label><input type="checkbox" class="dec" value="tiebreak" checked> tiebreak</label>
    <label><input type="checkbox" class="dec" value="exec_verify" checked> exec_verify</label>
    <label><input type="checkbox" class="dec" value="rd_consistency" checked> rd_consistency</label>
    <label><input type="checkbox" class="dec" value="beam"> beam <span class="flag">heavy</span></label>
  </div>
  <div style="margin-bottom:10px;"><strong style="color:#aab3bd">Tiers:</strong>
    <label><input type="checkbox" class="tier" value="short" checked> short</label>
    <label><input type="checkbox" class="tier" value="medium" checked> medium</label>
    <label><input type="checkbox" class="tier" value="long" checked> long</label>
    <label><input type="checkbox" class="tier" value="display" checked> display</label>
  </div>
  <button id="run">Run</button>
  <span id="status" style="margin-left:14px; color:#aab3bd;"></span>
</div>

<h2>Results</h2>
<div id="grid-wrap"></div>

<h2>Summary</h2>
<div class="summary" id="summary"></div>

<div class="detail" id="detail"></div>

<h2>Event log</h2>
<div class="log" id="log"></div>

<script>
const DECODERS = ['pure','tiebreak','exec_verify','beam'];
const LOG = document.getElementById('log');
const STATUS = document.getElementById('status');
const SUMMARY = document.getElementById('summary');
const DETAIL = document.getElementById('detail');
const GRID = document.getElementById('grid-wrap');

let RESULTS = {};

function log(msg, klass='') {
  const d = document.createElement('div');
  d.className = 'evt ' + klass;
  d.textContent = new Date().toLocaleTimeString() + '  ' + msg;
  LOG.appendChild(d);
  LOG.scrollTop = LOG.scrollHeight;
}

function fetchCfg() {
  fetch('/cfg').then(r=>r.json()).then(j=>{
    document.getElementById('ckpt').textContent = j.ckpt;
    document.getElementById('device').textContent = j.device;
  });
}

function selectedDecoders() {
  return Array.from(document.querySelectorAll('.dec:checked')).map(e=>e.value);
}
function selectedTiers() {
  return Array.from(document.querySelectorAll('.tier:checked')).map(e=>e.value);
}

function ensureGrid(decoders, tiers) {
  GRID.innerHTML = '';
  RESULTS = {};
  for (const d of decoders) RESULTS[d] = {tiers:{}};
  // Column layout: one column per decoder.
  for (const tier of tiers) {
    const tierLabel = document.createElement('h2');
    tierLabel.textContent = tier;
    GRID.appendChild(tierLabel);
    const tierTasks = TIER_TASKS[tier] || [];
    const grid = document.createElement('div');
    grid.className = 'grid';
    grid.style.gridTemplateColumns = 'minmax(160px,1fr) ' +
      decoders.map(()=>'minmax(120px,1fr)').join(' ');
    // Header row
    const cornerCell = document.createElement('div');
    cornerCell.className = 'cell tier-head';
    cornerCell.textContent = 'task';
    grid.appendChild(cornerCell);
    for (const d of decoders) {
      const h = document.createElement('div');
      h.className = 'cell decoder-head';
      h.textContent = d;
      grid.appendChild(h);
    }
    // Task rows
    for (const task of tierTasks) {
      const lbl = document.createElement('div');
      lbl.className = 'cell task';
      lbl.textContent = task;
      grid.appendChild(lbl);
      for (const d of decoders) {
        const c = document.createElement('div');
        c.className = 'cell pending';
        c.id = `cell-${d}-${tier}-${task}`;
        c.textContent = '—';
        c.addEventListener('click', ()=>showDetail(d, tier, task));
        grid.appendChild(c);
      }
    }
    GRID.appendChild(grid);
  }
}

function setCell(d, tier, tag, state, text) {
  const id = `cell-${d}-${tier}-${tag}`;
  const c = document.getElementById(id);
  if (!c) return;
  c.className = 'cell ' + state;
  c.textContent = text;
}

function updateSummary() {
  SUMMARY.innerHTML = '';
  for (const d of Object.keys(RESULTS)) {
    const r = RESULTS[d];
    let passes = 0, total = 0;
    for (const tier of Object.keys(r.tiers || {})) {
      for (const row of (r.tiers[tier].rows || [])) {
        total++;
        if (row.passed) passes++;
      }
    }
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<div class="label">${d}</div>
      <div class="value">${passes} / ${total}</div>
      <div class="small">${total ? Math.round(100*passes/total) : 0}% pass</div>`;
    SUMMARY.appendChild(card);
  }
}

function showDetail(d, tier, tag) {
  const rows = (RESULTS[d]?.tiers?.[tier]?.rows) || [];
  const row = rows.find(r=>r.tag===tag);
  if (!row) return;
  DETAIL.className = 'detail open';
  DETAIL.innerHTML = `<strong>${d} / ${tier} / ${tag}</strong>
    <pre>${JSON.stringify(row, null, 2)}</pre>`;
}

async function run() {
  const decoders = selectedDecoders();
  const tiers = selectedTiers();
  if (!decoders.length || !tiers.length) { alert('Pick at least one'); return; }
  const btn = document.getElementById('run');
  btn.disabled = true;
  STATUS.textContent = 'starting…';
  LOG.innerHTML = '';
  ensureGrid(decoders, tiers);
  const r = await fetch('/run', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({decoders, tiers})
  });
  const j = await r.json();
  if (j.status !== 'ok') { STATUS.textContent = 'error: ' + j.msg; btn.disabled=false; return; }
  streamEvents();
}

function streamEvents() {
  const s = new EventSource('/events');
  s.onmessage = (ev) => {
    const e = JSON.parse(ev.data);
    if (e.type === 'start') {
      log('start: ' + e.decoders.join(', '));
    } else if (e.type === 'decoder_start') {
      log('→ decoder ' + e.decoder);
      STATUS.textContent = 'running ' + e.decoder;
    } else if (e.type === 'tier_start') {
      log('  tier ' + e.tier + ' (' + e.total + ' tasks)');
    } else if (e.type === 'task_start') {
      setCell(e.decoder, e.tier, e.tag, 'running', '…');
    } else if (e.type === 'task_done') {
      const res = e.result;
      RESULTS[e.decoder] = RESULTS[e.decoder] || {tiers:{}};
      RESULTS[e.decoder].tiers[e.tier] = RESULTS[e.decoder].tiers[e.tier] || {rows: []};
      RESULTS[e.decoder].tiers[e.tier].rows.push(res);
      let text = res.passed ? '✓' : '✗';
      const meta = res.decoder_meta || {};
      if (meta.overrides) text += ` (${meta.overrides}o)`;
      else if (meta.interventions) text += ` (${meta.interventions}v)`;
      text += `  ${res.ops}op ${res.elapsed_s}s`;
      setCell(e.decoder, e.tier, res.tag, res.passed ? 'pass' : 'fail', text);
      log(`    ${res.passed?'✓':'✗'} ${res.tag}  ops=${res.ops}  ${res.elapsed_s}s`, 'task-done');
      updateSummary();
    } else if (e.type === 'tier_done') {
      log('  tier ' + e.tier + ': ' + e.passes + '/' + e.total);
    } else if (e.type === 'decoder_done') {
      log('✓ decoder ' + e.decoder + ' done');
    } else if (e.type === 'complete') {
      log('★ complete');
      STATUS.textContent = 'done';
      document.getElementById('run').disabled = false;
      s.close();
    } else if (e.type === 'error') {
      log('ERROR: ' + e.msg);
      STATUS.textContent = 'error';
      document.getElementById('run').disabled = false;
      s.close();
    }
  };
  s.onerror = () => { s.close(); document.getElementById('run').disabled = false; };
}

const TIER_TASKS = ${TIER_TASKS_JSON};

document.getElementById('run').addEventListener('click', run);
fetchCfg();
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
async def index():
    tier_tasks = {name: [row[0] for row in tasks]
                  for name, tasks in TIER_MAP.items()}
    html = INDEX_HTML.replace('${TIER_TASKS_JSON}', json.dumps(tier_tasks))
    return HTMLResponse(html)


@app.get('/cfg')
async def cfg():
    return JSONResponse({'ckpt': STATE.ckpt_path, 'device': STATE.device,
                         'num_instrs': STATE.cfg.get('num_instrs') if STATE.cfg else None,
                         'embed_dim': STATE.cfg.get('embed_dim') if STATE.cfg else None})


@app.get('/CONFESSION.md', response_class=HTMLResponse)
async def confession():
    p = Path(__file__).parent.parent / 'CONFESSION.md'
    if not p.exists():
        return HTMLResponse('<pre>CONFESSION.md not found</pre>')
    return HTMLResponse(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>CONFESSION</title>'
        '<style>body{background:#0f1014;color:#d9dce0;'
        'font-family:ui-monospace,monospace;padding:30px;max-width:820px;'
        'margin:0 auto;line-height:1.55}'
        'h1,h2,h3{color:#f0f3f7}a{color:#8ab4ff}'
        'pre,code{background:#181a22;padding:2px 4px;border-radius:3px}'
        '</style></head><body><pre style="white-space:pre-wrap">'
        + p.read_text().replace('<', '&lt;')
        + '</pre></body></html>')


@app.post('/run')
async def run_endpoint(body: dict):
    if STATE.running:
        return JSONResponse({'status': 'busy', 'msg': 'a run is already in progress'})
    decoders = body.get('decoders') or ['pure']
    tiers = body.get('tiers') or ['short', 'medium', 'long', 'display']
    kwargs = body.get('kwargs_by_decoder') or {}
    unknown = [d for d in decoders if d not in DECODERS]
    if unknown:
        return JSONResponse({'status': 'error',
                             'msg': f'unknown decoder(s): {unknown}'})
    STATE.events = asyncio.Queue()
    asyncio.create_task(run_suite(decoders, tiers, kwargs))
    return JSONResponse({'status': 'ok'})


@app.get('/events')
async def events():
    async def stream():
        if STATE.events is None:
            STATE.events = asyncio.Queue()
        while True:
            ev = await STATE.events.get()
            yield f'data: {json.dumps(ev)}\n\n'
            if ev.get('type') in ('complete', 'error'):
                break
    return StreamingResponse(stream(), media_type='text/event-stream',
                              headers={'Cache-Control': 'no-cache',
                                       'X-Accel-Buffering': 'no'})


@app.get('/last_results')
async def last_results():
    return JSONResponse(STATE.last_results or {})


# ───────────────────────────── entry ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='reflex.pt')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--port', type=int, default=8766)
    ap.add_argument('--host', default='0.0.0.0')
    args = ap.parse_args()

    print(f'Loading {args.ckpt} on {args.device}…', flush=True)
    STATE.model, STATE.tok, STATE.cfg = ckpt_load(args.ckpt, args.device)
    STATE.device = args.device
    STATE.ckpt_path = args.ckpt
    print(f'Loaded. num_instrs={STATE.cfg.get("num_instrs")} '
          f'embed_dim={STATE.cfg.get("embed_dim")}', flush=True)
    print(f'Serving on http://{args.host}:{args.port}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
