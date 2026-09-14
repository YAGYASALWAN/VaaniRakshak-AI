'use strict';

const el = id => document.getElementById(id);
let ws = null;
let stream = null;
let audioContext = null;
let sourceNode = null;
let workletNode = null;
let muteNode = null;
let timerId = null;
let startedAt = null;
let stopping = false;
let backendReady = false;

function setMessage(text, error = false) {
  el('message').textContent = text;
  el('message').classList.toggle('error', error);
}

function fmtTime(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  const mins = String(Math.floor(whole / 60)).padStart(2, '0');
  const secs = String(whole % 60).padStart(2, '0');
  return `${mins}:${secs}`;
}

function setConnection(text, active = false) {
  el('connection-pill').textContent = text;
  el('connection-pill').classList.toggle('active', active);
}

function resetLiveUI() {
  el('call-state').textContent = 'Ready';
  el('timer').textContent = '00:00';
  el('risk-score').textContent = '--';
  el('risk-label').textContent = 'No evidence yet';
  el('risk-fill').style.width = '0%';
  el('duration').textContent = '0.0 s';
  el('segments').textContent = '0';
  el('suspicious').textContent = '0';
  el('timeline').innerHTML = '<div class="empty">No segments yet.</div>';
  el('final-card').hidden = true;
}

function renderSummary(summary) {
  el('risk-score').textContent = summary.segments_analyzed ? String(summary.risk_score) : '--';
  el('risk-label').textContent = summary.risk_label;
  el('risk-fill').style.width = `${summary.risk_score}%`;
  el('duration').textContent = `${summary.duration_seconds.toFixed(1)} s`;
  el('segments').textContent = String(summary.segments_analyzed);
  el('suspicious').textContent = String(summary.suspicious_segments);
}

function addSegment(segment) {
  const timeline = el('timeline');
  timeline.querySelector('.empty')?.remove();
  const block = document.createElement('div');
  block.className = segment.synthetic_score >= 0.65 ? 'segment high' : segment.synthetic_score >= 0.5 ? 'segment mid' : 'segment low';
  block.title = `${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} s · score ${segment.synthetic_score.toFixed(3)}`;
  block.setAttribute('aria-label', block.title);
  timeline.appendChild(block);
}

function renderFinal(result) {
  renderSummary(result);
  el('final-score').textContent = `${result.risk_score}/100`;
  el('final-risk').textContent = result.risk_label;
  el('final-verdict').textContent = result.verdict;
  el('final-copy').textContent = result.segments_analyzed
    ? `${result.suspicious_segments} of ${result.segments_analyzed} analyzed segments crossed the current suspicious-score threshold.`
    : 'Not enough audio was available to produce a call-level result.';
  el('final-notice').textContent = result.notice;

  const regions = el('regions');
  regions.innerHTML = '';
  if (result.regions?.length) {
    const heading = document.createElement('h3');
    heading.textContent = 'Highest-scoring regions';
    regions.appendChild(heading);
    for (const region of result.regions) {
      const row = document.createElement('div');
      row.className = 'region-row';
      row.innerHTML = `<span>${region.start_seconds.toFixed(1)}–${region.end_seconds.toFixed(1)} s</span><strong>${region.synthetic_score.toFixed(3)}</strong>`;
      regions.appendChild(row);
    }
  }
  el('final-card').hidden = false;
  el('final-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function cleanupAudio() {
  if (timerId) clearInterval(timerId);
  timerId = null;
  workletNode?.disconnect();
  sourceNode?.disconnect();
  muteNode?.disconnect();
  stream?.getTracks().forEach(track => track.stop());
  if (audioContext && audioContext.state !== 'closed') await audioContext.close();
  workletNode = sourceNode = muteNode = stream = audioContext = null;
  startedAt = null;
}

function closeSocket() {
  if (ws && ws.readyState < WebSocket.CLOSING) ws.close();
  ws = null;
}

async function failActiveSession(text) {
  setMessage(text, true);
  setConnection('Stream error');
  el('call-state').textContent = 'Session stopped';
  await cleanupAudio();
  closeSocket();
  stopping = false;
  el('start').disabled = false;
  el('stop').disabled = true;
}

async function loadStatus() {
  try {
    const response = await fetch('/api/v2/status', { cache: 'no-store' });
    const status = await response.json();
    if (!response.ok) throw new Error(status.detail || 'V2 backend unavailable.');
    backendReady = status.ready;
    el('mode').textContent = status.analysis_mode.toUpperCase();
    el('notice').textContent = status.notice;
    el('start').disabled = !backendReady;
    setConnection('Backend ready', true);
    setMessage('Ready. Start a microphone analysis session.');
  } catch (error) {
    backendReady = false;
    el('start').disabled = true;
    el('mode').textContent = 'Offline';
    setConnection('Backend offline');
    setMessage(error.message, true);
  }
}

async function startAnalysis() {
  if (!backendReady || stopping) return;
  resetLiveUI();
  el('start').disabled = true;
  el('stop').disabled = true;
  stopping = false;

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false
    });
    audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule('/assets/pcm-worklet.js');

    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    ws = new WebSocket(`${scheme}://${location.host}/ws/v2/analyze`);
    ws.binaryType = 'arraybuffer';

    ws.onmessage = async event => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        await failActiveSession('Backend returned an invalid streaming message.');
        return;
      }
      if (data.type === 'connected') return;
      if (data.type === 'started') {
        el('call-state').textContent = 'Call in progress';
        el('stop').disabled = false;
        setConnection('Streaming', true);
        setMessage('Microphone audio is streaming to the local V2 backend.');
        el('notice').textContent = data.notice;
        startedAt = performance.now();
        timerId = setInterval(() => {
          el('timer').textContent = fmtTime((performance.now() - startedAt) / 1000);
        }, 250);
        return;
      }
      if (data.type === 'segment') {
        addSegment(data.segment);
        renderSummary(data.summary);
        return;
      }
      if (data.type === 'final') {
        renderFinal(data);
        el('call-state').textContent = 'Analysis complete';
        setConnection('Complete', true);
        setMessage('Final call report generated.');
        await cleanupAudio();
        el('start').disabled = false;
        el('stop').disabled = true;
        stopping = false;
        return;
      }
      if (data.type === 'error') {
        await failActiveSession(data.message || 'Streaming analysis error.');
      }
    };

    ws.onerror = () => setMessage('WebSocket connection error. Check the backend terminal.', true);
    ws.onclose = async event => {
      if (!stopping && el('call-state').textContent === 'Call in progress' && event.code !== 1000) {
        await failActiveSession('Streaming connection closed unexpectedly.');
      }
    };

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Timed out connecting to the V2 backend.')), 5000);
      ws.addEventListener('open', () => { clearTimeout(timeout); resolve(); }, { once: true });
      ws.addEventListener('error', () => { clearTimeout(timeout); reject(new Error('Could not open the V2 streaming connection.')); }, { once: true });
    });

    ws.send(JSON.stringify({ type: 'start', sample_rate: audioContext.sampleRate }));
    sourceNode = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, 'pcm16-capture');
    muteNode = audioContext.createGain();
    muteNode.gain.value = 0;
    workletNode.port.onmessage = event => {
      if (ws?.readyState === WebSocket.OPEN && event.data instanceof ArrayBuffer) ws.send(event.data);
    };
    sourceNode.connect(workletNode);
    workletNode.connect(muteNode);
    muteNode.connect(audioContext.destination);
  } catch (error) {
    await cleanupAudio();
    closeSocket();
    el('start').disabled = false;
    el('stop').disabled = true;
    setConnection('Ready', backendReady);
    setMessage(error.message || 'Could not start microphone analysis.', true);
  }
}

async function stopAnalysis() {
  if (!ws || ws.readyState !== WebSocket.OPEN || stopping) return;
  stopping = true;
  el('stop').disabled = true;
  el('call-state').textContent = 'Finalizing';
  setMessage('Flushing the final audio and aggregating call evidence…');
  workletNode?.port.postMessage({ type: 'flush' });
  await new Promise(resolve => setTimeout(resolve, 80));
  ws.send(JSON.stringify({ type: 'stop' }));
}

el('start').addEventListener('click', startAnalysis);
el('stop').addEventListener('click', stopAnalysis);
window.addEventListener('beforeunload', () => {
  stream?.getTracks().forEach(track => track.stop());
  closeSocket();
});

resetLiveUI();
loadStatus();
