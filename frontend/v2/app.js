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

function renderSpeechGate(value) {
  if (!value) return;
  el('speech-gate').textContent = Array.isArray(value) ? value.join(', ') : String(value);
}

function resetLiveUI() {
  el('call-state').textContent = 'Ready';
  el('timer').textContent = '00:00';
  el('risk-score').textContent = '--';
  el('risk-label').textContent = 'No evidence yet';
  el('risk-fill').style.width = '0%';
  el('duration').textContent = '0.0 s';
  el('usable-speech').textContent = '0.0 s';
  el('windows-seen').textContent = '0';
  el('segments').textContent = '0';
  el('skipped').textContent = '0';
  el('suspicious').textContent = '0';
  el('timeline').innerHTML = '<div class="empty">No windows yet.</div>';
  el('final-card').hidden = true;
}

function renderSummary(summary) {
  const showRisk = Boolean(summary.enough_evidence);
  el('risk-score').textContent = showRisk ? String(summary.risk_score) : '--';
  el('risk-label').textContent = summary.risk_label;
  el('risk-fill').style.width = showRisk ? `${summary.risk_score}%` : '0%';
  el('duration').textContent = `${summary.duration_seconds.toFixed(1)} s`;
  el('usable-speech').textContent = `${summary.usable_speech_seconds.toFixed(1)} s`;
  el('windows-seen').textContent = String(summary.windows_seen);
  el('segments').textContent = String(summary.segments_analyzed);
  el('skipped').textContent = String(summary.segments_skipped);
  el('suspicious').textContent = String(summary.suspicious_segments);
  renderSpeechGate(summary.speech_gate);
  if (summary.window_seconds && summary.hop_seconds) {
    el('windowing').textContent = `${summary.window_seconds.toFixed(0)}s / ${summary.hop_seconds.toFixed(0)}s`;
  }
}

function addSegment(segment) {
  const timeline = el('timeline');
  timeline.querySelector('.empty')?.remove();
  const block = document.createElement('div');
  const gate = segment.quality?.speech_gate || 'speech gate';

  if (!segment.analyzed || segment.synthetic_score == null) {
    block.className = 'segment skipped';
    const reason = segment.quality?.reason || 'quality_gate';
    block.title = `${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} s · skipped (${reason}) · ${gate}`;
  } else {
    const evidence = Number.isFinite(segment.evidence_signal) ? segment.evidence_signal : (segment.suspicious ? 0.6 : 0.4);
    block.className = evidence >= 0.75 ? 'segment high' : evidence >= 0.5 ? 'segment mid' : 'segment low';
    block.title = `${segment.start_seconds.toFixed(1)}–${segment.end_seconds.toFixed(1)} s · score ${segment.synthetic_score.toFixed(3)} · threshold ${segment.threshold.toFixed(3)} · speech ${(segment.quality.speech_ratio * 100).toFixed(0)}% · ${gate}`;
  }
  block.setAttribute('aria-label', block.title);
  timeline.appendChild(block);
}

function renderFinal(result) {
  renderSummary(result);
  el('final-score').textContent = result.enough_evidence ? `${result.risk_score}/100` : '--';
  el('final-risk').textContent = result.risk_label;
  el('final-verdict').textContent = result.verdict;

  if (!result.enough_evidence) {
    el('final-copy').textContent = `Only ${result.usable_speech_seconds.toFixed(1)} seconds of usable speech across ${result.segments_analyzed} analyzed windows were available. VaaniRakshak refused to force a call-level verdict.`;
  } else {
    el('final-copy').textContent = `${result.suspicious_segments} of ${result.segments_analyzed} analyzed windows crossed detector threshold ${result.threshold.toFixed(3)}; ${result.segments_skipped} windows were excluded by the active speech/quality gate.`;
  }
  el('final-notice').textContent = result.notice;

  const regions = el('regions');
  regions.innerHTML = '';
  if (result.regions?.length) {
    const heading = document.createElement('h3');
    heading.textContent = 'Highest-scoring analyzed regions';
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

async function loadStatus() {
  try {
    const response = await fetch('/api/v2/status', { cache: 'no-store' });
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || status.detail || 'V2 backend unavailable.');
    backendReady = status.ready;
    el('mode').textContent = status.analysis_mode.toUpperCase();
    renderSpeechGate(status.quality_gate);
    el('notice').textContent = `${status.notice} Detector threshold: ${status.threshold.toFixed(3)}. Scores are${status.calibrated_probability ? '' : ' not'} calibrated probabilities.`;
    if (status.window_seconds && status.hop_seconds) el('windowing').textContent = `${status.window_seconds}s / ${status.hop_seconds}s`;
    el('start').disabled = !backendReady;
    setConnection('Backend ready', true);
    setMessage(status.analysis_mode === 'mock' ? 'Ready in mock integration mode.' : `Ready with detector ${status.model}.`);
  } catch (error) {
    backendReady = false;
    el('start').disabled = true;
    el('mode').textContent = 'Offline';
    el('speech-gate').textContent = 'Unavailable';
    setConnection('Backend offline');
    setMessage(error.message, true);
  }
}

async function handleStreamMessage(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'connected') {
    renderSpeechGate(data.speech_gate);
    return;
  }
  if (data.type === 'started') {
    el('call-state').textContent = 'Call in progress';
    el('stop').disabled = false;
    renderSpeechGate(data.speech_gate);
    setConnection('Streaming', true);
    setMessage(`Microphone audio is streaming to ${data.model || 'the V2 detector'}.`);
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
    setMessage(data.message || 'Streaming analysis error.', true);
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
    ws.onmessage = event => { handleStreamMessage(event).catch(error => setMessage(error.message, true)); };
    ws.onerror = () => setMessage('WebSocket connection error. Check the backend terminal.', true);
    ws.onclose = async event => {
      if (!stopping && el('call-state').textContent === 'Call in progress' && event.code !== 1000) {
        setMessage('Streaming connection closed unexpectedly.', true);
        await cleanupAudio();
        el('start').disabled = false;
        el('stop').disabled = true;
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
