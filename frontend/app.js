'use strict';
const el = id => document.getElementById(id);
let selectedFile = null, previewUrl = null, ready = false, busy = false;
function message(text, error = false) { el('message').textContent = text; el('message').classList.toggle('error', error); }
function updateButton() { el('analyze').disabled = !selectedFile || !ready || busy; }
function clearResult() { el('result').hidden = true; }
function choose(file) {
  if (busy) return;
  clearResult();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; selectedFile = null;
  el('selected').hidden = true; el('player').removeAttribute('src');
  if (!file) { message('Choose a recording to begin.'); updateButton(); return; }
  if (!/\.(wav|flac)$/i.test(file.name) || file.size > 20 * 1024 * 1024 || file.size === 0) {
    message('Choose a nonempty WAV or FLAC file smaller than 20 MiB.', true); el('file').value = ''; updateButton(); return;
  }
  selectedFile = file; previewUrl = URL.createObjectURL(file);
  el('filename').textContent = file.name; el('player').src = previewUrl; el('selected').hidden = false;
  message(ready ? 'Ready to analyze.' : 'Audio selected. Waiting for a trained model.'); updateButton();
}
async function refresh() {
  try {
    const response = await fetch('/api/status'); const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not check model status.');
    ready = data.ready;
    el('model-status').textContent = ready ? 'Checkpoint available' : 'Waiting for a trained model';
    el('model-name').textContent = data.model || 'Connect the trained model adapter to enable analysis.';
    const metrics = data.evaluation?.test_metrics;
    el('metrics').textContent = metrics ? `Held-out experiment: F1 ${metrics.f1.toFixed(3)}, ROC-AUC ${metrics.roc_auc.toFixed(3)} across ${metrics.n.toLocaleString()} recordings.` : 'Not evaluated yet. No accuracy claim is available.';
    if (selectedFile && !busy) message(ready ? 'Ready to analyze.' : 'Audio selected. Waiting for a trained model.');
  } catch (error) { ready = false; el('model-status').textContent = 'Connection unavailable'; message(error.message, true); }
  updateButton();
}
el('file').addEventListener('change', event => choose(event.target.files[0]));
el('remove').addEventListener('click', () => { if (!busy) { el('file').value = ''; choose(null); } });
el('refresh').addEventListener('click', refresh);
for (const name of ['dragenter', 'dragover']) el('dropzone').addEventListener(name, event => { event.preventDefault(); el('dropzone').classList.add('drag'); });
for (const name of ['dragleave', 'drop']) el('dropzone').addEventListener(name, event => { event.preventDefault(); el('dropzone').classList.remove('drag'); });
el('dropzone').addEventListener('drop', event => { if (!busy) { el('file').value = ''; choose(event.dataTransfer.files[0]); } });
el('analyze').addEventListener('click', async () => {
  if (!selectedFile || busy || !ready) return;
  busy = true; updateButton(); clearResult(); el('file').disabled = true; el('remove').disabled = true;
  el('analyze').textContent = 'Analyzing…'; message('Processing your recording locally…');
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: selectedFile, signal: controller.signal });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || 'Analysis failed.');
    el('result-heading').textContent = result.verdict;
    el('result-copy').textContent = result.verdict === 'Inconclusive' ? 'The score is close to the decision threshold. This recording needs further review.' : result.verdict === 'Likely synthetic' ? 'The analyzed segment shows patterns the model associates with synthetic speech.' : 'The analyzed segment shows patterns the model associates with genuine speech.';
    el('score').textContent = result.synthetic_score.toFixed(3); el('threshold').textContent = result.threshold.toFixed(2);
    el('score-meter').value = result.synthetic_score; el('elapsed').textContent = `${result.elapsed_seconds.toFixed(1)} seconds`;
    el('scope').textContent = `Analyzed ${result.window_start_seconds.toFixed(1)}–${(result.window_start_seconds + result.window_seconds).toFixed(1)} seconds of a ${result.duration_seconds.toFixed(1)}-second recording. Model: ${result.model}.`;
    el('model-notice').textContent = result.notice;
    el('result').hidden = false; el('result-heading').focus(); message('Analysis complete.');
  } catch (error) { message(error.name === 'AbortError' ? 'Analysis timed out. Check the terminal before retrying.' : error.message, true); }
  finally { clearTimeout(timeout); busy = false; el('file').disabled = false; el('remove').disabled = false; el('analyze').textContent = 'Analyze recording'; updateButton(); }
});
refresh();
