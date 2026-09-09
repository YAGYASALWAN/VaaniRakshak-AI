"""Stream once into log-mel features, then run the existing CNN offline."""
from collections import Counter
import io
import json
from pathlib import Path
import shutil

from vaanirakshak.mlaad_cache import FeatureCache, SHAPE, atomic_json, check_partitions, digest, partition
from vaanirakshak.mlaad_source import REPO, Network, plan, prepare_real, resolve_url, verify_audio

NOTICE = ('English MLAAD + matching genuine M-AILABS experiment; centre four seconds per recording. '
          'Original-file, identical-window and normalized-transcript groups kept together. '
          'Speaker and generator independence are not established. '
          'Scores are uncalibrated, not proof of authenticity. MLAAD is non-commercial research data.')


def feature_transform(device):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    from vaanirakshak.english_training import Frontend
    frontend = Frontend().to(device).eval()

    def transform(raw):
        with sf.SoundFile(io.BytesIO(raw)) as audio:
            rate, frames = audio.samplerate, len(audio)
            if audio.channels not in (1, 2) or not 8000 <= rate <= 96000 or not 0 < frames <= rate * 120:
                raise ValueError('Audio must be nonempty, <=120 seconds, mono/stereo, 8-96 kHz')
            start = max(0, (frames - rate * 4) // 2)
            audio.seek(start)
            wave = audio.read(rate * 4, dtype='float32', always_2d=True).mean(axis=1)
        if not np.isfinite(wave).all() or np.sqrt(np.mean(wave**2)) < 1e-5:
            raise ValueError('Silent/nonfinite recording; no automatic class-specific exclusion')
        wave = torch.from_numpy(wave)
        if rate != 16000:
            wave = torchaudio.functional.resample(wave, rate, 16000)
        wave = torch.nn.functional.pad(wave[:64000], (0, max(0, 64000-len(wave))))
        with torch.no_grad():
            features = frontend(wave.unsqueeze(0).to(device)).cpu().numpy()[0].astype('<f2')
        return features, {'audio_sha256': digest(raw), 'window_sha256': digest(wave.numpy().tobytes()),
                          'duration_seconds': frames/rate, 'window_start_seconds': start/rate,
                          'sample_rate': rate, 'padded': frames < rate * 4}
    return transform


def identity(source_plan):
    return {'repository': REPO, 'revision': source_plan['revision'],
            'plan_sha256': digest(json.dumps(source_plan, sort_keys=True).encode())}


def prepare(root, revision, device, audio_root=None):
    from huggingface_hub import get_token
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use the known working GPU Python')
    torch.set_num_threads(4)
    token = get_token()
    if not token:
        raise ValueError('No saved Hugging Face login. Sign in on this computer.')
    network = Network(token)
    source_plan = plan(root, revision, network)
    cache = FeatureCache(root, identity(source_plan))
    try:
        complete = Path(root) / 'manifest.json'
        if complete.exists():
            load_manifest(root, cache, source_plan)
            print('Complete feature cache verified; no audio reads needed.', flush=True)
            return
        total = len(source_plan['rows']) + len(source_plan['originals'])
        remaining = total - len(cache.rows())
        # SQLite storage overhead plus room for transactions. No full matrix in RAM.
        needed = remaining * SHAPE[0] * SHAPE[1] * 2 * 1.25 + 1_000_000_000
        if shutil.disk_usage(root).free < needed:
            raise ValueError(f'Need about {needed/1e9:.1f} GB additional free space for features')
        print(f"Synthetic recordings: {len(source_plan['rows']):,}; matching genuine originals: {len(source_plan['originals']):,}", flush=True)
        print(f"Synthetic audio transfer once: {source_plan['synthetic_bytes']/1e9:.2f} GB plus genuine archive streams. No WAVs are retained.", flush=True)
        transform = feature_transform(device)
        # Obtain the genuine side first so broken source links cannot strand a
        # complete fake-only 45 GB acquisition. Full metadata checks precede both.
        prepare_real(root, source_plan, cache, network, transform)
        for index, row in enumerate(source_plan['rows'], 1):
            if cache.has(row['id']):
                continue
            if shutil.disk_usage(root).free < 500_000_000:
                raise ValueError('Less than 500 MB free; stop with existing cache preserved')
            local = Path(audio_root) / row['path'] if audio_root else None
            if local and local.is_file():
                if local.stat().st_size != row['size']:
                    raise ValueError('Existing local WAV length differs from pinned source')
                raw = local.read_bytes()
            else:
                raw = network.read(resolve_url(revision, row['path']), expected=row['size'])
            verify_audio(raw, row)
            try:
                features, properties = transform(raw)
            except Exception as error:
                raise ValueError(f"Could not prepare {row['path']}; completed recordings preserved") from error
            cache.put(dict(row, **properties), features)
            if index % 25 == 0 or index == len(source_plan['rows']):
                print(f"Synthetic features: {index:,}/{len(source_plan['rows']):,} committed", flush=True)
        rows = cache.rows()
        expected = {r['id'] for r in source_plan['rows']} | {'mailabs:' + p for p in source_plan['originals']}
        if {r['id'] for r in rows} != expected:
            raise ValueError('Feature coverage differs from full source plan')
        cache.verify()
        rows, audit = partition(rows)
        audit.update({'notice': NOTICE, 'prepared_recordings': total,
                      'generator_counts': dict(Counter(r['model_name'] for r in rows if r['label'] == 1)),
                      'duration_hours_by_class': {str(k): sum(r['duration_seconds'] for r in rows if r['label'] == k)/3600 for k in (0, 1)},
                      'analysed_window': 'centre four seconds, shorter audio zero-padded',
                      'reference_speaker_policy': 'retained, not used to certify disjointness'})
        atomic_json(Path(root) / 'audit.json', audit)
        atomic_json(complete, {'identity': identity(source_plan), 'rows': rows, 'audit': audit})
        print('FEATURE PREPARATION COMPLETE. Training can now run offline.', flush=True)
    finally:
        cache.close()


def load_manifest(root, cache, source_plan):
    manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity(source_plan):
        raise ValueError('Manifest source identity mismatch')
    cache.verify()
    # Regenerate the frozen grouping from committed provenance to detect edits.
    expected, _ = partition(cache.rows())
    if manifest['rows'] != expected:
        raise ValueError('Manifest differs from verified feature provenance')
    return manifest


def train(root, epochs=25, batch_size=16, resume=None, device='cuda'):
    from vaanirakshak.english_training import run
    root = Path(root)
    source_plan = json.loads((root / 'plan.json').read_text(encoding='utf-8'))
    cache = FeatureCache(root, identity(source_plan))
    try:
        manifest = load_manifest(root, cache, source_plan)
        rows = manifest['rows']
        class Features:
            def __init__(self, group):
                self.group = group
            def __getitem__(self, index):
                return cache.get(self.group[index]['id'])
        def load(root_arg, split, device_arg):
            group = [r for r in rows if r['split'] == split]
            return Features(group), group
        counts = {s: dict(Counter(r['label'] for r in rows if r['split'] == s)) for s in ('train', 'validation', 'test')}
        return run(root, epochs, batch_size, resume, device,
                   profile={'repository': REPO + ' + M-AILABS', 'revision': source_plan['revision'],
                            'notice': NOTICE, 'counts': counts, 'cache_split': load,
                            'check_disjoint': check_partitions, 'run_prefix': 'mlaad_'})
    finally:
        cache.close()
