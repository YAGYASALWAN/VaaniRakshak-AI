"""Pinned MLAAD metadata and bounded audio reads without an on-disk WAV cache."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
from urllib.parse import urlsplit

from vaanirakshak.mlaad_cache import atomic_json, digest, original_key

REPO = 'mueller91/MLAAD'
DEFAULT_REVISION = '30c3dec763fa4803f11c2df1557d3b0828395366'
ARCHIVES = {lang: f'https://ics.tau-ceti.space/data/Training/stt_tts/{lang}.tgz'
            for lang in ('en_UK', 'en_US')}
MAX_AUDIO = 32 * 1024**2


class Network:
    def __init__(self, token=None):
        import requests
        self.requests = requests
        self.session = requests.Session()
        self.token = token

    def open(self, url):
        if urlsplit(url).scheme != 'https':
            raise ValueError('HTTPS required')
        headers = {'Accept-Encoding': 'identity'}
        if self.token and urlsplit(url).hostname == 'huggingface.co':
            headers['Authorization'] = 'Bearer ' + self.token
        # requests strips Authorization on cross-host redirects. Never put a
        # token on the public M-AILABS request or persist signed redirect URLs.
        for attempt in range(4):
            try:
                response = self.session.get(url, headers=headers, stream=True, timeout=(20, 90))
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    response.close()
                    raise self.requests.ConnectionError('Transient source error')
                if response.status_code != 200:
                    status = response.status_code
                    response.close()
                    raise ValueError(f'Source returned HTTP {status}; check account access/source availability')
                if urlsplit(response.url).scheme != 'https':
                    response.close()
                    raise ValueError('Refusing insecure redirect')
                return response
            except self.requests.RequestException:
                if attempt == 3:
                    raise ConnectionError('Source connection failed after four attempts; saved features remain intact') from None
                time.sleep(2**attempt)

    def read(self, url, expected=None, maximum=MAX_AUDIO):
        if expected is not None and not 0 < expected <= maximum:
            raise ValueError('Source file exceeds bounded memory allowance')
        for attempt in range(4):
            try:
                with self.open(url) as response:
                    size = response.headers.get('Content-Length')
                    if size and int(size) > maximum:
                        raise ValueError('Source file exceeds bounded memory allowance')
                    result = bytearray()
                    for block in response.iter_content(256 * 1024):
                        result.extend(block)
                        if len(result) > maximum:
                            raise ValueError('Response exceeds bounded memory allowance')
                    if expected is not None and len(result) != expected:
                        raise self.requests.ConnectionError('Truncated file')
                    return bytes(result)
            except self.requests.RequestException:
                if attempt == 3:
                    raise ConnectionError('Audio read interrupted repeatedly; rerun to reuse committed recordings') from None
                time.sleep(2**attempt)


def resolve_url(revision, path):
    from huggingface_hub import hf_hub_url
    return hf_hub_url(REPO, path, repo_type='dataset', revision=revision)


def metadata_rows(raw, wav_paths):
    text = raw.decode('utf-8-sig')
    header = text.splitlines()[0] if text.splitlines() else ''
    delimiter = next((d for d in ('|', ';', ',') if 'original_file' in header.split(d)), None)
    if delimiter is None:
        raise ValueError('Unrecognized MLAAD metadata header')
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    required = {'path', 'original_file', 'language', 'model_name', 'transcript'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('MLAAD metadata lacks required provenance')
    rows = []
    for row in reader:
        if row['language'].strip().lower() not in ('en', 'en_us', 'en_uk', 'english'):
            raise ValueError('Non-English language in English metadata')
        p = row['path'].replace('\\', '/').lstrip('./')
        if not p.startswith('fake/en/') or p not in wav_paths or '..' in PurePosixPath(p).parts:
            raise ValueError(f'Metadata path not in pinned English inventory: {p}')
        original = original_key(row['original_file'])
        transcript = ' '.join(row['transcript'].casefold().split())
        if not row['model_name'].strip() or not transcript:
            raise ValueError('Missing model or transcript')
        rows.append({'id': 'mlaad:' + p, 'path': p, 'label': 1, 'original_file': original,
                     'model_name': row['model_name'], 'transcript_key': digest(transcript.encode()),
                     'reference_speaker': row.get('reference_speaker', ''), 'language': 'en'})
    return rows


def plan(root, revision, network):
    from huggingface_hub import HfApi
    root = Path(root)
    target = root / 'plan.json'
    if target.exists():
        saved = json.loads(target.read_text(encoding='utf-8'))
        if saved['revision'] != revision or saved['repository'] != REPO:
            raise ValueError('Plan revision changed; use a separate cache')
        return saved
    api = HfApi(token=network.token)
    api.auth_check(REPO, repo_type='dataset')
    # Fully pinned inventory; no audio API that creates hidden audio caches.
    print('Listing pinned English source files...', flush=True)
    files = list(api.list_repo_tree(REPO, repo_type='dataset', revision=revision,
                                   path_in_repo='fake/en', recursive=True))
    wavs = {f.path: f for f in files if hasattr(f, 'size') and f.path.lower().endswith('.wav')}
    metas = sorted([f for f in files if hasattr(f, 'size') and f.path.endswith('/meta.csv')], key=lambda f: f.path)
    if not wavs or not metas:
        raise ValueError('No English audio/metadata in source')
    rows = []
    for number, f in enumerate(metas, 1):
        cache = root / 'metadata' / (digest(f.path.encode()) + '.json')
        identity = {'path': f.path, 'revision': revision, 'size': f.size}
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['identity'] != identity:
                raise ValueError('Metadata cache identity differs')
            group = saved['rows']
        else:
            raw = network.read(resolve_url(revision, f.path), expected=f.size, maximum=16 * 1024**2)
            group = metadata_rows(raw, wavs)
            atomic_json(cache, {'identity': identity, 'rows': group})
        rows.extend(group)
        print(f'Metadata {number}/{len(metas)} checked', flush=True)
    if len({r['path'] for r in rows}) != len(rows) or {r['path'] for r in rows} != set(wavs):
        raise ValueError('Metadata must describe every English WAV exactly once; refusing a partial inventory')
    for row in rows:
        f = wavs[row['path']]
        lfs = getattr(f, 'lfs', None)
        row['size'] = f.size
        row['sha256'] = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        row['blob_id'] = getattr(f, 'blob_id', None)
    saved = {'repository': REPO, 'revision': revision, 'rows': sorted(rows, key=lambda r: r['id']),
             'synthetic_bytes': sum(f.size for f in wavs.values()), 'generator_folders': len(metas),
             'originals': sorted({r['original_file'] for r in rows}), 'archives': ARCHIVES}
    atomic_json(target, saved)
    return saved


def verify_audio(raw, row):
    if len(raw) != row['size']:
        raise ValueError('Source WAV length mismatch')
    if row.get('sha256'):
        if digest(raw) != row['sha256']:
            raise ValueError('Source WAV SHA-256 mismatch')
    elif row.get('blob_id'):
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != row['blob_id']:
            raise ValueError('Source WAV Git blob hash mismatch')
    else:
        raise ValueError('Source WAV missing integrity identifier')


def prepare_real(root, source_plan, cache, network, transform):
    """Read original archives in memory; no tar extraction and no archive on disk.

An interrupted gzip stream must restart from the beginning. Already committed
features are skipped, but compressed bytes may be reread. Pin HTTP validators
before reading any audio and compare them on every subsequent attempt.
"""
    originals = source_plan['originals']
    for language, url in source_plan['archives'].items():
        wanted = {p for p in originals if p.startswith(language + '/')}
        if not wanted or all(cache.has('mailabs:' + p) for p in wanted):
            continue
        print(f'Streaming genuine {language}; interrupted archives may reread network bytes.', flush=True)
        stamp = Path(root) / (language + '_source.json')
        with network.open(url) as response:
            identity = {'url': url, 'length': response.headers.get('Content-Length'),
                        'etag': response.headers.get('ETag'), 'modified': response.headers.get('Last-Modified')}
            if not identity['etag'] and not (identity['length'] and identity['modified']):
                raise ValueError('Genuine source lacks stable HTTP validators; cannot safely resume it')
            if stamp.exists() and json.loads(stamp.read_text()) != identity:
                raise ValueError('Genuine archive changed since preparation began; do not mix versions')
            atomic_json(stamp, identity)
            remaining = {p for p in wanted if not cache.has('mailabs:' + p)}
            try:
                with tarfile.open(fileobj=response.raw, mode='r|gz') as archive:
                    for member in archive:
                        if not member.isfile() or not member.name.lower().endswith('.wav'):
                            continue
                        try:
                            key = original_key(member.name)
                        except ValueError:
                            continue
                        if key not in remaining:
                            continue
                        if not 0 < member.size <= MAX_AUDIO:
                            raise ValueError('Genuine WAV exceeds bounded audio size')
                        stream = archive.extractfile(member)
                        with stream:
                            raw = stream.read(MAX_AUDIO + 1)
                        if len(raw) != member.size:
                            raise ValueError('Truncated genuine WAV')
                        row = {'id': 'mailabs:' + key, 'original_file': key, 'label': 0,
                               'language': 'en', 'model_name': 'bonafide', 'reference_speaker': ''}
                        feature, properties = transform(raw)
                        cache.put(dict(row, **properties), feature)
                        remaining.remove(key)
                        if len(remaining) % 100 == 0:
                            print(f'{language}: {len(wanted)-len(remaining):,}/{len(wanted):,} genuine features saved', flush=True)
                        if not remaining:
                            break
            except (tarfile.TarError, OSError) as error:
                raise ConnectionError(f'{language} archive interrupted or invalid; rerun to reuse completed features') from error
        if remaining:
            examples = sorted(remaining)[:3]
            raise ValueError(f'Genuine archive missing {len(remaining)} referenced originals, e.g. {examples}; training blocked')
