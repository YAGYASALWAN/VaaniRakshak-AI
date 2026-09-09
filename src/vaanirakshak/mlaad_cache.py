"""Durable per-recording feature cache and source-file-disjoint experiment splits.

No torch imports: recovery and split invariants can be tested without a GPU.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import numpy as np

SCHEMA = 'mlaad-en-centre-logmel-v1'
SHAPE = (64, 401)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def original_key(value):
    """Retain complete publisher identity, never match by basename alone."""
    parts = PurePosixPath(str(value).replace('\\', '/')).parts
    if '..' in parts:
        raise ValueError('Parent traversal in source metadata')
    starts = [i for i, p in enumerate(parts) if p in ('en_US', 'en_UK')]
    if len(starts) != 1 or not str(value).lower().endswith('.wav'):
        raise ValueError(f'Expected English M-AILABS WAV identity: {value!r}')
    return '/'.join(parts[starts[0]:])


class FeatureCache:
    """SQLite transactions atomically commit feature + record together.

A single DB avoids 143k small files. FULL synchronous mode keeps committed
recordings after process failure; the in-flight recording may be fetched again.
"""
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'features.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS contract (value TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS features (id TEXT PRIMARY KEY, record TEXT NOT NULL, sha TEXT NOT NULL, feature BLOB NOT NULL)')
        contract = json.dumps({'schema': SCHEMA, 'identity': identity}, sort_keys=True)
        previous = self.db.execute('SELECT value FROM contract').fetchone()
        if previous and previous[0] != contract:
            self.db.close()
            raise ValueError('Cache source/configuration changed; preserve this cache and use another directory')
        if not previous:
            self.db.execute('INSERT INTO contract VALUES (?)', (contract,))
            self.db.commit()

    def close(self):
        self.db.close()

    def has(self, key):
        return self.db.execute('SELECT 1 FROM features WHERE id=?', (key,)).fetchone() is not None

    def put(self, row, feature):
        feature = np.asarray(feature, dtype='<f2')
        if feature.shape != SHAPE or not np.isfinite(feature).all():
            raise ValueError('Invalid feature shape or nonfinite values')
        blob = feature.tobytes()
        with self.db:
            self.db.execute('INSERT INTO features VALUES (?, ?, ?, ?)',
                            (row['id'], json.dumps(row, sort_keys=True), digest(blob), blob))

    def rows(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT record FROM features ORDER BY id')]

    def get(self, key):
        found = self.db.execute('SELECT sha, feature FROM features WHERE id=?', (key,)).fetchone()
        if found is None or digest(found[1]) != found[0]:
            raise ValueError(f'Missing/corrupt feature: {key}')
        feature = np.frombuffer(found[1], dtype='<f2').reshape(SHAPE)
        if not np.isfinite(feature).all():
            raise ValueError(f'Nonfinite feature: {key}')
        return feature

    def verify(self):
        for (key,) in self.db.execute('SELECT id FROM features'):
            self.get(key)


def partition(rows, seed=42, minimum=100):
    """Keep original files, identical decoded windows, and transcripts together.

This is NOT a speaker-/generator-disjoint benchmark. Reference voice identity
is retained for audit, not inferred from filenames or conflated with originals.
"""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            next_key = parent[x]
            parent[x] = root
            x = next_key
        return root
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    seen_ids, window_owner, text_owner = set(), {}, {}
    for r in rows:
        if r['id'] in seen_ids or r['label'] not in (0, 1):
            raise ValueError('Duplicate identity or invalid class')
        seen_ids.add(r['id'])
        original = original_key(r['original_file'])
        find(original)
        for key, owners in ((r['window_sha256'], window_owner), (r.get('transcript_key'), text_owner)):
            if key:
                if key in owners:
                    union(original, owners[key])
                owners[key] = original
    output, unique, removed = [], {}, Counter()
    # Identical decoded windows with conflicting labels are not valid examples.
    for r in sorted(rows, key=lambda r: r['id']):
        key = r['window_sha256']
        if key in unique:
            if unique[key] != r['label']:
                raise ValueError('Identical audio window has conflicting genuine/synthetic labels')
            removed[str(r['label'])] += 1
            continue
        unique[key] = r['label']
        group = find(original_key(r['original_file']))
        bucket = int(digest(f'{seed}:{group}'.encode())[:8], 16) % 100
        split = 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
        output.append(dict(r, split=split, group=group))
    counts = {s: dict(Counter(r['label'] for r in output if r['split'] == s)) for s in ('train', 'validation', 'test')}
    for split, count in counts.items():
        if min(count.get(0, 0), count.get(1, 0)) < minimum:
            raise ValueError(f'{split} lacks {minimum} examples of each class after grouping: {count}. No random-row fallback.')
    check_partitions({s: [r for r in output if r['split'] == s] for s in counts})
    return output, {'counts': counts, 'duplicate_windows_removed': dict(removed), 'seed': seed,
                    'split_policy': '80/10/10 hash of connected original-file/transcript/window groups',
                    'speaker_independence': 'not established', 'generator_independence': 'not established'}


def check_partitions(parts):
    for field in ('id', 'original_file', 'group', 'window_sha256', 'transcript_key'):
        owners = {}
        for split, rows in parts.items():
            for r in rows:
                value = r.get(field)
                if not value:
                    continue
                if value in owners and owners[value] != split:
                    raise ValueError(f'{field} overlaps splits')
                owners[value] = split
