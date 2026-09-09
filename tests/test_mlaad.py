import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from vaanirakshak.mlaad_cache import FeatureCache, SHAPE, original_key, partition, check_partitions
from vaanirakshak.mlaad_source import metadata_rows, prepare_real, verify_audio


def row(i, label=0, original=None, window=None):
    return {'id': f'{label}:{i}', 'label': label,
            'original_file': original or f'en_US/by_book/male/speaker/book/wavs/{i}.wav',
            'window_sha256': window or f'{label}:wave:{i}'}


class CacheTests(unittest.TestCase):
    def test_committed_record_survives_reopen_and_has_exact_features(self):
        with tempfile.TemporaryDirectory() as d:
            c = FeatureCache(d, {'revision': 'a'})
            c.put(row(1), np.full(SHAPE, .25))
            c.close()
            c = FeatureCache(d, {'revision': 'a'})
            self.assertTrue(c.has('0:1'))
            np.testing.assert_array_equal(c.get('0:1'), np.full(SHAPE, .25))
            c.close()

    def test_partial_sql_transaction_does_not_mark_record_complete(self):
        with tempfile.TemporaryDirectory() as d:
            c = FeatureCache(d, {})
            c.db.execute("INSERT INTO features VALUES ('partial','{}','wrong',X'00')")
            c.close()
            c = FeatureCache(d, {})
            self.assertFalse(c.has('partial'))
            c.close()

    def test_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            c = FeatureCache(d, {})
            c.put(row(1), np.zeros(SHAPE))
            c.db.execute("UPDATE features SET feature=X'00'")
            c.db.commit()
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                c.verify()
            c.close()

    def test_revision_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            FeatureCache(d, {'revision': 'a'}).close()
            with self.assertRaisesRegex(ValueError, 'changed'):
                FeatureCache(d, {'revision': 'b'})

    def test_nonfinite_not_committed(self):
        with tempfile.TemporaryDirectory() as d:
            c = FeatureCache(d, {})
            with self.assertRaises(ValueError):
                c.put(row(1), np.full(SHAPE, np.nan))
            self.assertFalse(c.has('0:1'))
            c.close()


class SplitTests(unittest.TestCase):
    def test_original_and_synthetic_variants_never_cross_and_order_stable(self):
        rows = [row(i, label) for i in range(100) for label in (0, 1)]
        a, audit = partition(rows, minimum=1)
        b, _ = partition(list(reversed(rows)), minimum=1)
        self.assertEqual(a, b)
        for i in range(100):
            self.assertEqual(len({r['split'] for r in a if r['original_file'] == rows[2*i]['original_file']}), 1)
        self.assertEqual(sum(sum(c.values()) for c in audit['counts'].values()), 200)

    def test_shared_transcript_connects_originals(self):
        rows = [row(i, label) for i in range(100) for label in (0, 1)]
        rows[0]['transcript_key'] = rows[21]['transcript_key'] = 'same-text'
        result, _ = partition(rows, minimum=1)
        linked = [r for r in result if r['original_file'] in (rows[0]['original_file'], rows[21]['original_file'])]
        self.assertEqual(len({r['split'] for r in linked}), 1)

    def test_duplicate_window_removed_and_conflicting_label_blocks(self):
        rows = [row(i, label) for i in range(100) for label in (0, 1)]
        rows.append(row('copy', 1, window=rows[1]['window_sha256']))
        _, audit = partition(rows, minimum=1)
        self.assertEqual(audit['duplicate_windows_removed'], {'1': 1})
        rows.append(row('bad', 0, window=rows[1]['window_sha256']))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            partition(rows, minimum=1)

    def test_one_class_cannot_train(self):
        with self.assertRaisesRegex(ValueError, 'lacks'):
            partition([row(i) for i in range(100)], minimum=1)

    def test_manual_split_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            check_partitions({'train': [row(1)], 'test': [row(1, 1)]})


class SourceTests(unittest.TestCase):
    def test_metadata_delimiters_and_original_normalization(self):
        for delimiter in ('|', ';', ','):
            data = delimiter.join(['path','original_file','language','model_name','transcript']) + '\n'
            data += delimiter.join(['./fake/en/model/a.wav','real/en_US/by_book/s/wavs/a.wav','en','generator','hello'])
            rows = metadata_rows(data.encode(), {'fake/en/model/a.wav'})
            self.assertEqual(rows[0]['original_file'], 'en_US/by_book/s/wavs/a.wav')
            self.assertEqual(rows[0]['label'], 1)

    def test_nonenglish_or_unmapped_paths_rejected(self):
        data = b'path|original_file|language|model_name|transcript\nfake/en/m/a.wav|en_US/a.wav|de|m|hi'
        with self.assertRaises(ValueError):
            metadata_rows(data, {'fake/en/m/a.wav'})
        with self.assertRaises(ValueError):
            original_key('en_US/../../a.wav')
        self.assertNotEqual(original_key('en_US/one/a.wav'), original_key('en_US/two/a.wav'))

    def test_audio_hash_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            verify_audio(b'abc', {'size': 3, 'sha256': 'wrong'})

    def test_real_stream_skips_committed_and_never_extracts_audio(self):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode='w:gz') as tar:
            for name, content in [('en_US/a.wav', b'one'), ('en_US/b.wav', b'two')]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        class Response:
            headers = {'ETag': 'version1', 'Content-Length': str(len(archive.getvalue()))}
            def __enter__(self):
                self.raw = io.BytesIO(archive.getvalue())
                return self
            def __exit__(self, *args):
                self.raw.close()
        class Net:
            def open(self, url):
                return Response()
        called = []
        def transform(raw):
            called.append(raw)
            return np.zeros(SHAPE), {'window_sha256': raw.decode()}
        with tempfile.TemporaryDirectory() as d:
            c = FeatureCache(d, {})
            c.put({'id': 'mailabs:en_US/a.wav'}, np.ones(SHAPE))
            source = {'originals': ['en_US/a.wav', 'en_US/b.wav'], 'archives': {'en_US': 'https://example.org/en_US.tgz'}}
            prepare_real(d, source, c, Net(), transform)
            self.assertEqual(called, [b'two'])
            self.assertFalse(list(Path(d).rglob('*.wav')))
            self.assertFalse(list(Path(d).rglob('*.tgz')))
            c.close()

    def test_changed_genuine_archive_blocks_before_audio(self):
        class Response:
            headers = {'ETag': 'changed'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Net:
            def open(self, url): return Response()
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'en_US_source.json').write_text('{}')
            c = FeatureCache(d, {})
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare_real(d, {'originals': ['en_US/a.wav'], 'archives': {'en_US': 'https://example.org/a'}}, c, Net(), None)
            c.close()


if __name__ == '__main__':
    unittest.main()
