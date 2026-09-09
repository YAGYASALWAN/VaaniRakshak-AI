"""Prepare all English MLAAD as durable features, then train the CNN offline."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from vaanirakshak.mlaad_source import DEFAULT_REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'prepare', 'train', 'all', 'status'))
    parser.add_argument('--data', type=Path, default=ROOT / 'data/mlaad_en')
    parser.add_argument('--revision', default=DEFAULT_REVISION)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--audio-root', type=Path, help='Optional already-downloaded root containing fake/en/')
    args = parser.parse_args()
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 64:
        parser.error('Use 1-100 epochs and batch size 1-64')
    if args.resume and args.command not in ('train', 'all'):
        parser.error('--resume applies to train/all only')
    from filelock import FileLock
    args.data.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.data / 'run.lock'), timeout=0):
        from vaanirakshak.mlaad_training import prepare, train, identity
        from vaanirakshak.mlaad_source import Network, plan
        from vaanirakshak.mlaad_cache import FeatureCache
        import json
        if args.command == 'status':
            path = args.data / 'plan.json'
            if not path.exists():
                print('No completed metadata plan yet. Rerun plan/prepare to reuse saved metadata.')
                return
            source = json.loads(path.read_text(encoding='utf-8'))
            cache = FeatureCache(args.data, identity(source))
            try:
                from collections import Counter
                print('Committed features:', dict(Counter(r['label'] for r in cache.rows())))
                print('Offline training ready:', (args.data / 'manifest.json').exists())
            finally:
                cache.close()
        if args.command == 'plan':
            from huggingface_hub import get_token
            source = plan(args.data, args.revision, Network(get_token()))
            print('Synthetic files:', len(source['rows']))
            print('Matching originals:', len(source['originals']))
            print('Synthetic transfer GB:', round(source['synthetic_bytes']/1e9, 3))
        if args.command in ('prepare', 'all'):
            prepare(args.data, args.revision, args.device, args.audio_root)
        if args.command in ('train', 'all'):
            train(args.data, args.epochs, args.batch_size, args.resume, args.device)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Committed features and completed-epoch checkpoints are preserved.')
        sys.exit(130)
