"""Cross-platform orchestration script to train all RSSA models sequentially."""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
RECOMMENDER_DIR = SCRIPT_DIR.parent
STORAGE_DATA_DIR = (RECOMMENDER_DIR.parent / 'rssa-storage' / 'data' / 'seed_data').resolve()

RATINGS_FILE = STORAGE_DATA_DIR / 'sliced_movielens_ratings.csv'
EMOTIONS_FILE = STORAGE_DATA_DIR / 'sliced_ieRS_emotions_g20.csv'


def run_training(name: str, cmd: list[str]):
    """Train a recommender model."""
    print(f'\n{"=" * 60}\nTraining {name}...\n{"=" * 60}')
    try:
        subprocess.run(cmd, cwd=RECOMMENDER_DIR, check=True)
        print(f'{name} trained successfully.')
    except subprocess.CalledProcessError as e:
        sys.exit(f'\n❌ Training failed for {name}. Exit code: {e.returncode}')


def main():
    """Build model script to train recommener models for the RSSA dev environment."""
    if not RATINGS_FILE.exists():
        sys.exit(f'❌ Missing ratings file: {RATINGS_FILE}\nRun the storage extraction pipeline first.')

    base_cmd = ['uv', 'run', 'python', 'scripts/train_mfs.py', '-d', str(RATINGS_FILE)]

    # ieRS Model
    cmd_iers = base_cmd + [
        '-o',
        'assets/models/implicit_als_ers_ml32m/',
        '-a',
        'implicit',
        '--item_popularity',
        '--ave_item_score',
        '--cluster_index',
        '--emotion_index',
        str(EMOTIONS_FILE),
        '--filter_list',
        str(EMOTIONS_FILE),
        '--resample_count',
        '5',
    ]

    # Alt Algo Model
    cmd_alt = base_cmd + [
        '-o',
        'assets/models/implicit_als_ml32m/',
        '-a',
        'implicit',
        '--item_popularity',
        '--ave_item_score',
        '--cluster_index',
        '--resample_count',
        '5',
    ]

    # Preference Viz Model
    cmd_pref = base_cmd + [
        '-o',
        'assets/models/biased_als_ml32m/',
        '-a',
        'biased',
        '--item_popularity',
        '--ave_item_score',
        '--cluster_index',
    ]

    run_training('Implicit ALS (ieRS)', cmd_iers)
    run_training('Implicit ALS (Alt Algo)', cmd_alt)
    run_training('Biased ALS (Pref Viz)', cmd_pref)

    print('\nAll models successfully trained and serialized to assets/models/!')


if __name__ == '__main__':
    main()
