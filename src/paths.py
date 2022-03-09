from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
DUMP_DIR = RESOURCES_DIR / 'dumps'
DUMP_DIR.mkdir(exist_ok=True, parents=True)

DATASET_FILEPATH = RESOURCES_DIR / 'dataset/data.csv'
PROCESSED_DATA_DIR = REPO_DIR / f'resources/processed_data'

EMBEDDINGS_FILEPATH = RESOURCES_DIR / 'others/cc.fr.300.vec'


