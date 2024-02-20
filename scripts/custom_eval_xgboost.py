# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.paths import REPO_DIR
from src.preprocessor import Preprocessor
from src.helper import save_evaluation_report
import pickle

if __name__ == '__main__':

    testset_filepath = REPO_DIR / "resources/dataset/custom_testset.csv"
    checkpoint_dir = None # will load the latest checkpoint
    # checkpoint_dir = '1708369305' # for specific checkpoint
    

    
    features_args =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]

    
    preprocessor = Preprocessor(features_args)
    x, y, sents = preprocessor.preprocess_custom_data2(testset_filepath)
    
    p = Path(REPO_DIR / f'models/XGBoost')
    dirs = sorted(p.iterdir(), key=lambda f: f.stat().st_mtime)

    if len(dirs) > 0:
        if checkpoint_dir:
            checkpoint_dir = REPO_DIR / f'models/XGBoost/{checkpoint_dir}'
        else:
            checkpoint_dir = Path(str(dirs[-1])) # load the last checkpoint
            
        model_dump_filepath = checkpoint_dir / 'model.pk'
        model = pickle.load(model_dump_filepath.open('rb'))
    
        predictions = model.predict(x)

        output_dir = REPO_DIR/'outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        model_name = checkpoint_dir.parent.stem + '_' + checkpoint_dir.stem
        
        save_evaluation_report(predictions, y, sents, output_dir, model_name, features_args)
        
        

    
    

    