# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.paths import REPO_DIR
from src.train import train, train_and_evaluate_n_times
from src.evaluate import evaluate
from src.preprocessor import Preprocessor

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
    x, y, sents = preprocessor.preprocess_custom_data(testset_filepath)
    evaluate(x, y, sents, model_dir=checkpoint_dir, features=features_args, output_dir=REPO_DIR/'outputs')
    
    

    