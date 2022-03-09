
# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.preprocessor import Preprocessor
from src.train import train, train_and_evaluate_n_times
from src.evaluate import evaluate


features =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]
preprocessor = Preprocessor(features)
x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents = preprocessor.load_preprocessed_data(seed=42)
# model_dir = train(x_train, y_train, x_valid, y_valid, features)
evaluate(x_test, y_test, x_test_sents, model_dir=None, features=features)