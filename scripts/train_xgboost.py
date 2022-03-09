# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.models.xgboost_classifier import xgboost_train_and_evaluate_n_times


# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=1)


# features_args =['CamemBertEmbeddingFeature']

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)


# # =======================================


# features_args =['FastTextEmbeddingFeature', 'CamemBertEmbeddingFeature']
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =['FastTextEmbeddingFeature']
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)


# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)

# features_args =[
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]

# xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=5)


features_args =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]

xgboost_train_and_evaluate_n_times(features_args, max_depth=10, n_estimators=500, learning_rate=0.03, n=1)