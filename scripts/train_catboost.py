# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.models.catboost_classifier import catboost_train_and_evaluate_n_times




# features_args =['CamemBertEmbeddingFeature']

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

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
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =[
#     'CamemBertEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =[
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)


# # =======================================


# features_args =['FastTextEmbeddingFeature', 'CamemBertEmbeddingFeature']
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =['FastTextEmbeddingFeature']
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)


# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
# ]

# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)
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
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =[
#     'FastTextEmbeddingFeature',
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

# features_args =[
#     'WordLengthFeature',
#     'WordSyllableFeature',
#     'VowelCountFeature',
#     'TFIDFFeature',
#     'WordRankFeature',
#     'LangGenFrequencyFeature',
#     'ClearFrequencyFeature'
# ]
# catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=5)

features_args =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]

catboost_train_and_evaluate_n_times(features_args, iterations=1200, learning_rate=0.03, n=1)
