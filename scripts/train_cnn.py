# -- fix path --
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# -- end fix path --

from src.train import train_and_evaluate_n_times

if __name__ == '__main__':
    # print('Training a model...')

    # features_args =['FastTextEmbeddingFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #         'WordLengthFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #         'WordLengthFeature', 
    #         'WordSyllableFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #         'WordLengthFeature', 
    #         'WordSyllableFeature', 
    #         'VowelCountFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #     'WordLengthFeature', 
    #     'WordSyllableFeature',
    #     'VowelCountFeature', 
    #     'TFIDFFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #     'WordLengthFeature',
    #     'WordSyllableFeature', 
    #     'VowelCountFeature', 
    #     'TFIDFFeature', 
    #     'WordRankFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 
    #     'WordLengthFeature', 
    #     'WordSyllableFeature', 
    #     'VowelCountFeature',
    #     'TFIDFFeature', 
    #     'WordRankFeature', 
    #     'LangGenFrequencyFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature',
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'ClearFrequencyFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=5)

#     features_args =[
#         # 'ContextFeature',
#         'WordLengthFeature',
#         'WordSyllableFeature',
#         'VowelCountFeature',
#         'TFIDFFeature',
#         'WordRankFeature',
#         'LangGenFrequencyFeature',
#         'ClearFrequencyFeature'
#     ]
#     train_and_evaluate_n_times(features_args, n=5)


#     #================================================================
#     features_args =['CamemBertEmbeddingFeature']
#     train_and_evaluate_n_times(features_args, n=5)

#     features_args =['CamemBertEmbeddingFeature', 
#             'WordLengthFeature']
#     train_and_evaluate_n_times(features_args, n=5)

#     features_args =['CamemBertEmbeddingFeature', 
#             'WordLengthFeature', 
#             'WordSyllableFeature']
#     train_and_evaluate_n_times(features_args, n=5)

#     features_args =['CamemBertEmbeddingFeature', 
#             'WordLengthFeature', 
#             'WordSyllableFeature', 
#             'VowelCountFeature']
#     train_and_evaluate_n_times(features_args, n=5)

#     features_args =['CamemBertEmbeddingFeature', 
#         'WordLengthFeature', 
#         'WordSyllableFeature',
#         'VowelCountFeature', 
#         'TFIDFFeature']
#     train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature', 
    #     'WordLengthFeature',
    #     'WordSyllableFeature', 
    #     'VowelCountFeature', 
    #     'TFIDFFeature', 
    #     'WordRankFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature', 
    #     'WordLengthFeature', 
    #     'WordSyllableFeature', 
    #     'VowelCountFeature',
    #     'TFIDFFeature', 
    #     'WordRankFeature', 
    #     'LangGenFrequencyFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature',
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'ClearFrequencyFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature',
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'ClearFrequencyFeature',
    #     'GeneralFrequencyFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =[
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'ClearFrequencyFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=5)



    # features_args =['CamemBertEmbeddingFeature', 
    #     'WordLengthFeature',
    #     'WordSyllableFeature', 
    #     'VowelCountFeature', 
    #     'TFIDFFeature', 
    #     'WordRankFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature', 
    #     'WordLengthFeature', 
    #     'WordSyllableFeature', 
    #     'VowelCountFeature',
    #     'TFIDFFeature', 
    #     'WordRankFeature', 
    #     'LangGenFrequencyFeature']
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['CamemBertEmbeddingFeature',
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'StopWordFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=4)

    # features_args =['FastTextEmbeddingFeature',
    #     'WordLengthFeature',
    #     'WordSyllableFeature',
    #     'VowelCountFeature',
    #     'TFIDFFeature',
    #     'WordRankFeature',
    #     'LangGenFrequencyFeature',
    #     'StopWordFeature'
    # ]
    # train_and_evaluate_n_times(features_args, n=5)

    # features_args =['FastTextEmbeddingFeature', 'CamemBertEmbeddingFeature']
    # train_and_evaluate_n_times(features_args, n=1)


    features_args =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]
    train_and_evaluate_n_times(features_args, n=1) 

    