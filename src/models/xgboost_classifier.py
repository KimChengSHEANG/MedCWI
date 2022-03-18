from src.helper import count_training, evaluation_report, write_lines
from src.paths import REPO_DIR
from src.preprocessor import Preprocessor
import xgboost as xgb
from pathlib import Path
import time 
import pickle

xgb.set_config(verbosity=2)

def train_and_evaluate(features, max_depth=10, n_estimators=300, learning_rate=0.05, use_gpu=True, n_seed=1, save_model=True):
    print("Features: ", features)
    print('Preparing data...')
    timestamp = str(int(time.time()))
    out_dir = REPO_DIR / f'models/XGBoost/{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(features, Path(out_dir) / 'features.txt')

    preprocessor = Preprocessor(features)
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents = preprocessor.load_preprocessed_data2(seed=42+n_seed)
    
    print('Creating model...')
    if use_gpu: 
        print('Training model with GPU')
        gbm = xgb.XGBClassifier(max_depth=max_depth, 
                        n_estimators=n_estimators, 
                        learning_rate=learning_rate,
                        use_label_encoder=False,
                        objective='binary:logistic',
                        eval_metric='logloss',
                        tree_method='gpu_hist', 
                        predictor='gpu_predictor',
                        gpu_id=0
                        )
    else:
        print('Training model with CPU')
        gbm = xgb.XGBClassifier(max_depth=max_depth, 
                            n_estimators=n_estimators, 
                            learning_rate=learning_rate,
                            use_label_encoder=False,
                            objective='binary:logistic',
                            eval_metric='logloss'
                            )
    print('Training model...')
    gbm_model = gbm.fit(x_train, y_train, 
                        eval_set=[(x_valid, y_valid)],
                        verbose=1)

    if save_model:
        model_dump_filepath = out_dir / 'model.pk'
        pickle.dump(gbm_model, model_dump_filepath.open('wb'))

    predictions = gbm_model.predict(x_test)

    # print(classification_report(y_test, predictions, digits=4))
    return evaluation_report(predictions, y_test, x_test_sents, out_dir, features)

    
def xgboost_train_and_evaluate_n_times(features, max_depth=10, n_estimators=300, learning_rate=0.05, use_gpu=False, n=1):
    while True:
        nb_train = count_training(features, 'XGBoost')
        if nb_train >= n:
            print(f'You have trained {nb_train} time. If you want to train more, increase the value n in scripts/train_cnn.py  or delete old trained models.')
            break

        print(f'Training: {nb_train+1}/{n}')
        train_and_evaluate(features, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, use_gpu=use_gpu, n_seed=n-nb_train)
