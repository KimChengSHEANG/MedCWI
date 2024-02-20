from catboost import CatBoostClassifier
from src.helper import count_training, save_evaluation_report, write_lines
from src.paths import REPO_DIR
from src.preprocessor import Preprocessor
import pickle
from pathlib import Path
import time
from catboost.utils import get_gpu_device_count

def train_and_evaluate(features, iterations=1200, learning_rate=0.03, use_gpu=True, n_seed=1, save_model=True):
    print("Features: ", features)
    print('Preparing data...')
    timestamp = str(int(time.time()))
    out_dir = REPO_DIR / f'models/CatBoost/{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    write_lines(features, Path(out_dir) / 'features.txt')

    preprocessor = Preprocessor(features)
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents = preprocessor.load_preprocessed_data2(seed=42+n_seed)
    
    print('Creating model...')
    # create a model   
    device = 'CPU'
    if use_gpu and get_gpu_device_count() > 0: 
        device = 'GPU'
        print("Training using GPU")
    else:
        print("Training using CPU")

    model = CatBoostClassifier(iterations=iterations, 
                            learning_rate=learning_rate,
                            task_type=device
                            )
    # train the model
    print('Training model...')
    model.fit(x_train, y_train,
        eval_set=(x_valid, y_valid),
        verbose=1
    )

    if save_model:
        model_dump_filepath = out_dir / 'model.pk'
        pickle.dump(model, model_dump_filepath.open('wb'))

    # evaluate the model with test data
    predictions = model.predict(x_test)

    model_name = out_dir.parent.stem + '_' + out_dir.stem
    return save_evaluation_report(predictions, y_test, x_test_sents, out_dir, model_name, features)


def catboost_train_and_evaluate_n_times(features, iterations=1200, learning_rate=0.03, n=1, use_gpu=False):
    
    while True:
        nb_train = count_training(features, 'CatBoost')
        if nb_train >= n:
            print(f'You have trained {nb_train} time. If you want to train more, increase the value n in scripts/train_cnn.py  or delete old trained models.')
            break
        
        print(f'Training: {nb_train+1}/{n}')
        train_and_evaluate(features, iterations=iterations, learning_rate=learning_rate, use_gpu=use_gpu, n_seed=n-nb_train)

