from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.paths import REPO_DIR
from src.models.catboost_classifier import train_and_evaluate
import optuna

def run_tuning(params):
    features =['CamemBertEmbeddingFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        'LangGenFrequencyFeature',
        'ClearFrequencyFeature'
    ]
    return train_and_evaluate(features, iterations=params['iterations'], learning_rate=params['learning_rate'], use_gpu=True, n_seed=1, save_model=False)


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-6, 5e-3),
        'iterations': trial.suggest_categorical('iterations', [300, 500, 800, 1000, 1200, 1300, 1500]),
    }
    return run_tuning(params)

if __name__ == '__main__':
    tuning_log_dir = REPO_DIR / f'models'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name='CatBoost_study', direction="maximize",
                                storage=f'sqlite:///{tuning_log_dir}/study.db.db', load_if_exists=True)
    study.optimize(objective, n_trials=10)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))