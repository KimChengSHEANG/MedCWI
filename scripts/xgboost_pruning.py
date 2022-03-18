from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.paths import REPO_DIR
from src.models.xgboost_classifier import train_and_evaluate
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

    return train_and_evaluate(features, max_depth=params['max_depth'], n_estimators=params['n_estimators'], learning_rate=params['learning_rate'], use_gpu=False, n_seed=1, save_model=False)


def objective(trial: optuna.trial.Trial) -> float:
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 5e-2),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'n_estimators': trial.suggest_categorical('n_estimators', [10, 20, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]),
    }
    return run_tuning(params)

if __name__ == '__main__':
    tuning_log_dir = REPO_DIR / f'models'
    tuning_log_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name='XGBoost_study', direction="maximize",
                                storage=f'sqlite:///{tuning_log_dir}/study.db', load_if_exists=True)
    study.optimize(objective, n_trials=500)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))