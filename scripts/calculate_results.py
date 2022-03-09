from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import re
import numpy as np
from src.paths import REPO_DIR
from src.helper import log_stdout, read_lines
import pandas as pd
import shutil

# CURRENT_DIR = Path(__file__).parent.resolve()

'''
              precision    recall  f1-score   support

           0     0.8944    0.8740    0.8841       921
           1     0.7917    0.8228    0.8070       536

    accuracy                         0.8552      1457
   macro avg     0.8431    0.8484    0.8455      1457
weighted avg     0.8567    0.8552    0.8557      1457

Accuracy: 0.855
f1 score: 0.807
Precision: 0.792
Recall: 0.823
Mean absolute error: 0.145
'''


def yield_lines(filepath):
    with filepath.open('r') as f:
        for line in f:
            yield line
def normalize_whitespace(text):
    return ' '.join(text.split())

def get_features(filepath):
    sep = "="*50 + "\n"
    lines = list(yield_lines(filepath))
    text = " ".join(lines)
    chunks = text.split(sep)
    return chunks[1]

def get_macro_avg(filepath):
    lines = list(yield_lines(filepath))
    # calculate avg
    for line in lines:
        res = re.search(r'macro avg', line)
        if res:
            line = normalize_whitespace(line.strip())
            # print(float(line.split()[4]))
            chunks = line.split()
            precision = float(chunks[2]) 
            recall = float(chunks[3]) 
            f1 = float(chunks[4]) 
            return precision, recall, f1

def extract_results(model_dir):
    filepaths = list(model_dir.glob('*/report.txt'))

    scores = {}
    for filepath in filepaths:
        hash = get_features(filepath)
        precision, recall, f1 = get_macro_avg(filepath)
        if hash in scores:
            scores[hash].append([precision, recall, f1])
        else:
            scores[hash] = [[precision, recall, f1]]
    keys = sorted(scores.keys())

    data = {'feature': [], 'score':[], 'P': [], 'R':[], 'F':[]}

    with log_stdout(model_dir / 'results.txt'):
        for key in keys:
            print(key)
            print("scores:\t", scores[key])
            avg = np.mean(scores[key], axis=0)
            print("AVG:\t", avg)
            print()
            data['feature'].append(key)
            data['score'].append(scores[key])
            data['P'].append(avg[0])
            data['R'].append(avg[1])
            data['F'].append(avg[2])

    data = pd.DataFrame(data)
    data.to_excel(model_dir / 'raw_results.xlsx', index=False)



# MODELS_DIR = REPO_DIR / f'models/FR'
# MODELS_DIR = REPO_DIR / f'models/FR.CamemBert'

# extract_results(REPO_DIR / f'models/publish')
# extract_results(REPO_DIR / f'models/CatBoost')
extract_results(REPO_DIR / f'models/XGBoost')






# clean up
# model_dir = REPO_DIR / f'models/XGBoost'
# # filepaths = list(model_dir.glob('*/features.txt'))
# filepaths = list(model_dir.glob('*'))
# for filepath in filepaths:
#     file = filepath / 'report.txt'
#     if not file.exists():
#         print(filepath)

# d = {}
# p = {}
# for filepath in filepaths:
#     lines = read_lines(filepath)
#     text = ' '.join(lines)
#     p[text] = filepath
#     if text in d: 
#         d[text] += 1
#     else:
#         d[text] = 1

#     if d[text] > 5:
#         print(filepath)
#         # shutil.rmtree(filepath.parent)
    
# for key in d:
#     print(d[key], p[key])