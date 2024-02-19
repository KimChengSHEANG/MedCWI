import numpy as np
import os

from sklearn.metrics import accuracy_score, classification_report, f1_score, mean_absolute_error, precision_score, recall_score
from src.paths import DATASET_FILEPATH, DUMP_DIR
from src.utils import *
import pandas as pd
from tqdm import tqdm
from pathlib import Path 
from src.paths import REPO_DIR

from contextlib import contextmanager
from functools import lru_cache, wraps
import hashlib
import time
import sys
from bs4 import UnicodeDammit

def generate_vocab_fr(output_filepath):

    data = pd.read_csv(DATASET_FILEPATH)
    vocabulary = []
    for sentence in tqdm(data['sentence'], desc='Generating vocab'):
        words = process_line(sentence)
        vocabulary += [word.strip() for word in words if word not in vocabulary]

    vocabulary = list(set(vocabulary))
    vocabulary.sort()
    print("Write vocab to file...")
    with open(output_filepath, 'w') as write_file:
        for word in vocabulary:
            write_file.write("%s\n" % word)

def load_vocabs(lang):
    filepath = DUMP_DIR / f'{lang}_vocab.txt'
    if not filepath.exists():
        generate_vocab_fr(filepath)

    vocabs = read_lines(filepath)
    return vocabs

def zero_pad(sequence, max_len=600):
    # sequence length is approx. equal to
    # the max length of sequence in train set
    return np.pad(sequence, (0, max_len - len(sequence)), mode='constant') \
        if len(sequence) < max_len else np.array(sequence[:max_len])

def normalize(x, min, max):
    return (x - min) / (max - min)


def write_configure_to_file(args, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fout = open(out_dir + "/configs.txt", "w")
    for arg in vars(args):
        fout.write("{} = {} \n".format(arg.upper(), getattr(args, arg)))
    fout.close()

def batch_iter(data, batch_size, n_epochs, shuffle=False):
    print("Generating batch iterator ...")
    data = np.array(data, dtype=object)
    data_size = len(data)
    n_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(n_epochs):
        # Shuffle the resources at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(n_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def print_execution_time(func):
    @wraps(func)  # preserve name and doc of the function
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time({func.__name__}):{time.time() - start}")
        return result

    return wrapper



def predict_encoding(file_path) -> str:
    with open(file_path, 'rb') as file:
        content = file.read()
        suggestion = UnicodeDammit(content)
        return suggestion.original_encoding

def generate_hash(data):
    h = hashlib.new('md5')
    h.update(str(data).encode())
    return h.hexdigest()


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as fout:
        for line in lines:
            fout.write(line + '\n')


def read_lines(filepath, encoding=None):
    return [line.rstrip() for line in yield_lines(filepath, encoding)]


def yield_lines(filepath, encoding=None):
    filepath = Path(filepath)
    #print('encoding: ', predict_encoding(filepath))
    #print(filepath)
    if encoding is None:
        encoding = predict_encoding(filepath)
    with filepath.open('r', encoding=encoding) as f:
        for line in f:
            yield line.rstrip()

@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''

    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def evaluation_report(predictions, y_test, x_test_sents, output_dir, features):

    sents = []
    targets = []
    for sent in x_test_sents:
        sents.append(' '.join(sent[0] + sent[1] + sent[2]))
        targets.append(' '.join(sent[1]))
    data = pd.DataFrame()
    data['sentence'] = sents 
    data['target'] = targets
    data['label'] = y_test 
    data['predicted'] = predictions
    data.to_excel(output_dir / 'results.xlsx')


    print("all_prediction.shape:", predictions.shape)
    print("y_test.shape:", y_test.shape)

    print('prediction', predictions)
    print('y_test', y_test)

    print(classification_report(y_test, predictions, digits=4))
    # from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("f1 score: {:.4f}".format(f1_score(y_test, predictions)))
    print("Precision: {:.4f}".format(precision_score(y_test, predictions)))
    print("Recall: {:.4f}".format(recall_score(y_test, predictions)))
    print("Mean absolute error: {:.4f}".format(mean_absolute_error(y_test, predictions)))


    # save evaluation results to a file
    # output_dir = REPO_DIR / f"reports/{lang}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # dirname = str(checkpoint_dir).split('/')[-1]
    fout = open(output_dir / f"report.txt", "w")
    fout.write(classification_report(y_test, predictions, digits=4) + "\n")

    fout.write("Accuracy: {:.3f}\n".format(accuracy_score(y_test, predictions)))
    fout.write("f1 score: {:.3f}\n".format(f1_score(y_test, predictions)))
    fout.write("Precision: {:.3f}\n".format(precision_score(y_test, predictions)))
    fout.write("Recall: {:.3f}\n".format(recall_score(y_test, predictions)))
    fout.write("Mean absolute error: {:.3f}\n".format(mean_absolute_error(y_test, predictions)))

    fout.write("=" * 50 + "\n")
    fout.write("Features: \n")
    # features = read_lines(output_dir / 'features.txt')
    for feature in features: 
        fout.write(f"\t {feature} \n")

    fout.write("=" * 50 + "\n")
    # fout.write(str(checkpoint_dir) + "\n")
    # fout.write("=" * 50 + "\n")
    fout.close()

    features_filepath = Path(output_dir) / 'features.txt'
    if not features_filepath.exists():
        write_lines(features, features_filepath)

    print("Save report to a file. Completed!")
    return f1_score(y_test, predictions, average='macro')


def count_training(features, model_type):
    out_dir = REPO_DIR / f'models/{model_type}'
    filepaths = out_dir.glob('*/features.txt')
    train_count = 0
    for filepath in filepaths:
        lines = read_lines(filepath)
        if lines == features:
            train_count += 1
    return train_count
