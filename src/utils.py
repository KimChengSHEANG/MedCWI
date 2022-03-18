from functools import lru_cache
import spacy
import pickle
# import dill as pickle
from tqdm import tqdm
from string import punctuation
import multiprocessing as mp

import stopwordsiso as stopwords

stopwords = stopwords.stopwords("fr")

def dump(obj, filepath):
    pickle.dump(obj, open(filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def load_dump(filepath):
    return pickle.load(open(filepath, "rb"))

def round(val):
    return '%.2f' % val


def safe_division(a, b):
    return a / b if b else 0


def normalize(val, minVal, maxVal):
    return (val - minVal) / (maxVal - minVal)
    

def normalizeOneMinusOne(x):
    inMin = 0
    inMax = 1
    outMin = -1
    outMax = 1
    return (((x - inMin) * (outMax - outMin)) / (inMax - inMin)) + outMin
    # return x

@lru_cache(maxsize=1)
def get_spacy_model():
    model = 'fr_core_news_md'
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, force=True,
                       model_path=spacy.util.get_package_path(model))
    return spacy.load(model)  # python -m spacy download en_core_web_sm`

@lru_cache(maxsize=10 ** 6)
def spacy_process(text):
    return get_spacy_model()(str(text))

@lru_cache(maxsize=None)
def is_punctuation(word):
    return ''.join([char for char in word if char not in punctuation]) == ''

def remove_punctuation(tokens):
    tokens = [token for token in tokens if not is_punctuation(token)]
    tokens = [token.strip(punctuation) for token in tokens] # strip punc: !!hello#!! -> hello
    return tokens
    
def remove_stopword(tokens):
    return [token for token in tokens if token not in stopwords]

@lru_cache(maxsize=10 ** 6)
def process_line(text, remove_stopwords=False, remove_punctuation=False):
    doc = spacy_process(text)
    tokens = []
    for token in doc:
        # if not (remove_stopwords and token.is_stop):
        tokens.append(token.text.lower())
            # tokens.append(token.lemma_.lower())
    # tokens = [token for token in tokens if token.isalpha()]
    if remove_punctuation:
        tokens = remove_punctuation(tokens)
    if remove_stopwords:
        tokens = remove_stopword(tokens)

    return tokens

def multiprocess(func, iterable, size, desc='Processing'):
    # pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(1)
    return list(tqdm(pool.imap(func, iterable), total=size, desc=desc))


