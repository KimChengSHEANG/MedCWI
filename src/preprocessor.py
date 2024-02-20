from pathlib import Path
import sys

from numpy import size
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
from src.resources.download_data import download_fasttext
from src.configurations import Language
from src.paths import DATASET_FILEPATH, EMBEDDINGS_FILEPATH, PROCESSED_DATA_DIR, DUMP_DIR, RESOURCES_DIR
from src.utils import *
from src.helper import *
import pandas as pd
from tqdm import tqdm
from src import helper
from src import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import WordEmbeddings, TransformerWordEmbeddings
from transformers import FlaubertModel, FlaubertTokenizer
import flair, torch
from hyphen import Hyphenator
from wordfreq import word_frequency

flair.device = torch.device('cpu') 
MAX_TARGET_WORDS = 12

class AbstractFeature():
    use_cache = True

    def __init__(self, feature_extractor_context, feature_extractor_target, lang):
        self.feature_extractor_context = feature_extractor_context
        self.feature_extractor_target = feature_extractor_target
        self.lang = lang

        self.cache_filepath = DUMP_DIR / f'features_cache/{self.__class__.__name__}.pk'
        if self.cache_filepath.exists() and self.use_cache:
            self.cache = load_dump(self.cache_filepath)
        else:
            self.cache = {}

    def generate_cache_key(self, prefix, words):
        return f'{prefix}_' + helper.generate_hash(words)
    
    def save_cache_to_file(self):
        if not self.cache_filepath.exists() and self.use_cache:
            self.cache_filepath.parent.mkdir(parents=True, exist_ok=True)
            dump(self.cache, self.cache_filepath)

    def class_name(self):
        name = self.__class__.__name__.replace('Feature', '')
        # return ''.join(re.findall(r'[A-Z]+', name)) # acronym
        return name

    def process(self, left_context_words, target_words, right_context_words):
        self.left_context_words = left_context_words
        self.target_words = target_words
        self.right_context_words = right_context_words
    
    def get_left_context(self):
        key = self.generate_cache_key('LC', self.left_context_words)
        if key in self.cache:
            return self.cache[key]
        else:
            val = self.feature_extractor_context(self.left_context_words)
            self.cache[key] = val
            return val

    def get_right_context(self):
        key = self.generate_cache_key('RC', self.right_context_words)
        if key in self.cache:
            return self.cache[key]
        else:
            val = self.feature_extractor_context(self.right_context_words)
            self.cache[key] = val
            return val

    def get_target(self):
        key = self.generate_cache_key('T', self.target_words)
        if key in self.cache:
            return self.cache[key]
        else:
            val = self.feature_extractor_target(self.target_words)
            self.cache[key] = val
            return val

class AbstractEmbeddingFeature(AbstractFeature):
    def __init__(self, embedding, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)
        self.embedding = embedding

        # Get embedding size
        sentence = Sentence(['a'])
        self.embedding.embed(sentence)
        self.EMBEDDING_DIM = sentence[0].embedding.size()[0]
    
    def process(self, left_words, target_words, right_words):
        super().process(left_words, target_words, right_words)

        self.all_words = left_words + target_words + right_words
        
        self.sentence = Sentence(self.all_words)
        # self.embedding.embed(self.sentence)
        

    def extractor_context(self, words):
        size = len(words)
        if size > 0:
            self.embedding.embed(self.sentence)
            
            word_embeddings_matrix = np.zeros((size, self.EMBEDDING_DIM))
            for i, word in enumerate(words):
                index = self.all_words.index(word)
                word_embeddings_matrix[i] = self.sentence[index].embedding

            return word_embeddings_matrix.mean(axis=0)
        else:
            return [0] * self.EMBEDDING_DIM

    def extractor_target(self, words):
        self.embedding.embed(self.sentence)
        word_embeddings_matrix = np.zeros((MAX_TARGET_WORDS, self.EMBEDDING_DIM))
        for i, word in enumerate(words):
            index = self.all_words.index(word)
            word_embeddings_matrix[i] = self.sentence[index].embedding

        return word_embeddings_matrix

class FastTextEmbeddingFeature(AbstractEmbeddingFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(embedding=WordEmbeddings('fr'), *args, **kwargs)

class CamemBertEmbeddingFeature(AbstractEmbeddingFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(embedding=TransformerWordEmbeddings('camembert-base'), *args, **kwargs)

class FlauBertEmbeddingFeature(AbstractFeature):

    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)
        # Choose among ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased', 
    #               'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']

        modelname = 'flaubert/flaubert_base_cased' 

        # Load pretrained model and tokenizer
        self.flaubert, log = FlaubertModel.from_pretrained(modelname, output_loading_info=True)
        self.flaubert_tokenizer = FlaubertTokenizer.from_pretrained(modelname, do_lowercase=False)
        # do_lowercase=False if using cased models, True if using uncased ones

        # Get embedding size
        embeddings = self.extract_embeddings(['a'])
        self.EMBEDDING_DIM = len(embeddings[0])

        print("Embedding size: ", self.EMBEDDING_DIM)

    # @lru_cache(maxsize=2024)
    def extract_embeddings(self, words):
        tokens = self.flaubert_tokenizer.encode(words, add_special_tokens=False)
        token_ids = torch.tensor([tokens])
        last_layer = self.flaubert(token_ids)[0]
        embeddings = []
        for i in range(len(words)):
            embedding = last_layer[:, i, :][0].cpu().detach().numpy() # extract embedding for each word and convert fron tensor to numpy array
            embeddings.append(embedding)
        return embeddings

    def process(self, left_words, target_words, right_words):
        super().process(left_words, target_words, right_words)

        self.all_words = left_words + target_words + right_words
        
        # self.embeddings = self.extract_embeddings(self.all_words)

    def extractor_context(self, words):
        size = len(words)
        if size > 0:
            self.embeddings = self.extract_embeddings(self.all_words)

            word_embeddings_matrix = np.zeros((size, self.EMBEDDING_DIM))
            for i, word in enumerate(words):
                index = self.all_words.index(word)
                word_embeddings_matrix[i] = self.embeddings[index]

            return word_embeddings_matrix.mean(axis=0)
        else:
            return [0] * self.EMBEDDING_DIM

    def extractor_target(self, words):
        self.embeddings = self.extract_embeddings(self.all_words)

        word_embeddings_matrix = np.zeros((MAX_TARGET_WORDS, self.EMBEDDING_DIM))
        for i, word in enumerate(words):
            index = self.all_words.index(word)
            word_embeddings_matrix[i] = self.embeddings[index]

        return word_embeddings_matrix

class WordLengthFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)
        self.minLength = 1
        self.maxLength = 2
        vocabs = load_vocabs(self.lang)
        for word in vocabs:
            word_length = len(word)
            if self.maxLength < word_length:
                self.maxLength = word_length

    def extractor_context(self, words):
        size = len(words)
        if size > 0:
            sum = 0.0
            for word in words:
                sum += self.normalize(len(word))
            return sum / size
        else:
            return 0.0

    def extractor_target(self, words):
        word_lengths = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            word_lengths[i] = self.normalize(len(word))

        return word_lengths

    def normalize(self, length):
        val = 0
        if length > 0:
            val = (length - self.minLength) / (self.maxLength - self.minLength)

        return normalizeOneMinusOne(val)

class WordSyllableFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

        self.minSyllable = 1
        self.maxSyllable = 2
        vocabs = load_vocabs(self.lang)
        for word in vocabs:
            count = self.count_syllable(word)
            if self.maxSyllable < count:
                self.maxSyllable = count

    @lru_cache(maxsize=1)
    def get_hypernator(self):
        return Hyphenator('fr_FR')

    @lru_cache(maxsize=10**6)
    def count_syllable(self, word):
        if len(word) >= 80: word = word[:80] # if the word is too long, crop it.
        
        h = self.get_hypernator()  
        return len(h.syllables(word))

    def normalize(self, nb_syllable):
        val = nb_syllable
        if nb_syllable > 0:
            val = (nb_syllable - self.minSyllable) / (self.maxSyllable - self.minSyllable)
        return normalize(val, self.minSyllable, self.maxSyllable)

    def extractor_context(self, words):
        size = len(words)
        if size > 0:
            count = 0
            for word in words:
                count += self.normalize(self.count_syllable(word))
            return count / size
        else:
            return 0.0

    def extractor_target(self, words):
        target_words_syllable = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            target_words_syllable[i] = self.normalize(self.count_syllable(word))
        return target_words_syllable

class WordRankFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

    @lru_cache(maxsize=1)
    def get_word2rank(self, vocab_size=np.inf):
        model_filepath = DUMP_DIR / f"{self.lang}_{EMBEDDINGS_FILEPATH.stem}.pk"
        if model_filepath.exists():
            return load_dump(model_filepath)
        else:
            print("\nPreprocessing word2rank...")
            download_fasttext()
            lines_generator = yield_lines(EMBEDDINGS_FILEPATH, 'utf-8')
            word2rank = {}
            # next(lines_generator)
            for i, line in enumerate(tqdm(lines_generator, total=vocab_size, desc="Reading fasttext: ")):
                if i >= vocab_size: 
                    break
                word = line.split(' ')[0]
                word2rank[word] = i
            dump(word2rank, model_filepath)
            return word2rank
    
    @lru_cache(maxsize=10000)
    def get_normalized_rank(self, word):
        vocab_size = len(self.get_word2rank())
        rank = self.get_word2rank().get(word, vocab_size)
        return np.log(1 + rank) / np.log(1 + vocab_size)

    def extractor_context(self, words):
        size = len(words)
        val = 0
        if size > 0:
            val = sum([self.get_normalized_rank(word) for word in words]) / size
        return val

    def extractor_target(self, words):
        word_ranks = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            word_ranks[i] = self.get_normalized_rank(word)
        return word_ranks

class VowelCountFeature(AbstractFeature):
    vowels_fr = ['a','á','â','æ','e','é','è','ê','ë','i','î','ï','o','ô','œ','u','ù','û','ü','A','Á','Â','Æ','E','É','È','Ê','Ë','I','Î','Ï','O','Ô','Œ','U','Ù','Û','Ü','ɛ']

    vowels = vowels_fr

    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

        self.max = 1
        vocabs = load_vocabs(self.lang)
        self.max = max([self.count_vowels(word) for word in vocabs])

    def normalize(self, nbvowels):
        val = nbvowels
        if nbvowels > 0:
            val = nbvowels / self.max
        return val

    def count_vowels(self, word):
        return len([1 for c in word.lower() if c in self.vowels])

    def extractor_context(self, words):
        size = len(words)
        if size > 0:
            return sum([self.normalize(self.count_vowels(word)) for word in words]) / size
        else:
            return 0

    def extractor_target(self, words):
        vector = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            vector[i] = self.normalize(self.count_vowels(word))
        return vector

class TFIDFFeature(AbstractFeature):
    # tfidf using sentence
    # using paragraph 

    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

        tfidf_dump_file = DUMP_DIR / f"{self.lang}_tfidf_model.pk"
        if not tfidf_dump_file.exists():
            data = pd.read_csv(DATASET_FILEPATH)
            docs = data['sentence'].tolist()

            tfidf = TfidfVectorizer(tokenizer=process_line)
            self.model = tfidf.fit(docs)
            utils.dump(self.model, tfidf_dump_file)
        else:
            self.model = utils.load_dump(tfidf_dump_file)

        self.feature_words = self.model.get_feature_names()

    def process(self, left_words, target_words, right_words):
        super().process(left_words, target_words, right_words)

        sentence = ' '.join(left_words + target_words + right_words)
        x = self.model.transform([sentence])
        self.X = np.array(x.todense())[0]

    def get_tfidf(self, word):
        if word in self.feature_words:
            index = self.feature_words.index(word)
            return self.X[index]
        else:
            return 0.0

    def extractor_context(self, words):
        size = len(words)
        val = 0
        if size > 0:
            total = 0.0
            for word in words:
                total += self.get_tfidf(word)
            val = total / size
        return val

    def extractor_target(self, words):
        vec = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            if i < MAX_TARGET_WORDS:
                vec[i] = self.get_tfidf(word)
        return vec

class LangGenFrequencyFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

        lang_gen_filepath = RESOURCES_DIR / 'others/lang-genFINAL.freq'
        data = pd.read_csv(lang_gen_filepath, delimiter='|', header=None, names=['word', 'freq'])
        values = set(data['freq'])
        maxVal = max(values)
        minVal = min(values)

        data['norm_freq'] = data['freq'].apply(lambda x: (x - minVal)/(maxVal - minVal))
        self.freq = dict(zip(data['word'], data['norm_freq']))

    def extractor_context(self, words):
        size = len(words)
        val = 0
        if size > 0:
            val = sum([self.freq[word] for word in words if word in self.freq]) / size
        return val

    def extractor_target(self, words):
        frequencies = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            freq = 0
            if i < MAX_TARGET_WORDS and word in self.freq:
                freq = self.freq[word]
                frequencies[i] = freq
        return frequencies

class ClearFrequencyFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

        lang_gen_filepath = RESOURCES_DIR / 'others/clearFINAL.freq'
        data = pd.read_csv(lang_gen_filepath, delimiter='|', header=None, names=['word', 'freq'])
        values = set(data['freq'])
        maxVal = max(values)
        minVal = min(values)

        data['norm_freq'] = data['freq'].apply(lambda x: (x - minVal)/(maxVal - minVal))
        self.freq = dict(zip(data['word'], data['norm_freq']))

    def extractor_context(self, words):
        size = len(words)
        val = 0
        if size > 0:
            val = sum([self.freq[word] for word in words if word in self.freq]) / size
        return val

    def extractor_target(self, words):
        frequencies = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            freq = 0
            if i < MAX_TARGET_WORDS and word in self.freq:
                freq = self.freq[word]
                frequencies[i] = freq
        return frequencies

class GeneralFrequencyFeature(AbstractFeature):

    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)

    def extractor_context(self, words):
        size = len(words)
        val = 0
        if size > 0:
            val = sum([word_frequency(word, 'fr') for word in words]) / size
        return val

    def extractor_target(self, words):
        frequencies = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            if i < MAX_TARGET_WORDS:
                freq = word_frequency(word, 'fr')
                frequencies[i] = freq
        return frequencies

class StopWordFeature(AbstractFeature):
    def __init__(self, *args, **kwargs):
        super().__init__(self.extractor_context, self.extractor_target, *args, **kwargs)
        self.nlp = get_spacy_model()

    def process(self, left_words, target_words, right_words):
        super().process(left_words, target_words, right_words)
        self.all_words = left_words + target_words + right_words
        doc = self.nlp(" ".join(self.all_words))

        self.stopword_values = []
        for token in doc:
            self.stopword_values.append(1.0 if token.is_stop else 0.0)


    def extractor_context(self, words):
        # size = len(words)
        # val = 0
        # if size > 0:
        #     values = []
        #     for word in words:
        #         index = self.all_words.index(word)
        #         values.append(self.stopword_values[index])
        #     val = sum(values) / size
        # return val
        return 0

    def extractor_target(self, words):
        values = [0] * MAX_TARGET_WORDS
        for i, word in enumerate(words):
            if i < MAX_TARGET_WORDS:
                index = self.all_words.index(word)
                values[i] = self.stopword_values[index]
        return values

class Preprocessor:
    def __init__(self, features_args=None, lang=Language.FRENCH):
        self.lang = lang
        self.features_args = features_args
        # self.features = self.get_features(self.features_args)
        # print(self.features)
        if features_args:
            self.hash = helper.generate_hash(features_args)
            self.num_feature = len(features_args)
        else:
            self.hash = "no_feature"
            self.num_feature = 0

        self.PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / self.lang / self.hash
        self.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_FILEPATH = self.PROCESSED_DATA_PATH / f'{DATASET_FILEPATH.stem}.pk'
            
    def get_num_features(self):
        return self.num_feature
    
    def get_hash(self):
        return self.hash
    
    def get_class(self, class_name, *args, **kwargs):
        return globals()[class_name](*args, **kwargs)

    def get_features(self, features_args):
        features = []
        for feature_name in features_args:
            features.append(self.get_class(class_name=feature_name, lang=self.lang))
        return features

    def left_context_vector(self):
        vec = []
        for feature in self.features:
            name = 'LC_' + feature.class_name()
            vec.append((name, feature.get_left_context()))
        return vec

    def right_context_vector(self):
        vec = []
        for feature in self.features:
            name = 'RC_' + feature.class_name()
            vec.append((name, feature.get_right_context()))
        return vec

    def target_word_vectors(self):
        matrix = [[] for _ in range(MAX_TARGET_WORDS)]
        for feature in self.features:
            name = 'T_' + feature.class_name()
            vec = feature.get_target()
            for i in range(MAX_TARGET_WORDS):
                matrix[i].append((f'{name}_{i}', vec[i]))
        return matrix


    def preprocess_custom_data(self, testset_filepath):
        print("Prepossessing with features: ")
        print("\t" + "\n\t".join(self.features_args))

        self.features = self.get_features(self.features_args)
        pd_data = pd.read_csv(testset_filepath)
        data = {}
        labels = []
        for i in tqdm(range(len(pd_data)), desc='Preprocessing'):
            x, y = self.process(pd_data.iloc[i])
            for key, val in x:
                if key in data:
                    data[key].append(val)
                else:
                    data[key] = [val]
            labels.append(y)

        data['label'] = labels
        
        data = pd.DataFrame(data) 
        data.to_excel(self.PROCESSED_DATA_PATH / 'test.xlsx', index=False)
        size = len(data)

        d = data.iloc[:, 3:-1]
        y = data.loc[:, 'label'].tolist()
        sents = data.iloc[:, :3].to_numpy()

        x = []
        for i in range(size):
            x_ = []
            n = len(d.iloc[i]) // (MAX_TARGET_WORDS + 2)
            
            for j in range(MAX_TARGET_WORDS + 2):
                values = d.iloc[i][j*n : j*n + n].tolist()
                row = np.array([])
                for val in values:
                    row = np.append(row, val)
                x_.append(row)

            x.append(np.array(x_))
            
        x = np.array(x)
        y = np.array([[0,1] if label==1 else [1,0] for label in y])

        return x, y, sents
    
    def preprocess_custom_data2(self, testset_filepath):
        x, y, sents = self.preprocess_custom_data(testset_filepath)
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1] * x_shape[2])
        y = np.argmax(y, axis=1)
        return x, y, sents
 
    
    def process(self, row):
        sentence = row['sentence']
        target_phrase = row['target']

        target_index = sentence.lower().index(target_phrase.lower())
        len_target = len(target_phrase)

        left_context = sentence[:target_index].strip()
        right_context = sentence[target_index + len_target:].strip()

        left_words = process_line(left_context)
        target_words = process_line(target_phrase)
        right_words = process_line(right_context)

        for feature in self.features:
            feature.process(left_words, target_words, right_words)

        left_context_vector = self.left_context_vector()
        right_context_vector = self.right_context_vector()
        target_word_vectors = self.target_word_vectors()

        x = [('LC', left_words), ('T', target_words), ('RC', right_words)] + [item for item in left_context_vector] + [item for item in right_context_vector] + [item for items in target_word_vectors for item in items]

        y = 0 if row['label'] == 'par-défaut' or row['label'] == 'simple' else 1
        return (np.array(x, dtype=object), y)
            
    def preprocess_data(self):
        if not self.PROCESSED_DATA_FILEPATH.exists():
            print("Prepossessing with features: ")
            print("\t" + "\n\t".join(self.features_args))

            self.features = self.get_features(self.features_args)
            # pd.DataFrame(features_args).to_csv(self.PROCESSED_DATA_PATH / 'features.txt', index=False, header=None)
            write_lines(self.features_args, self.PROCESSED_DATA_PATH / 'features.txt')

            print("Train dump not found. Preparing resources...")
            pd_data = pd.read_csv(DATASET_FILEPATH)
            # pd_data = pd_data[:5]
            data = {}
            labels = []
            # results = multiprocess(self.process, pd_data.iloc, len(pd_data), 'Preprocessing')
            for i in tqdm(range(len(pd_data)), desc='Preprocessing'):
                x, y = self.process(pd_data.iloc[i])
                for key, val in x:
                    if key in data:
                        data[key].append(val)
                    else:
                        data[key] = [val]
                labels.append(y)

            data['label'] = labels
            for feature in self.features:
                feature.save_cache_to_file()
                # print(feature.cache)

            dump_data = pd.DataFrame(data)
            dump_data.to_pickle(self.PROCESSED_DATA_FILEPATH)
            # dump_data.to_excel(self.PROCESSED_DATA_PATH / f'{DATASET_FILEPATH.stem}.xlsx', index=False)

    def load_preprocessed_data(self, train_size=0.7, valid_size=0.15, seed=42):
        self.preprocess_data()

        data = pd.read_pickle(self.PROCESSED_DATA_FILEPATH)
        size = len(data)

        d = data.iloc[:, 3:-1]
        y = data.loc[:, 'label'].tolist()
        d_sents = data.iloc[:, :3].to_numpy()

        x = []
        for i in range(size):
            x_ = []
            n = len(d.iloc[i]) // (MAX_TARGET_WORDS + 2)
            
            for j in range(MAX_TARGET_WORDS + 2):
                values = d.iloc[i][j*n : j*n + n].tolist()
                row = np.array([])
                for val in values:
                    row = np.append(row, val)
                x_.append(row)

            x.append(np.array(x_))

        x = np.array(x)
        y = np.array([[0,1] if label==1 else [1,0] for label in y])

        np.random.seed(seed)
        shuff_idx = np.random.permutation(np.arange(size))
        x, y = x[shuff_idx], y[shuff_idx]
        d_sents = d_sents[shuff_idx]

        train_size = int(train_size * size)
        valid_size = int(valid_size * size)
        x_train, y_train = x[:train_size], y[:train_size]
        x_valid, y_valid = x[train_size:train_size+valid_size], y[train_size:train_size+valid_size]
        x_test, y_test   = x[train_size+valid_size:], y[train_size+valid_size:]

        x_test_sents = d_sents[train_size+valid_size:]

        return x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents

    def load_preprocessed_data2(self, train_size=0.7, valid_size=0.15, seed=42):
        x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents = self.load_preprocessed_data(train_size, valid_size, seed)
    
        print('Reshaping data...')
        # x_train shape (6796, 14, 775)
        x_train_shape = x_train.shape
        x_train = x_train.reshape(x_train_shape[0], x_train_shape[1] * x_train_shape[2])
        y_train = np.argmax(y_train, axis=1)

        x_valid_shape = x_valid.shape
        x_valid = x_valid.reshape(x_valid_shape[0], x_valid_shape[1] * x_valid_shape[2])
        y_valid = np.argmax(y_valid, axis=1)

        x_test_shape = x_test.shape
        x_test = x_test.reshape(x_test_shape[0], x_test_shape[1] * x_test_shape[2])
        y_test = np.argmax(y_test, axis=1)

        return x_train, y_train, x_valid, y_valid, x_test, y_test, x_test_sents

if __name__ == '__main__':
    features_args =[
        'ContextFeature',
        'WordLengthFeature',
        'WordSyllableFeature',
        'VowelCountFeature',
        'TFIDFFeature',
        'WordRankFeature',
        # SynonymFeature(),
        # AntonymFeature(),
        #  LinguisticFeatures(lang),
    ]
    preprocessor = Preprocessor(features_args)
    preprocessor.preprocess_data()

    # tfidf_dump_file = REPO_DIR / f"resources/dumps/FR_tfidf_model.pk"

    # if not tfidf_dump_file.exists():
    #     data = pd.read_csv(DATASET_FILEPATH)
    #     docs = data['sentence'].tolist()
    #     # docs = data['paragraph'].tolist()

    #     tfidf = TfidfVectorizer(tokenizer=process_line)
    #     model = tfidf.fit(docs)
    #     utils.dump(model, tfidf_dump_file)
    # else:
    #     model = utils.load_dump(tfidf_dump_file)

    # feature_words = model.get_feature_names()

    # # self.feature_words = self.model.get_feature_names()

    # s = ['après', 'trois', 'mois', 'de', 'ce', 'régime'] +['la', 'voix']+['du', 'patient', 'est', 'plus', 'forte']
    # sentence = ' '.join(s)
    # x = model.transform([sentence])
    # X = np.array(x.todense())[0]
    # words = word_tokenize(sentence)
    # for word in words:
    #     if word in feature_words:
    #         index = feature_words.index(word)
    #         print(word, X[index])