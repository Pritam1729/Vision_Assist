from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def create_tokenizer(descriptions, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(descriptions)
    return tokenizer

def save_tokenizer(tokenizer, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def pad_sequences_wrapper(sequences, maxlen):
    return pad_sequences(sequences, maxlen=maxlen, padding='post')

