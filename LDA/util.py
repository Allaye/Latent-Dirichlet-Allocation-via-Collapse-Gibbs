import numpy as np
import spacy as sc
import random
import pandas as pd
from collections import Counter

# load email dataset
dataset = pd.read_csv('mail_data.csv')

# hyper parameters
BETA, ALPHA, NUM_TOPICS = 0.1, 0.1, 20

# load tokenizer
model = sc.load('en_core_web_sm')

# seed for reproducibility
np.random.seed(20)
random.seed(20)


def frequency_generator(dataset, max_docs=1000):
    frequency = Counter()
    all_stopwords = model.Defaults.stop_words
    all_stopwords.add('enron')
    total_tokens = 0

    for doc in dataset[:max_docs]:
        all_tokens = model.tokenizer(doc)
        for token in all_tokens:
            token_text = token.text.lower()
            if token_text not in all_stopwords and token.is_alpha:
                total_tokens += 1
                frequency[token_text] += 1
    return frequency


def collect_vocab(freqs, freq_threshold=1):
    vocab = {}
    vocab_idx = 0
    vocab_str_idx = {}
    for word in freqs:
        if freqs[word] >= freq_threshold:
            vocab[word] = vocab_idx
            vocab_str_idx[vocab_idx] = word
            vocab_idx += 1
    return vocab, vocab_str_idx


def dataset_tokenizer(data, vocab, max_docs=1000):
    total_token = 0
    total_docs = 0
    docs = []

    for doc in data[:max_docs]:
        tokens = model.tokenizer(doc)

        if len(tokens) >= 2:
            doc = []
            for token in tokens:
                token_text = token.text.lower()
                if token_text in vocab:
                    doc.append(token_text)
                    total_token += 1
            total_docs += 1
            docs.append(doc)
    print(f"Total number of emails: {total_docs}")
    print(f"Total number of tokens {total_token}")

    corpus = []
    for doc in docs:
        corpus_d = []

        for token in doc:
            corpus_d.append(vocab[token])
        corpus.append(np.asarray(corpus_d))
    return docs, corpus


def bundle_all():
    data = dataset['Message'].sample(frac=0.200, random_state=20).values
    frequency = frequency_generator(data)
    vocab, vocab_str_idx = collect_vocab(frequency)
    _, corpus = dataset_tokenizer(data, vocab)
    print(f'Vocab Size: {len(vocab)}')
    return corpus, len(vocab), vocab_str_idx

