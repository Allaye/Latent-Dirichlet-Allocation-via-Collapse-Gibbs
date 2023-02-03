import random
import numpy as np
from tqdm import tqdm
from util import bundle_all, NUM_TOPICS, ALPHA, BETA


class LDA:
    # implementation based on "A Theoretical and Practical Implementation
    # Tutorial on Topic modeling and Gibbs Sampling paper by William M. Darling"
    def __init__(self, corpus, vocab_size, num_iter=100):
        self.zee = []
        self.total_docs = len(corpus)
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.num_iter = num_iter

    def collapse_gibbs(self):
        for _, doc in enumerate(self.corpus):
            zeed = np.random.randint(low=0, high=NUM_TOPICS, size=len(doc))
            self.zee.append(zeed)

        ndk = np.zeros((self.total_docs, NUM_TOPICS))
        for d in range(self.total_docs):
            for k in range(NUM_TOPICS):
                ndk[d, k] = np.sum(self.zee[d] == k)

        nkw = np.zeros((NUM_TOPICS, self.vocab_size))
        for doc_idx, doc in enumerate(self.corpus):
            for i, word in enumerate(doc):
                topic = self.zee[doc_idx][i]
                nkw[topic, word] += 1

        nk = np.sum(nkw, axis=1)
        topic_len = [i for i in range(NUM_TOPICS)]

        for _ in tqdm(range(self.num_iter)):
            for doc_idx, doc in enumerate(self.corpus):
                for i in range(len(doc)):
                    word = doc[i]
                    topic = self.zee[doc_idx][i]

                    ndk[doc_idx, topic] -= 1
                    nkw[topic, word] -= 1
                    nk[topic] -= 1

                    pee_zee = (ndk[doc_idx, :] + ALPHA) * (nkw[:, word] + BETA) / (nk[:] + BETA * self.vocab_size)
                    topic = random.choices(topic_len, weights=pee_zee, k=1)[0]

                    self.zee[doc_idx][i] = topic
                    ndk[doc_idx, topic] += 1
                    nkw[topic, word] += 1
                    nk[topic] += 1
        return self.zee, ndk, nkw, nk


if __name__ == "__main__":
    corpus, vocal_size, vocal_str_idx = bundle_all()
    lda = LDA(corpus, vocal_size)
    Z, ndk, nkw, nk = lda.collapse_gibbs()
    phi = nkw / nk.reshape(NUM_TOPICS, 1)

    num_words = 5
    for k in range(NUM_TOPICS):
        common_words = np.argsort(phi[k])[::-1][:num_words]
        print(f"Common words for Topic {k}")
        for word in common_words:
            print(f"{vocal_str_idx}")

