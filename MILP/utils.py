import collections
import itertools
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer


# Tested
def one_hot_doc(sentences, labels, max_num_words):
    """
    generate one-hot representation (bag-of-words) for each document per class
    :param sentences:
    :param labels:
    :param max_num_words: maximum number of word in vocabulary dict
    """

    word_counts = collections.Counter(itertools.chain(*sentences)).most_common()
    if max_num_words != 0 and max_num_words < len(word_counts):
        word_counts = word_counts[:max_num_words]
    vocab = [w[0] for w in word_counts]
    print('Building vocab is done!')

    sentences_vectors = dict()
    for idx, sentence in enumerate(sentences):
        sent_vec = []
        for token in vocab:
            sent_vec.append(1 if token in sentence else 0)
        sentences_vectors.setdefault(labels[idx], list())
        if sum(sent_vec) > 0:
            sentences_vectors[labels[idx]].append(sent_vec)
    print('Building one-hot representation is done!')

    return sentences_vectors, vocab


# Tested
def one_hot_doc_with_keras(sentences, labels, max_num_words):
    """
    generate one-hot representation (bag-of-words) for each document per class
    :param sentences:
    :param labels:
    :param max_num_words: maximum number of word in vocabulary dict
    """

    labels = np.array(labels)

    # Tokenize words and build vocabulary
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(sentences)
    vocab = list(tokenizer.word_index.keys())[:max_num_words]
    print('Building vocab is done!')

    # Build bag-of-words representation
    vectors = tokenizer.texts_to_matrix(sentences)
    print('Building bag-of-words representation is done!')

    # Empty sentences for space utilization
    sentences = None

    sentences_vectors = dict()
    label_set = set(labels)
    for label_idx in label_set:
        sentences_vectors[label_idx] = vectors[labels == label_idx].tolist()

        # Remove the selected input sample of label_idx category from vectors and labels. This helps not to duplicate
        # the size of the data which save memory space
        not_indices = labels != label_idx
        labels = labels[not_indices]
        vectors = vectors[not_indices]

    print('Building sentence vectors dictionary is done!')

    return sentences_vectors, vocab
