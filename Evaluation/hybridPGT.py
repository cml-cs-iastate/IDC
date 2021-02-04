import numpy as np
import pickle as pkl


def hybrid_PGT(milp_if, average_IF, vocabulary, num_classes):
    """
    :param milp_if: MILP interpretation feature dictionary                                  # shape dict(category: set(interpretation features), ...)
    :param average_IF: The average relevant score of each word per category                 # shape dict(category: dict(word: relevance score, ...), ...)
    :param vocabulary: The list of the most frequent words used to build BoW for milp_if    # shape list(word1, word2, ...)
    :param num_classes: The number of classes
    """
    hybrid = dict()

    for word in vocabulary:
        for category in [1, 2, 4, 5]:  # range(num_classes):
            hybrid.setdefault(category, list)
            if word in milp_if[category] and word in average_IF[category]:
                hybrid[category].append(average_IF[category][word])
            else:
                hybrid[category].append(0)

    hybrid_pgt = dict()
    for category in [1, 2, 4, 5]:  # range(num_classes):
        hybrid_pgt.setdefault(category, set())
        for word_idx in range(len(vocabulary)):
            if hybrid[category][word_idx] > 0:
                hybrid_pgt[category].add(vocabulary[word_idx])

    return hybrid_pgt, hybrid


with open('../MILP/milp_if.pkl', 'rb') as f:
    milp_if_ = pkl.load(f)

with open('average_IF.pkl', 'rb') as f:
    average_IF_ = pkl.load(f)
