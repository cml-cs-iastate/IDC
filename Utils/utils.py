import csv
import numpy as np

from tensorflow.keras import backend as K


# Tested
def jaccard_sim(true_features, predicted_features):
    """
    Calculate Jaccard Similarity (Only taking the similarity of 1's into account not the 0's)
    :param true_features: the true interpretation features of the input text                # shape [batch_size, sequence_length]
    :param predicted_features: the predicted interpretation features of the input text      # shape [batch_size, sequence_length]
    """
    intersection = K.sum(true_features * predicted_features, axis=1)
    union = K.sum(true_features + predicted_features, axis=1)
    j_sim = intersection / ((union - intersection) + 1e-10)

    j_loss = K.mean(1.0 - j_sim)
    j_accuracy = K.mean(j_sim)

    return j_loss, j_accuracy


def generate_ground_truth(input_x, input_y, vocabulary, if_dict, binary=True):
    # convert input_y from one-hot format into index format
    # input_y = K.argmax(input_y, axis=1)
    truth = []
    for idx, sample in enumerate(input_x):
        temp = []
        for word in sample:
            if word in vocabulary and input_y[idx] in if_dict:
                if vocabulary[word] in if_dict[input_y[idx]]:
                    if binary:
                        temp.append(1)
                    else:
                        temp.append(if_dict[input_y[idx]][word])
                else:
                    temp.append(0)
            else:
                temp.append(0)
        truth.append(temp)
    # shape [batch_size, sequence] (must be check)
    return truth


def uniform_remove_word(x_data, real_lengths, remove_ratio, interpretation_heatmap, most_relevant):
    """
    remove top/bottom words from input based on their important values from the heat-map
    :param x_data: the input data samples
    :param real_lengths: the lengths of the input data samples
    :param remove_ratio: the ratio of the words to be remove from the real length of the input sample [0, 1]
    :param interpretation_heatmap: contains the importance value for each word
    :param most_relevant: if True, remove the most relevant words, otherwise, remove the least relevant words
    :return:
    """
    # remove words from input
    for doc_idx in range(len(x_data)):
        # Number of words to be removed
        removed_words = int(real_lengths[doc_idx] * remove_ratio)
        removed_words = 1 if removed_words == 0 else removed_words

        if most_relevant:
            remove_list = np.array(interpretation_heatmap[doc_idx]).argsort()[-removed_words:]
        else:
            remove_list = np.array(interpretation_heatmap[doc_idx]).argsort()[::-1][-removed_words:]

        x_data[doc_idx][remove_list] = 0

    return x_data


def average_drop(logits, logits_after_removal, predictions, batch_size):
    avg_drop = 0
    for batch_idx in range(len(logits)):
        for doc_idx in range(batch_size):
            avg_drop += max(0, logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] - logits_after_removal[batch_idx][doc_idx][predictions[batch_idx][doc_idx]]) / logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]]
    return avg_drop / (len(logits)*batch_size)


def increase_confidence(logits, logits_after_removal, predictions, batch_size):
    increase_conf = 0
    for batch_idx in range(len(logits)):
        for doc_idx in range(batch_size):
            increase_conf += 1 if logits[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] < logits_after_removal[batch_idx][doc_idx][predictions[batch_idx][doc_idx]] else 0
    return increase_conf / (len(logits)*batch_size)


def precision_and_recall(prediction_file):
    # Read the CSV file and get its contents
    with open(prediction_file, 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        true_class = header.index('True class')
        prediction = header.index('Prediction')

        tp, fn, fp = 0, 0, 0
        for line in csv_reader:
            if line[true_class] == '1' and line[prediction] == '1.0':
                tp += 1
            elif line[true_class] == '1' and line[prediction] == '0.0':
                fn += 1
            elif line[true_class] == '0' and line[prediction] == '1.0':
                fp += 1

        precision = tp / ((tp + fp) + 1e-10)
        recall = tp / ((tp + fn) + 1e-10)

        return precision, recall
