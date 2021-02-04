import csv
import random
import numpy as np
import tensorflow as tf

import src.settings as settings
from src.Utils.utils import generate_ground_truth, uniform_remove_word
from src.Training.data_helpers import train_test_split, batch_iter
from src.WGA.interpretation_de import get_interpretation_de

tf.compat.v1.disable_eager_execution()


# Tested
def table_1(if_dict):
    """
    if_dict
    return the number of pseudo ground truth interpretation features per class
    :param if_dict: the dictionary of the interpretation features
    """
    for label, if_set in if_dict.items():
        print(f'The number of interpretation features of class {label} is {len(if_set)}')


# Tested
def compare(expert, if_dict, data, labels):
    """
    :param expert: list of the interpretation features by expert        shape [250, None]
    :param if_dict: the dictionary of the interpretation features
    :param data: the list of the 250 Amazon review                      shape [250, None]
    :param labels: the 250 label of the data samples                    shape [250, None]
    """
    # compare between entity_1 and entity_2 (Human with Human / or Human with Machine)
    a, b, c, d, counter = 0, 0, 0, 0, 0
    for idx, line in enumerate(data):
        label = labels[idx]

        # for word in line.strip().split():
        #     counter += 1
        #     if word in expert[idx] and word in if_dict[label]:
        #         a += 1
        #     elif word in expert[idx] and word not in if_dict[label]:
        #         b += 1
        #     elif word not in expert[idx] and word in if_dict[label]:
        #         c += 1
        #     elif word not in expert[idx] and word not in if_dict[label]:
        #         d += 1

        # TODO: Since the experts classified the review into negative and positive only
        if label == 0:
            l1 = 1
            l2 = 2
        else:
            l1 = 4
            l2 = 5

        for word in line.strip().split():
            counter += 1
            if word in expert[idx] and (word in if_dict[l1] or word in if_dict[l2]):
                a += 1
            elif word in expert[idx] and (word not in if_dict[l1] and word not in if_dict[l2]):
                b += 1
            elif word not in expert[idx] and (word in if_dict[l1] or word in if_dict[l2]):
                c += 1
            elif word not in expert[idx] and (word not in if_dict[l1] and word not in if_dict[l2]):
                d += 1

        po = (a + d) / (a + b + c + d)
        p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
        p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
        pe = p1 + p0
        k = (po - pe) / (1 - pe)

    print(f'a: {a}, b: {b}, c: {c}, d: {d}, sum(a,b,c,d): {a + b + c + d}, len(data): {counter}')
    print(f'po: {po}, p0: {p0}, p1: {p1}, pe: {pe}, Kappa: {k}')


# Tested
def table_2(if_dict):
    """
    250 Amazon user-review
    Calculate the the Cohen’s kappa inter-agreement rating of 250 user reviews of Amazon dataset.
    :param if_dict: the dictionary of the interpretation features
    """

    with open('../data/Interagreement/Amazon_250_review.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        data = list()
        labels = list()
        for sample in csv_reader:
            data.append(sample[content_idx])
            labels.append(int(sample[label_idx]))

    # load Natalie's data
    with open('../data/Interagreement/Natalie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        natalie = list()
        for sample in csv_reader:
            natalie.append(sample[0].strip().split())

    # load Carrie's data
    with open('../data/Interagreement/Carrie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        carrie = list()
        for sample in csv_reader:
            carrie.append(sample[0].strip().split())

    compare(natalie, if_dict, data, labels)
    compare(carrie, if_dict, data, labels)


# Hardcoded
def table_2_with_interpretation_methods():
    """
    250 Amazon user-review
    Calculate the the Cohen’s kappa inter-agreement rating of 250 user reviews of Amazon dataset with the
     interpretation methods.
    """

    with open('../data/Interagreement/Amazon_250_review.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        data = list()
        labels = list()
        for sample in csv_reader:
            data.append(sample[content_idx])
            labels.append(int(sample[label_idx]))

    # load Natalie's data
    with open('../data/Interagreement/Natalie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        natalie = list()
        for sample in csv_reader:
            natalie.append(sample[0].strip().split())

    # load Carrie's data
    with open('../data/Interagreement/Carrie.csv', 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        carrie = list()
        for sample in csv_reader:
            carrie.append(sample[0].strip().split())

    # get the heat-map
    x_train, y_train, l_train, _, _, _, vocab_dict = train_test_split(settings.data_source,
                                                                      settings.test_split,
                                                                      settings.sequence_length,
                                                                      vocabulary=settings.vocabulary_dict,
                                                                      shuffle=False)

    # Get the heatmap interpretation of the interpretation method
    interpretation_heatmap = get_interpretation_de(settings.model_path, x_train, y_train, settings.num_classes, interpretation_method='deeplift')
    threshold = 0.5
    print(' '.join([str(round(i, 1)) for i in interpretation_heatmap[0]]))
    print(data[0])

    def compare_(expert, interpretation_heatmap_, data_):
        a, b, c, d, counter = 0, 0, 0, 0, 0
        for line_idx, line in enumerate(data_):
            for word_idx, word in enumerate(line.strip().split()):
                if word_idx < settings.sequence_length:
                    counter += 1
                    if word in expert[line_idx] and interpretation_heatmap_[line_idx][word_idx] >= threshold:
                        a += 1
                    elif word in expert[line_idx] and interpretation_heatmap_[line_idx][word_idx] <= threshold:
                        b += 1
                    elif word not in expert[line_idx] and interpretation_heatmap_[line_idx][word_idx] >= threshold:
                        c += 1
                    elif word not in expert[line_idx] and interpretation_heatmap_[line_idx][word_idx] <= threshold:
                        d += 1

            po = (a + d) / (a + b + c + d)
            p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
            p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
            pe = p1 + p0
            k = (po - pe) / (1 - pe)

        print(f'a: {a}, b: {b}, c: {c}, d: {d}, sum(a,b,c,d): {a + b + c + d}, len(data): {counter}')
        print(f'po: {po}, p0: {p0}, p1: {p1}, pe: {pe}, Kappa: {k}')

    compare_(natalie, interpretation_heatmap, data)
    compare_(carrie, interpretation_heatmap, data)


# Tested
def sort_second(val):
    """
    function to return the second element of the two elements passed as the parameter
    """
    return val[1]


# Tested
def figure_2(if_dict, n):
    """
    if_dict
    Return the ten most relevant interpretation features from Hybrid-PGT. The interpretation features are listed in
    decreasing of their relevance score to each class.
    :param if_dict: the dictionary of the interpretation features
    :param n: top n interpretation features
    """
    for label, if_set in if_dict.items():
        temp = list()
        for if_word, relevance_value in if_set.items():
            temp.append((if_word, relevance_value))

        # sorts the array in descending according to second element
        temp.sort(key=sort_second, reverse=True)
        print(f'The most relevance {n} interpretation features of class {label} are : {temp[:n] if n < len(temp) else temp}')


# Tested
def table_3(if_dict, interpretation_heatmap, x_data, y_data, l_data, vocab_dict, threshold=0.3):
    """
    MR Dataset
    Return the interpretation effectiveness using different PGTs on the MR dataset in terms of Kappa enter-agreement,
    interpretation precision and recall
    :param if_dict: the dictionary of the interpretation features
    :param interpretation_heatmap: the heatmap interpretation generated by de
    :param x_data: The data being interpreted
    :param l_data: the real length of the documents being interpreted
    :param y_data: labels
    :param vocab_dict: the vocabulary dictionary of the trained model
    :param threshold: the threshold for the interpretation feature to be considered
    """
    precision = 0
    recall = 0
    kappa = 0
    counter = 0
    for sample_idx in range(len(x_data)):

        if sample_idx >= len(interpretation_heatmap):
            break

        counter += 1
        a, b, c, d = 0, 0, 0, 0

        for word_idx in range(l_data[sample_idx]):
            word_relevance_score = 0
            if x_data[sample_idx][word_idx] in vocab_dict and y_data[sample_idx] in if_dict:
                if vocab_dict[x_data[sample_idx][word_idx]] in if_dict[y_data[sample_idx]]:
                    word_relevance_score = 1

            if word_relevance_score == 1 and interpretation_heatmap[sample_idx][word_idx] >= threshold:
                a += 1  # TP
            elif word_relevance_score == 1 and interpretation_heatmap[sample_idx][word_idx] < threshold:
                b += 1  # FN
            elif word_relevance_score == 0 and interpretation_heatmap[sample_idx][word_idx] >= threshold:
                c += 1  # FP
            elif word_relevance_score == 0 and interpretation_heatmap[sample_idx][word_idx] < threshold:
                d += 1  # TN

        # Precision and Recall
        precision += a / ((a + c) + 1e-10)
        recall += a / ((a + b) + 1e-10)
        # Kappa
        po = (a + d) / (a + b + c + d)
        p1 = ((a + b) / (a + b + c + d)) * ((a + c) / (a + b + c + d))
        p0 = ((d + b) / (a + b + c + d)) * ((d + c) / (a + b + c + d))
        pe = p1 + p0
        kappa += (po - pe) / ((1 - pe) + 1e-10)

    precision = round(precision / counter, 2)
    recall = round(recall / counter, 2)
    kappa = round(kappa / counter, 2)

    print(f'Interpretation precision: {precision}, Interpretation recall: {recall}, Kappa: {kappa}')

    return [kappa, precision, recall]


# Tested
def random_perturb_input(x_data, l_data, remove_ratio):
    """
    """
    for doc_idx in range(len(x_data)):
        removed_words = int(l_data[doc_idx] * remove_ratio)
        # If the removed_words is less than 1
        removed_words = 1 if removed_words == 0 else removed_words
        remove_list = random.sample(range(l_data[doc_idx]), removed_words)
        x_data[doc_idx][remove_list] = 0
    return x_data


# Tested
def interpretation_perturb_input(x_data, l_data, interpretation_heatmap, most_relevant, remove_ratio):
    """
    """
    x_perturb = uniform_remove_word(x_data, l_data, remove_ratio, interpretation_heatmap, most_relevant)

    return x_perturb


# Tested
def perturb_input(x_data, y_data, if_dict, vocab_dict):
    """
    """
    # Generate the interpretation heatmap for the documents using the if_dict
    interpretation_heatmap = generate_ground_truth(x_data, y_data, vocab_dict, if_dict, binary=True)

    # remove words from input
    for doc_idx in range(len(x_data)):
        if_index = [i for i, word in enumerate(interpretation_heatmap[doc_idx]) if word == 1]
        x_data[doc_idx][if_index] = 0

    return x_data


# Tested
def figure_4_intrinsic_validation(model, interpretation_heatmap, x_test, y_test, l_test, vocab_dict, if_dict, model_accuracy,
                                  random_, remove_ratio, remove_based_on_if):
    """
    """
    batches = batch_iter(x_test, y_test, l_test, settings.batch_size, num_epochs=1)

    drop = 0
    batch_count = 0
    incorrect_prediction = 0
    for batch_idx, batch in enumerate(batches):
        batch_count += 1
        x_data, y_data, l_data = batch

        predictions = model.predict(x_data)

        # Perturb Input
        if random_:
            x_perturb = random_perturb_input(x_data, l_data, remove_ratio)
        else:
            # Only change remove_ratio and keep most_relevant unchanged
            if remove_based_on_if:
                x_perturb = perturb_input(x_data, y_data, if_dict, vocab_dict)
            else:
                x_perturb = interpretation_perturb_input(x_data, l_data, interpretation_heatmap, most_relevant=True, remove_ratio=remove_ratio)

        predictions_perturb = model.predict(x_perturb)

        for doc_idx in range(len(x_data)):
            label = np.argmax(predictions[doc_idx])
            # Only count the correct predictions
            if label == y_data[doc_idx]:
                drop += max(0, predictions[doc_idx][label] - predictions_perturb[doc_idx][label]) / predictions[doc_idx][label]
            else:
                incorrect_prediction += 1

    intrinsic_validation = model_accuracy - (drop / ((batch_count * settings.batch_size) - incorrect_prediction))

    return intrinsic_validation


# Tested
def figure_5_increase_confidence(model, interpretation_heatmap, x_test, y_test, l_test, vocab_dict, if_dict,
                                 random_, remove_ratio, remove_based_on_if):
    """
    """
    batches = batch_iter(x_test, y_test, l_test, settings.batch_size, num_epochs=1)

    increase_conf = 0
    batch_count = 0
    incorrect_prediction = 0
    for batch_idx, batch in enumerate(batches):
        batch_count += 1
        x_data, y_data, l_data = batch

        predictions = model.predict(x_data)

        # Perturb Input
        if random_:
            x_perturb = random_perturb_input(x_data, l_data, remove_ratio)
        else:
            # Only change remove_ratio and keep most_relevant unchanged
            if remove_based_on_if:
                x_perturb = perturb_input(x_data, y_data, if_dict, vocab_dict)
            else:
                x_perturb = interpretation_perturb_input(x_data, l_data, interpretation_heatmap, most_relevant=False, remove_ratio=remove_ratio)

        predictions_perturb = model.predict(x_perturb)

        for doc_idx in range(len(x_data)):
            label = np.argmax(predictions[doc_idx])
            # Only count the correct predictions
            if label == y_data[doc_idx]:
                increase_conf += 1 if predictions[doc_idx][label] < predictions_perturb[doc_idx][label] else 0
            else:
                incorrect_prediction += 1

    return increase_conf / ((batch_count * settings.batch_size) - incorrect_prediction)
