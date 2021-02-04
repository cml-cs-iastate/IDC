import os
import logging
import collections
import numpy as np
import pickle as pkl

import src.settings as settings
from src.MILP.ortoolsSolver import milp_solver


if __name__ == '__main__':

    data_source_ = ''
    log_file_name = 'log_file'
    max_words_ = 3000
    num_classes_ = 2
    min_num_words_ = 2

    open_saved_files = False
    save_one_hot_encoding = True
    save_output_results = True

    # Logging
    logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s - %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    logging.info(f'data_source_: {data_source_} \n \nmax_words_: {max_words_} \nnum_classes_: {num_classes_} \nmin_num_words_: {min_num_words_} \n')
    logging.info(f'open_saved_files: {open_saved_files} \nsave_one_hot_encoding: {save_one_hot_encoding} \nsave_output_results: {save_output_results} \n')

    # ------------------------------------------------------------------------------------------------------------------

    if open_saved_files:
        with open('vocabulary_.pkl', 'rb') as f:
            vocabulary_ = pkl.load(f)
            print('Loading vocabulary is over!')
        with open('data_.pkl', 'rb') as f:
            data_ = pkl.load(f)
            print('Loading data is over!')

    else:
        # Load vocabulary processor and Load training data
        with open(os.path.join(settings.output_directory, 'vocabulary.pkl'), 'rb') as f:
            vocabulary = pkl.load(f, encoding='bytes')
        with open(os.path.join(settings.output_directory, 'x_train.pkl'), 'rb') as f:
            x_train = pkl.load(f, encoding='bytes')
        with open(os.path.join(settings.output_directory, 'y_train.pkl'), 'rb') as f:
            y_train = pkl.load(f, encoding='bytes')

        # Load IF relevance score
        with open(settings.average_IF, 'rb') as f:
            if_average = pkl.load(f)

        # ------------------------------------------------------------------------------------------------------------------

        # Combine the Average relevance score for multiple classes
        if_dict = if_average[0]
        for word, value in if_average[1].items():
            if word in if_dict:
                if_dict[word] = (value + if_dict[word]) / 2
            else:
                if_dict[word] = value

        # Create the super_vector
        super_vector = list()
        super_vector_scores = list()
        for word, value in if_dict.items():
            super_vector.append(word)
            super_vector_scores.append(value)

        super_vector = np.array(super_vector)
        super_vector_scores = np.array(super_vector_scores)

        # Sort the super_vector based on the average IF scores
        argsort = super_vector_scores.argsort()[::-1]
        # Resort super_vector based on argsort and cut it to the vocab_size
        vocabulary_ = super_vector[argsort][:max_words_]

        # Generate the vocabulary
        vocab_new_inv = {word: idx for idx, word in enumerate(vocabulary_)}

        data_ = collections.defaultdict(list)
        for doc_idx, doc in enumerate(x_train):
            temp = [0] * max_words_
            for word_idx in doc:
                word = vocabulary[word_idx]
                if word in vocab_new_inv:
                    temp[vocab_new_inv[word]] = 1

            data_[y_train[doc_idx]].append(temp)

    # ------------------------------------------------------------------------------------------------------------------

    # Remove the documents that have less than min_num_words_ words in the vocabulary dictionary
    if min_num_words_ != 0:
        for i in {0, 1}:
            end = len(data_[i])
            j = 0
            while j < end:
                if sum(data_[i][j]) < min_num_words_:
                    data_[i].pop(j)
                    end -= 1
                else:
                    j += 1
        print(f'Remove the documents that have less than {min_num_words_} words in the vocabulary dictionary is over!')

    # ------------------------------------------------------------------------------------------------------------------

    print(f'the length of the data_ dict = {len(data_)}')
    for k, v in data_.items():
        print(f'class: {k}, with length {len(v)}')
        logging.info(f'class: {k}, with length {len(v)} \n')

    milp_solver_if_ = milp_solver(data=data_, vocabulary=vocabulary_, num_classes=len(data_), vocab_size=max_words_)

    for k, v in milp_solver_if_.items():
        print(f'class: {k}\n{v} \n with length: {len(v)}')
        logging.info(f'class: {k}\n{v} \n')

    if save_one_hot_encoding:
        with open('data_.pkl', 'wb') as f:
            pkl.dump(data_, f, pkl.HIGHEST_PROTOCOL)
        with open('vocabulary_.pkl', 'wb') as f:
            pkl.dump(vocabulary_, f, pkl.HIGHEST_PROTOCOL)

    if save_output_results:
        with open('milp_if_solver.pkl', 'wb') as f:
            pkl.dump(milp_solver_if_, f, pkl.HIGHEST_PROTOCOL)