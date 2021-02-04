import logging
import pickle5 as pkl

from src.Training.data_helpers import load_data_and_labels
from src.MILP.utils import one_hot_doc
from src.MILP.ortoolsSolver import milp_solver


if __name__ == '__main__':

    data_source_ = ''
    input_output_dir = 'Interagreement/'
    log_file_name = input_output_dir + 'log_file'
    max_words_ = 3000
    num_classes_ = 2
    min_num_words_ = 20

    open_saved_files = True
    save_one_hot_encoding = False
    save_output_results = True

    # Logging
    logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s - %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
    logging.info(f'data_source_: {data_source_} \ninput_output_dir: {input_output_dir} \nmax_words_: {max_words_} \nnum_classes_: {num_classes_} \nmin_num_words_: {min_num_words_} \n')
    logging.info(f'open_saved_files: {open_saved_files} \nsave_one_hot_encoding: {save_one_hot_encoding} \nsave_output_results: {save_output_results} \n')

    # ------------------------------------------------------------------------------------------------------------------

    if open_saved_files:
        with open(input_output_dir + 'vocabulary_.pkl', 'rb') as f:
            vocabulary_ = pkl.load(f)
            print('Loading vocabulary is over!')
        with open(input_output_dir + 'data_.pkl', 'rb') as f:
            data_ = pkl.load(f)
            print('Loading data is over!')

        logging.info(f'vocabulary_ path: {input_output_dir}vocabulary_.pkl \n')
        logging.info(f'data_ path: {input_output_dir}data_.pkl \n')

    else:
        # Load and preprocess data
        sentences_, labels_, _ = load_data_and_labels(data_source_, remove_stopword=True, run_with_keras=False)
        print('Load data is done!')

        data_, vocabulary_ = one_hot_doc(sentences_, labels_, max_num_words=max_words_)
        print(f'One hot encoding is over')

        # Empty sentences_ and labels_
        sentences_ = None
        labels_ = None

        # save the files for future re-usability
        if save_one_hot_encoding:
            with open(input_output_dir + 'data_.pkl', 'wb') as f:
                pkl.dump(data_, f, pkl.HIGHEST_PROTOCOL)

            with open(input_output_dir + 'vocabulary_.pkl', 'wb') as f:
                pkl.dump(vocabulary_, f, pkl.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------------------------------------------------------

    # Convert data_ from 5 stars rating into positive(1)/negative(0) rating
    if num_classes_ == 5:
        if 0 in data_:
            del data_[0]
        if 3 in data_:
            del data_[3]

        data_[0] = data_[1] + data_[2]
        data_[1] = data_[4] + data_[5]

        del data_[2]
        del data_[4]
        del data_[5]
        print('Convert data_ from 5 stars rating into positive(1)/negative(0) rating is over!')

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

    # Select the first 25000 documents
    # data_[0] = data_[0][:25000]
    # data_[1] = data_[1][:25000]

    print(f'the length of the data_ dict = {len(data_)}')
    for k, v in data_.items():
        print(f'class: {k}, with length {len(v)}')
        logging.info(f'class: {k}, with length {len(v)} \n')

    milp_solver_if_ = milp_solver(data=data_, vocabulary=vocabulary_, num_classes=len(data_), vocab_size=max_words_)

    for k, v in milp_solver_if_.items():
        print(f'class: {k}, with length {len(v)}\n{v}')
        logging.info(f'class: {k}, with length {len(v)}\n{v} \n')

    if save_output_results:
        with open(input_output_dir + 'milp_if_solver.pkl', 'wb') as f:
            pkl.dump(milp_solver_if_, f, pkl.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------------------------------------------------------
    # TODO: Debugging
    # data_ = {0: [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]],
    #          1: [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]}
    # vocabulary_ = ['he', 'she', 'it', 'they', 'those', 'cat', 'dog', 'mouse', 'pig', 'fly']
    # num_classes_ = 2
    # vocab_size_ = 10
    #
    # t = milp_solver(data_, vocabulary_, num_classes_, vocab_size_)
    #
    # for cls, word in t.items():
    #     print(f'{cls} : {word}')
