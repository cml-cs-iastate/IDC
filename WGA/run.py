import os
import operator
import pickle as pkl

import src.settings as settings
from src.WGA.wgaSolver import wga_solver


if __name__ == '__main__':

    print_if_dict = True
    save_if_dict = True

    # ------------------------------------------------------------------------------------------------------------------

    # Load vocabulary processor
    with open(os.path.join(settings.output_directory, 'vocabulary.pkl'), 'rb') as f:
        vocabulary = pkl.load(f, encoding='bytes')

    # ------------------------------------------------------------------------------------------------------------------

        # Load training data
        with open(os.path.join(settings.output_directory, 'x_train.pkl'), 'rb') as f:
            x_train = pkl.load(f, encoding='bytes')
        with open(os.path.join(settings.output_directory, 'y_train.pkl'), 'rb') as f:
            y_train = pkl.load(f, encoding='bytes')
        with open(os.path.join(settings.output_directory, 'l_train.pkl'), 'rb') as f:
            l_train = pkl.load(f, encoding='bytes')

        # Run the WGA solver
        output = wga_solver(model_path=settings.model_path,
                            window_size=settings.filter_sizes,
                            target_layer_name=settings.target_layer_name,
                            max_length=settings.sequence_length,
                            batch_size=settings.batch_size,
                            x_train=x_train,
                            y_train=y_train,
                            l_train=l_train,
                            vocab_dict=vocabulary)

        post_processed_if_dict, if_dict, if_count, if_average = output

    # ------------------------------------------------------------------------------------------------------------------

        if print_if_dict:
            for key, value in post_processed_if_dict.items():
                print(f'The interpretation features of class: {key} is :')
                for item in sorted(value.items(), key=operator.itemgetter(1)):
                    print('{} : {:.2f}'.format(item[0], item[1]))
                print(f'The length of the dictionary for class {key} is = {len(value)}')
                print()

        # Save IF dictionaries
        if save_if_dict:
            with open(settings.post_processed_IF, 'wb') as f:
                pkl.dump(post_processed_if_dict, f, pkl.HIGHEST_PROTOCOL)
            with open(settings.IF, 'wb') as f:
                pkl.dump(if_dict, f, pkl.HIGHEST_PROTOCOL)
            with open(settings.count_IF, 'wb') as f:
                pkl.dump(if_count, f, pkl.HIGHEST_PROTOCOL)
            with open(settings.average_IF, 'wb') as f:
                pkl.dump(if_average, f, pkl.HIGHEST_PROTOCOL)
            print(f'Saving the files to {settings.post_processed_IF} is over!')