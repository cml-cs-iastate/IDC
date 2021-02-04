import os
import pickle5 as pkl

import src.settings as settings
from src.Evaluation.experimental_results import table_3, figure_4_intrinsic_validation, figure_5_increase_confidence, \
    table_2_with_interpretation_methods, table_2
from src.WGA.interpretation_de import get_interpretation_de


if __name__ == '__main__':

    run_ = 'Figure 4'  # {'Table 2', 'Table 3', 'Figure 4', 'Figure 5'}

    PGT_method = 'WGA'  # {'WGA', 'MILP', 'Hybrid'}
    # Load IF dictionary
    if PGT_method == 'WGA':
        with open(settings.post_processed_IF, 'rb') as f:
            if_dict = pkl.load(f)
    elif PGT_method == 'MILP':
        with open('../MILP/Interagreement/milp_if_solver.pkl', 'rb') as f:
            if_dict = pkl.load(f)
    elif PGT_method == 'Hybrid':
        with open('../Hybrid/MR_Docs_7K_Vocab_3000_min_5/milp_if_solver.pkl', 'rb') as f:
            if_dict = pkl.load(f)
    else:
        if_dict = ''

    # Load vocabulary processor
    with open(os.path.join(settings.output_directory, 'vocabulary.pkl'), 'rb') as f:
        vocabulary = pkl.load(f, encoding='bytes')

    # Load testing data
    with open(os.path.join(settings.output_directory, 'x_test.pkl'), 'rb') as f:
        x_test = pkl.load(f, encoding='bytes')
    with open(os.path.join(settings.output_directory, 'y_test.pkl'), 'rb') as f:
        y_test = pkl.load(f, encoding='bytes')
    with open(os.path.join(settings.output_directory, 'l_test.pkl'), 'rb') as f:
        l_test = pkl.load(f, encoding='bytes')
        # real sample length
        l_test = [l if l < settings.sequence_length else settings.sequence_length for l in l_test]

    # ------------------------------------------------------------------------------------------------------------------

    # table_1(settings.post_processed_IF)

    # ------------------------------------------------------------------------------------------------------------------
    # Table 2
    if run_ == 'Table 2':
        table_2(if_dict)

    # ------------------------------------------------------------------------------------------------------------------

    # figure_2(settings.post_processed_IF, 50)

    # ------------------------------------------------------------------------------------------------------------------
    # Table 3
    elif run_ == 'Table 3':

        # Get the heatmap interpretation of the interpretation method
        # {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift', 'random', 'if_dict' SD'+0.5'}
        interpretation_method = 'elrp'

        interpretation_heatmap = get_interpretation_de(model_path=settings.model_path,
                                                       batch_size=settings.batch_size,
                                                       x_test=x_test.copy(),
                                                       y_test=y_test.copy(),
                                                       l_test=l_test.copy(),
                                                       vocab_dict=vocabulary,
                                                       num_classes=settings.num_classes,
                                                       interpretation_method=interpretation_method)

        table_3(if_dict=if_dict,
                interpretation_heatmap=interpretation_heatmap,
                x_data=x_test,
                y_data=y_test,
                l_data=l_test,
                vocab_dict=vocabulary,
                threshold=0.3)

    # ------------------------------------------------------------------------------------------------------------------
    # Figure 4 and 5
    elif run_ == 'Figure 4' or 'Figure 5':

        # Get the heatmap interpretation of the interpretation method
        # {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift', 'random', 'if_dict' SD'+0.5'}
        interpretation_method = 'deeplift'

        # Get the interpretation heatmap
        if interpretation_method == 'random':
            random = True
            remove_based_on_if = False
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            interpretation_heatmap = []

        elif interpretation_method == 'if_dict':
            random = False
            remove_based_on_if = True
            ratios = [0]
            interpretation_method = 'if_dict'
            interpretation_heatmap = []

        else:
            random = False
            remove_based_on_if = False
            ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            interpretation_heatmap = get_interpretation_de(model_path=settings.model_path,
                                                           batch_size=settings.batch_size,
                                                           x_test=x_test.copy(),
                                                           y_test=y_test.copy(),
                                                           l_test=l_test.copy(),
                                                           vocab_dict=vocabulary,
                                                           num_classes=settings.num_classes,
                                                           interpretation_method=interpretation_method)

        # Loading model
        model = settings.classifier(num_classes=settings.num_classes,
                                    dropout=settings.dropout,
                                    embedding_dim=settings.embedding_dim,
                                    num_filters=settings.num_filters,
                                    filter_sizes=settings.filter_sizes,
                                    sequence_length=settings.sequence_length,
                                    fully_connected_dim=settings.fully_connected_dim,
                                    vocabulary_inv=vocabulary)
        model.load_weights(settings.model_path)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_accuracy = model.evaluate(x_test, y_test, verbose=0)[1]

        # Calculate the results
        for ratio in ratios:
            a = figure_4_intrinsic_validation(model=model,
                                              interpretation_heatmap=interpretation_heatmap.copy(),
                                              x_test=x_test.copy(),
                                              y_test=y_test.copy(),
                                              l_test=l_test.copy(),
                                              vocab_dict=vocabulary,
                                              if_dict=if_dict,
                                              model_accuracy=model_accuracy,
                                              random_=random,
                                              remove_ratio=ratio,
                                              remove_based_on_if=remove_based_on_if)

            # b = figure_5_increase_confidence(model=model,
            #                                  interpretation_heatmap=interpretation_heatmap.copy(),
            #                                  x_test=x_test.copy(),
            #                                  y_test=y_test.copy(),
            #                                  l_test=l_test.copy(),
            #                                  vocab_dict=vocabulary,
            #                                  if_dict=if_dict,
            #                                  random_=random,
            #                                  remove_ratio=ratio,
            #                                  remove_based_on_if=remove_based_on_if)

            print(f'The intrinsic validation results with {interpretation_method} for ratio: {ratio} equals {a}  \n')
            # print(f'The increase confidence results with {interpretation_method} for ratio: {ratio} equals {b} \n')

    # ------------------------------------------------------------------------------------------------------------------
