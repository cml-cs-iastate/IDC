import os
import pickle5 as pkl

import src.settings as settings
from src.WGA.attentionInterpretation import attention_interpretation, attention_interpretation_h
from src.Evaluation.experimental_results import table_3, figure_4_intrinsic_validation, figure_5_increase_confidence


if __name__ == '__main__':

    PGT_method = 'Hybrid'  # {'WGA', 'MILP', 'Hybrid'}
    # Load IF dictionary
    if PGT_method == 'WGA':
        with open(settings.post_processed_IF, 'rb') as f:
            if_dict = pkl.load(f)
    elif PGT_method == 'MILP':
        with open('../MILP/MR_Seq_60_Vocab_1000_min_10/milp_if_solver.pkl', 'rb') as f:
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

    # Load model
    if settings.model == 'h_attention_classifier':
        model = settings.classifier(num_classes=settings.num_classes,
                                    dropout=settings.dropout,
                                    embedding_dim=settings.embedding_dim,
                                    rnn_dim=settings.embedding_dim,
                                    sequence_length=settings.sequence_length,
                                    fully_connected_dim=settings.fully_connected_dim,
                                    vocabulary_inv=vocabulary)
    else:
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

    if settings.model == 'Attention':
        interpretation_heatmap = attention_interpretation(model, x_test, settings.filter_sizes, settings.target_layer_name, l_test)
    else:
        interpretation_heatmap = attention_interpretation_h(model, x_test, l_test)

    table_3(if_dict=if_dict,
            interpretation_heatmap=interpretation_heatmap,
            x_data=x_test,
            y_data=y_test,
            l_data=l_test,
            vocab_dict=vocabulary,
            threshold=0.8)

    # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     a = figure_4_intrinsic_validation(model=model,
    #                                       interpretation_heatmap=interpretation_heatmap.copy(),
    #                                       x_test=x_test.copy(),
    #                                       y_test=y_test.copy(),
    #                                       l_test=l_test.copy(),
    #                                       vocab_dict=vocabulary,
    #                                       if_dict=if_dict,
    #                                       model_accuracy=model_accuracy,
    #                                       random_=False,
    #                                       remove_ratio=ratio,
    #                                       remove_based_on_if=False)
    #
    #     print(f'The intrinsic validation results of attention network for ratio: {ratio} is {a}')
