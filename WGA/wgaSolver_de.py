import collections
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import src.settings as settings
from src.Classifiers.cnn import cnn_classifier
from src.WGA.wgaSolver import post_processing
from src.Training.data_helpers import batch_iter

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain


# Tested
def wga_solver_de(model_path, batch_size, x_train, y_train, l_train, vocab_dict, num_classes, interpretation_method='grad*input'):
    """
    interpretation_method: {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift'}
    """

    with DeepExplain(session=tf.compat.v1.keras.backend.get_session()) as de:

        model = cnn_classifier(num_classes=settings.num_classes,
                               dropout=settings.dropout,
                               embedding_dim=settings.embedding_dim,
                               num_filters=settings.num_filters,
                               filter_sizes=settings.filter_sizes,
                               sequence_length=settings.sequence_length,
                               fully_connected_dim=settings.fully_connected_dim,
                               vocabulary_inv=vocab_dict)

        model.load_weights(model_path)
        print('Loading model is over!')

        if_dict = collections.defaultdict(dict)     # The summation of the relevance scores of each interpretation feature in each document
        if_count = collections.defaultdict(dict)    # The count of each interpretation feature
        if_average = collections.defaultdict(dict)  # Average relevance score for each interpretation features

        # --------------------------------------------------------------------------------------------------------------

        # shape [None, 50, 128]
        embedding_tensor = model.layers[1].output

        # Get tensor before the final activation (logits)
        # shape [batch_size, num_classes]
        logits_tensor = model.layers[-2].output

        batches = batch_iter(x_train, y_train, l_train, batch_size, num_epochs=1)

        for batch_idx, batch in enumerate(batches):

            print(f'Processing batch {batch_idx} ...')
            x_data, y_data, l_data = batch

            # convert the index labels to one-hot format
            ys = np.zeros(shape=(len(y_data), num_classes))
            for i, label in enumerate(y_data):
                ys[i, label] = 1

            y_prediction = model.predict(x_data)
            y_prediction = np.argmax(y_prediction, axis=1)

            # Interpretation
            # ----------------------------------------------------------------------------------------------------------

            # Evaluate the embedding tensor on the model input
            embedding_function = K.function([model.input], [embedding_tensor])
            embedding_output = embedding_function([x_data])

            # Run DeepExplain with the embedding as input
            # Shape [batch_size, sequence_length, embedding_size]
            heat_map = de.explain(interpretation_method, logits_tensor * ys, embedding_tensor, embedding_output[0])

            # Sum the values for the embedding dimension to get the relevance value of each word
            # Shape [batch_size, sequence_length]
            interpretation_heatmap = np.sum(heat_map, axis=-1)
            # Remove negative values
            interpretation_heatmap = np.maximum(interpretation_heatmap, 0)

            for sample_idx in range(len(x_data)):
                words = [vocab_dict[w] if w in vocab_dict else '<PAD/>' for w in x_data[sample_idx][:l_data[sample_idx]]]
                sample_heatmap = np.round(interpretation_heatmap[sample_idx][:l_data[sample_idx]] / (max(interpretation_heatmap[sample_idx][:l_data[sample_idx]]) + 0.00001), 2)

                # get the interpretation feature information
                if y_data[sample_idx] == y_prediction[sample_idx]:
                    for j, w in enumerate(sample_heatmap):
                        if w > 0:
                            if_dict[y_data[sample_idx]][words[j]] = if_dict[y_data[sample_idx]].setdefault(words[j], 0) + w
                            if_count[y_data[sample_idx]][words[j]] = if_count[y_data[sample_idx]].setdefault(words[j], 0) + 1
        print('Generating the interpretation features for all the data is over')

        # ------------------------------------------------------------------------------------------------------------------

        # Get the average importance values of the interpretation_features
        for category, sub_dict in if_dict.items():
            for word, value in sub_dict.items():
                if_average[category][word] = np.divide(value, if_count[category][word])
        print(f'Get the average importance values of the interpretation_features is Over!')

        # ------------------------------------------------------------------------------------------------------------------

        # Post-Processing
        post_processed_if_dict = post_processing(if_average, if_count)
        print(f'Post-Processing is over!')

        return [post_processed_if_dict, if_dict, if_count, if_average]
