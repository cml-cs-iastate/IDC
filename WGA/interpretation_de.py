import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

import src.settings as settings
from src.Training.data_helpers import batch_iter

# Import DeepExplain
from deepexplain.tensorflow import DeepExplain


# Tested
def get_interpretation_de(model_path, batch_size, x_test, y_test, l_test, vocab_dict, num_classes, interpretation_method='grad*input'):
    """
    interpretation_method: {'saliency', 'grad*input', 'intgrad', 'elrp', 'deeplift'}
    """
    output = list()

    with DeepExplain(session=tf.compat.v1.keras.backend.get_session()) as de:

        model = settings.classifier(num_classes=settings.num_classes,
                                    dropout=settings.dropout,
                                    embedding_dim=settings.embedding_dim,
                                    num_filters=settings.num_filters,
                                    filter_sizes=settings.filter_sizes,
                                    sequence_length=settings.sequence_length,
                                    fully_connected_dim=settings.fully_connected_dim,
                                    vocabulary_inv=vocab_dict)
        model.load_weights(model_path)
        # print('Loading model is over!')

        # --------------------------------------------------------------------------------------------------------------

        # shape [None, 50, 128]
        embedding_tensor = model.layers[1].output

        # Get tensor before the final activation (logits)
        # shape [batch_size, num_classes]
        logits_tensor = model.layers[-2].output

        batches = batch_iter(x_test, y_test, l_test, batch_size, num_epochs=1)

        for batch_idx, batch in enumerate(batches):

            x_data, y_data, l_data = batch

            # Convert the index labels to one-hot format
            ys = np.zeros(shape=(len(y_data), num_classes))
            for i, label in enumerate(y_data):
                ys[i, label] = 1

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

            # Process the interpretation heatmap
            output = list()
            for sample_idx in range(len(l_data)):
                sample_heatmap = interpretation_heatmap[sample_idx][:l_data[sample_idx]]
                sample_heatmap = sample_heatmap - np.mean(sample_heatmap)
                sample_heatmap = np.maximum(sample_heatmap, 0)
                max_value = np.max(sample_heatmap) if np.max(
                    sample_heatmap) > 0 else 0.000001  # Replace the zero value by 0.000001
                new_cams = sample_heatmap / max_value
                output.append(new_cams)

        return output
