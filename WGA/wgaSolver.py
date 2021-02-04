import collections
import numpy as np
import tensorflow as tf

from tensorflow.keras import models

from src.Training.data_helpers import batch_iter
from src.WGA.interpretation import get_interpretation


# Tested
def wga_solver(model_path, window_size, target_layer_name, max_length, batch_size, x_train, y_train, l_train, vocab_dict):
    """
    This function generate interpretation features using the dataset that has been trained on the same dataset that we
    need to generate the interpretation features from.
    IMPORTANT: This function need to run on the Amazon_Yelp dataset on the server due to database size.
    """
    tf.compat.v1.disable_eager_execution()

    if_dict = collections.defaultdict(dict)     # The summation of the relevance scores of each interpretation feature in each document
    if_count = collections.defaultdict(dict)    # The count of each interpretation feature
    if_average = collections.defaultdict(dict)  # Average relevance score for each interpretation features

    # Load the model
    model = models.load_model(model_path)
    print('Loading model is over!')

    # ------------------------------------------------------------------------------------------------------------------

    batches = batch_iter(x_train, y_train, l_train, batch_size, num_epochs=1)

    y_prediction = model.predict(x_train)
    y_prediction = np.argmax(y_prediction, axis=1)
    print('Classification is over!')

    # ------------------------------------------------------------------------------------------------------------------

    for batch_idx, batch in enumerate(batches):

        print(f'Processing batch {batch_idx} ...')
        x_data, y_data, l_data = batch

        # Get the heatmap interpretation of the interpretation method
        interpretation_heatmap = get_interpretation('grad_cam', model, x_data, y_data, window_size, target_layer_name)

        for sample_idx in range(batch_size):
            # Get the real sample length
            real_length = l_data[sample_idx] if l_data[sample_idx] < max_length else max_length
            # Get the interpretation feature information (For correctly classified documents only)
            if y_data[sample_idx] == y_prediction[(batch_idx * batch_size) + sample_idx]:
                for j, w in enumerate(interpretation_heatmap[sample_idx][:real_length]):
                    if w > 0 and x_data[sample_idx][j] in vocab_dict:
                        if_dict[y_data[sample_idx]][vocab_dict[x_data[sample_idx][j]]] = if_dict[y_data[sample_idx]].setdefault(vocab_dict[x_data[sample_idx][j]], 0) + w
                        if_count[y_data[sample_idx]][vocab_dict[x_data[sample_idx][j]]] = if_count[y_data[sample_idx]].setdefault(vocab_dict[x_data[sample_idx][j]], 0) + 1
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


# Tested
def post_processing(if_average, if_count, alpha_threshold=0.1, beta_threshold=0.25, min_count=5):
    """
    """
    # final dictionary
    post_processed_if_dict = dict()

    for category, sub_dict in if_average.items():
        for word, value in sub_dict.items():
            if value >= alpha_threshold and if_count[category][word] > min_count and len(word) >= 2:
                is_true = True
                for category_idx in range(len(if_average)):
                    if category_idx != category and category_idx in if_average:
                        # TODO: Relax the beta_threshold between the classes that share similar properties (4 and 5 stars)
                        if abs(category_idx - category) == 1 and len(if_average) > 2:
                            new_beta_threshold = 0.1
                        else:
                            new_beta_threshold = beta_threshold

                        if word not in if_average[category_idx] or value >= if_average[category_idx][word] + new_beta_threshold:
                            continue
                        else:
                            is_true = False
                            break
                if is_true:
                    post_processed_if_dict.setdefault(category, dict())
                    post_processed_if_dict[category][word] = value

    print(f'the length of the interpretation_features dict for all categories: {len(post_processed_if_dict)}')

    return post_processed_if_dict
