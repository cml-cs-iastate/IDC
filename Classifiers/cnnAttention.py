from tensorflow.keras import models, layers


def attention_classifier(num_classes, dropout, embedding_dim, num_filters, filter_sizes, sequence_length, fully_connected_dim, vocabulary_inv, combined_loss=False):
    """
    CNN classification model for sentiment analysis based on "Convolutional Neural Networks for Sentence Classification"
     by Yoon Kim
    """

    input_shape = (sequence_length,)
    model_input = layers.Input(shape=input_shape, name='input')

    x = layers.Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name='embedding')(model_input)

    x = layers.Dropout(dropout)(x)

    # Convolutional block
    conv_blocks = []

    for sz in filter_sizes:
        # padding='same' is better but we are not sure if the padding is only going to be at the end of the text
        conv = layers.Convolution1D(filters=num_filters, kernel_size=sz, padding='valid', activation='relu', strides=1)(x)
        conv = layers.MaxPooling1D(pool_size=sequence_length - sz + 1)(conv)
        conv = layers.Flatten()(conv)
        conv_blocks.append(conv)

    # Concatenate the convolutional layers
    x = layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    # Attention block
    units = x.shape[-1]
    score = layers.Dense(units, activation='relu')(x)
    attention_weights = layers.Activation('softmax')(score)
    context_vector = layers.multiply([x, attention_weights])

    x = layers.Dropout(dropout)(context_vector)
    x = layers.Dense(fully_connected_dim, activation='relu')(x)

    logits = layers.Dense(num_classes, activation='linear', name='logits')(x)
    model_output = layers.Activation('softmax', name='predictions')(logits)

    model = models.Model(model_input, model_output)

    if combined_loss:
        return model, model_input, logits, conv_blocks

    return model
