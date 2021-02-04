from tensorflow.keras import layers, backend as K, regularizers, models


def h_attention_classifier(num_classes, dropout, embedding_dim, rnn_dim, sequence_length, fully_connected_dim, vocabulary_inv):
    """
    """
    input_shape = (sequence_length,)
    # shape=(None, sequence_length)
    model_input = layers.Input(shape=input_shape, name='input')

    # shape=(None, sequence_length, embedding_dim)
    embedded_sequences = layers.Embedding(len(vocabulary_inv) + 1, embedding_dim, input_length=sequence_length, trainable=True, name='embedding')(model_input)

    # shape=(None, sequence_length, 2 * embedding_dim)
    lstm_sentence = layers.Bidirectional(layers.LSTM(rnn_dim, return_sequences=True, recurrent_dropout=dropout, kernel_regularizer=regularizers.l2()))(embedded_sequences)

    # shape=(None, sequence_length, embedding_dim)
    dense_sentence = layers.Dense(rnn_dim, activation='tanh')(lstm_sentence)

    # Attention Layer
    # shape=(None, sequence_length, 1)
    attention = layers.Dense(1, activation='tanh')(lstm_sentence)

    # shape=(None, sequence_length)
    attention = layers.Flatten()(attention)
    # shape=(None, sequence_length)
    scores = layers.Activation('softmax', name='scores')(attention)
    # shape=(None, embedding_dim, sequence_length)
    attention = layers.RepeatVector(rnn_dim)(scores)
    # shape=(None, sequence_length, embedding_dim)
    attention = layers.Permute([2, 1])(attention)
    # shape=(None, sequence_length, embedding_dim)
    sent_representation = layers.multiply([dense_sentence, attention])
    # shape=(None, embedding_dim)
    context_vector = layers.Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(rnn_dim,))(sent_representation)

    # shape=(None, num_classes)
    logits = layers.Dense(num_classes, activation='linear', name='logits')(context_vector)
    model_output = layers.Activation('softmax', name='predictions')(logits)

    model = models.Model(model_input, model_output)

    return model
