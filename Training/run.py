import os
import time
import pickle as pkl
import numpy as np

from tensorflow.keras import models, callbacks, utils

import src.settings as settings
# from src.Training.w2v import train_word2vec
from src.Training.plot import plot_loss_accuracy
from src.Training.data_helpers import train_test_split

np.random.seed(0)

# Loading Dataset
# ----------------------------------------------------------------------------------------------------------------------
print('Loading data...')
start = time.time()

x_train, y_train, l_train, x_test, y_test, l_test, vocabulary = train_test_split(settings.data_source,
                                                                                 settings.test_split,
                                                                                 settings.sequence_length,
                                                                                 max_num_words=settings.max_words)

assert len(x_train) == len(y_train) == len(l_train)
assert len(x_test) == len(y_test) == len(l_test)
print('Dataset has been built successfully.')
print(f'Run time: {round(time.time() - start)} second')
print(f'Number of training samples: {len(x_train)}')
print(f'Number of testing samples: {len(x_test)}')
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print(f'Files have been saved on: {settings.output_directory}')
print(f'Model type is: {settings.model_type}')

# Save Vocabulary, Training, and Testing data
# ----------------------------------------------------------------------------------------------------------------------
# Save vocabulary processor
with open(os.path.join(settings.output_directory, 'vocabulary.pkl'), 'wb') as f:
    pkl.dump(vocabulary, f, pkl.HIGHEST_PROTOCOL)

# Save training and testing data
with open(os.path.join(settings.output_directory, 'x_train.pkl'), 'wb') as f:
    pkl.dump(x_train, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_directory, 'y_train.pkl'), 'wb') as f:
    pkl.dump(y_train, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_directory, 'l_train.pkl'), 'wb') as f:
    pkl.dump(l_train, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_directory, 'x_test.pkl'), 'wb') as f:
    pkl.dump(x_test, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_directory, 'y_test.pkl'), 'wb') as f:
    pkl.dump(y_test, f, pkl.HIGHEST_PROTOCOL)
with open(os.path.join(settings.output_directory, 'l_test.pkl'), 'wb') as f:
    pkl.dump(l_test, f, pkl.HIGHEST_PROTOCOL)

# Load the model
# ----------------------------------------------------------------------------------------------------------------------
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

# Plot the sub_model
if settings.plot_model:
    print(model.summary())
    utils.plot_model(model, to_file=os.path.join(settings.output_directory, 'model.png'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # metrics=['accuracy']

checkpoint = callbacks.ModelCheckpoint(settings.model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')


# # Initialize weights with word2vec
# if settings.model_type == 'non-static':
#     embedding_weights = train_word2vec(np.vstack((x_train, x_test)),
#                                        vocabulary,
#                                        num_features=settings.embedding_dim,
#                                        min_word_count=settings.min_word_count,
#                                        context=settings.context)
#
#     weights = np.array([v for v in embedding_weights.values()])
#     print(f'Initializing embedding layer with word2vec weights, shape {weights.shape}')
#     embedding_layer = model.get_layer('embedding')
#     embedding_layer.set_weights([weights])


# Train the model
# ----------------------------------------------------------------------------------------------------------------------
start_time_training = time.time()

history = model.fit(x_train, y_train,
                    batch_size=settings.batch_size,
                    epochs=settings.num_epochs,
                    validation_split=settings.validation_split,
                    verbose=1,
                    callbacks=[checkpoint])

print(f'Training is over. The model saved to disk. \nThe training took {round(time.time() - start_time_training)} seconds\n')


# Plot the training results
plot_loss_accuracy(history.history, settings.output_directory)


# Evaluate the model
# ----------------------------------------------------------------------------------------------------------------------
new_model = models.load_model(settings.model_path)

results = new_model.evaluate(x_test, y_test, verbose=0)
print(f'\nTesting is over. The classification accuracy: {results[1]} , and loss: {results[0]}')
