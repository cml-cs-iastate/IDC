import os
import logging

from src.Classifiers.cnn import cnn_classifier
from src.Classifiers.cnnAttention import attention_classifier
from src.Classifiers.hAttention import h_attention_classifier

# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = 'rand'  # {rand, non-static} rand means start the embedding layer randomly, non-static means start with the word2vec embedding

# Model
model = 'h_attention_classifier'  # {'CNN', 'Attention', 'h_attention_classifier'}
if model == 'CNN':
    classifier = cnn_classifier
elif model == 'Attention':
    classifier = attention_classifier
elif model == 'h_attention_classifier':
    classifier = h_attention_classifier

# Data source
dataset = 'MR'  # {'Amazon', 'Yelp', 'MR', 'IMDB', 'AmazonYelpCombined', 'AmazonYelp_10K', 'Interagreement'}
data_source = '../data/MR/MR_Polarity.csv'  # {'../data/MR/MR_Polarity.csv', '../data/IMDB/IMDB_Dataset.csv', '../data/AmazonYelp_10K.csv', '../data/Interagreement/Amazon_250_review.csv'}
validation_split = 0.2
test_split = 0.2

# Model path
model_name = 'cnn_model.h5'
output_directory = os.path.join('../Output', model, dataset)
model_path = os.path.join(output_directory, model_name)
plot_model = False

if not os.path.exists(output_directory):
    os.makedirs(output_directory)


# Model Hyper-parameters
embedding_dim = 128  # 256 (For Amazon and Yelp)
filter_sizes = (3, 4, 5)
target_layer_name = ('conv1d', 'conv1d_1', 'conv1d_2')
num_filters = 50  # 100 (For Amazon and Yelp)
dropout = 0.1
fully_connected_dim = 50  # 100 (For Amazon and Yelp)


# Training Hyper-parameters
batch_size = 32
num_epochs = 20

model_name = 'base_model'
if dataset == 'Amazon' or dataset == 'Yelp':
    num_classes = 6  # 5 stars +1 for class 0
elif dataset == 'MR' or dataset == 'IMDB' or dataset == 'Interagreement':
    num_classes = 2
else:
    num_classes = 6


# Prepossessing parameters
if dataset == 'MR':
    sequence_length = 60
elif dataset == 'IMDB':
    sequence_length = 200
else:
    sequence_length = 200
max_words = 5000  # 30000  (For Amazon and Yelp)  # 5000 (For MR)


# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10


# Interpretation features dicts
if not os.path.exists(os.path.join(output_directory, 'interpretation_features_dicts')):
    os.makedirs(os.path.join(output_directory, 'interpretation_features_dicts'))

post_processed_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'post_processed_IF.pkl')
IF = os.path.join(output_directory, 'interpretation_features_dicts', 'IF.pkl')
count_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'count_IF.pkl')
average_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'average_IF.pkl')

milp_IF = os.path.join(output_directory, 'interpretation_features_dicts', 'MILP/milp_if.pkl')
milp_vocab = os.path.join(output_directory, 'interpretation_features_dicts', 'milp_vocab.pkl')


# Logging settings
log_file_name = os.path.join(output_directory, 'log')
logging.basicConfig(filename=log_file_name, filemode='a', format='%(asctime)s - %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)
logging.info(f'dataset: {dataset}')
logging.info(f'data_source: {data_source}')
logging.info(f'validation_split: {validation_split}')
logging.info(f'test_split: {test_split}')
logging.info(f'model_name: {model_name}')
logging.info(f'output_directory: {output_directory}')
logging.info(f'model_path: {model_path}')
logging.info(f'log_file_name: {log_file_name}')
logging.info(f'plot_model: {plot_model}')
logging.info(f'embedding_dim: {embedding_dim}')
logging.info(f'filter_sizes: {filter_sizes}')
logging.info(f'target_layer_name: {target_layer_name}')
logging.info(f'num_filters: {num_filters}')
logging.info(f'dropout: {dropout}')
logging.info(f'fully_connected_dim: {fully_connected_dim}')
logging.info(f'batch_size: {batch_size}')
logging.info(f'num_epochs: {num_epochs}')
logging.info(f'model_name: {model_name}')
logging.info(f'num_classes: {num_classes}')
logging.info(f'sequence_length: {sequence_length}')
logging.info(f'max_words: {max_words}')
logging.info(f'min_word_count: {min_word_count}')
logging.info(f'context: {context}')
# logger = logging.getLogger('urbanGUI')
