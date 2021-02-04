import re
import csv
import collections
import numpy as np
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer


# Tested
def clean_str(string, remove_stopword, sequence_length):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    if remove_stopword:
        stop_words = {'ourselves', 'hers', 'between', 'yourself', 'there', 'about', 'having', 'with', 'they', 'an',
                      'be', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'is', 's', 'am',
                      'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'are', 'we', 'these',
                      'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'himself', 'this', 'our', 'their',
                      'while', 'both', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'them', 'and',
                      'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what',
                      'over', 'why', 'did', 'now', 'he', 'you', 'herself', 'has', 'where', 'myself', 'which', 'those',
                      'i', 'whom', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was',
                      'here', 'than', "'s"}

        if sequence_length > 0:
            string = ' '.join([w for w in string.split(' ') if w not in stop_words][:sequence_length + 1])
        else:
            string = ' '.join([w for w in string.split(' ') if w not in stop_words])

    return string


# Tested
def load_data(data_source, sequence_length=0):
    """
    Loads data from files as sentences for each document
    :param data_source: string path to the input dataset csv file
    :param sequence_length: the maximum length of the documents we need to read
    output:
    x_text: list of the text documents              # shape [strings - number of docs]
    y_text: list of the labels                      # shape [int - number of labels]
    """
    # Read the CSV file and get its contents
    with open(data_source, 'r', encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.reader(f)
        # get the header
        header = next(csv_reader)
        label_idx = header.index('label')
        content_idx = header.index('content')
        print(f'The label index is : {label_idx} and the content index is : {content_idx}')

        x_text = list()
        y_text = list()

        for line in csv_reader:
            # get the sentence from the line
            sentence = line[content_idx].strip()
            sentence = clean_str(sentence, True, sequence_length=sequence_length)
            x_text.append(sentence)
            y_text.append(int(line[label_idx]))

    return x_text, y_text


# Tested
def get_tfidf(x_text):
    """
    :param x_text: list of the text documents              # shape [strings - number of docs]
    output:
    vocab_tfidf: numpy.array contains dataset vocabulary of string type
    data_tfidf: dict {doc_idx: {word_idx: tfidf_value}}
    """
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf = vectorizer.fit_transform(x_text)
    print('vectorizer.fit_transform() is over!')

    # Get the vocabulary list
    vocab_tfidf = np.array(vectorizer.get_feature_names())
    print('vectorizer.get_feature_names() is over!')

    # Get the tfidf values for each word in the documents (ordered based on the vocabulary list)
    # We store the output as sparse matrix to save memory space. numpy array will require a lot of space
    data_tfidf = dict()
    for doc_idx, doc in enumerate(tfidf):
        data_tfidf[doc_idx] = dict()
        for word_idx, word_tfidf in enumerate(doc.toarray()[0]):
            if word_tfidf > 0:
                data_tfidf[doc_idx][word_idx] = word_tfidf

    return vocab_tfidf, data_tfidf


# Tested
def post_processing(vocab, data, labels, vocab_size=0, min_num_words=10, get_avg=False):
    """
    :param vocab: numpy.array contains dataset vocabulary of string type
    :param data: dict {doc_idx: {word_idx: word_tfidf_value}}
    :param labels: list of the labels [int - number of labels]
    :param vocab_size: the maximum number of vocabulary int
    :param min_num_words: the minimum number of words per document
    :param get_avg: calculate the average of tfidf, otherwise calculate the sum
    """
    vocab_length = len(vocab)

    # Calculate the Sum/Avg of the tfidf for each word
    tfidf_sum = np.zeros(shape=vocab_length)
    count = np.zeros(shape=vocab_length)
    for doc_idx, doc in data.items():
        for word_idx, word in doc.items():
            tfidf_sum[word_idx] += word
            count[word_idx] += 1

    if get_avg:
        for word_idx in range(vocab_length):
            if tfidf_sum[word_idx] > 0:
                tfidf_sum[word_idx] = tfidf_sum[word_idx] / count[word_idx]

    # Get the index of the data_tfidf_sum based on the Sum/Avg sorted values (Descending)
    argsort = tfidf_sum.argsort()[::-1]
    # Resort vocab based on argsort
    vocab = vocab[argsort][:vocab_size]

    # Generate new vocab based on the new sorted index
    argsort_idx = {v: i for i, v in enumerate(argsort[:vocab_size])}

    # Resort data based on argsort
    not_included_doc = 0
    new_data = collections.defaultdict(list)
    for doc_idx, doc in data.items():
        temp = [0] * vocab_size
        for word_idx, word in doc.items():
            if word_idx in argsort_idx:
                temp[argsort_idx[word_idx]] = 1  # word  # Use the "word" if we need to store the tfidf instead of binary
        if sum(temp) > min_num_words:
            new_data[labels[doc_idx]].append(temp)
        else:
            not_included_doc += 1

    print(f'The number of not included documents is {not_included_doc}')

    for label, docs in new_data.items():
        print(f'The number of documents of class {label} is : {len(docs)}')

    return vocab, new_data


# ======================================================================================================================
if __name__ == '__main__':

    save_files_ = True
    data_source_ = '../data/Interagreement/Amazon_250_review.csv'  # '../data/MR/MR_Polarity.csv'  # '../data/AmazonYelpCombined/folds/fold_14.csv'
    sequence_length_ = 100
    max_words_ = 3000  # 5000
    min_num_words_ = 1

    # ------------------------------------------------------------------------------------------------------------------
    # Load the data from csv file
    x_text_, y_text_ = load_data(data_source_, sequence_length=sequence_length_)
    print('Loading data is over!')

    # ------------------------------------------------------------------------------------------------------------------
    # Get TF-IDF
    vocab_, data_ = get_tfidf(x_text_)
    print('Building vocabulary and data is over!')
    # Clean up the space
    x_text_ = None

    # ------------------------------------------------------------------------------------------------------------------
    # Post-process results
    vocab_, data_ = post_processing(vocab_, data_, y_text_, vocab_size=max_words_, min_num_words=min_num_words_)
    print('Post-processing is over!')

    print(f'the length of the data_ dict = {len(data_)}')
    for k, v in data_.items():
        print(f'class: {k}, with length {len(v)}')

    # ------------------------------------------------------------------------------------------------------------------
    # Save data and vocab after post-processing
    if save_files_:
        with open('vocabulary_.pkl', 'wb') as f:
            pkl.dump(vocab_, f, pkl.HIGHEST_PROTOCOL)
        with open('data_.pkl', 'wb') as f:
            pkl.dump(data_, f, pkl.HIGHEST_PROTOCOL)
        print('Saving post-processed vocabulary and data is over!')

    # ------------------------------------------------------------------------------------------------------------------
    # TODO: Debugging
    # arr1 = np.array([5, 4, 1, 2, 3])
    # arr2 = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15], [10, 20, 30, 40, 50]])
    # print(np.shape(arr1))
    # print(np.shape(arr2))
    # argsort = arr1.argsort()[::-1]
    # print(argsort)
    # print(arr1)
    # print(arr1[argsort])
    # print(arr2)
    # print(arr2[:, argsort])


# # Tested
# def get_tfidf(data):
#     """
#     """
#     vectorizer = TfidfVectorizer(use_idf=True)
#     tfidf = vectorizer.fit_transform(data)
#
#     # Get the vocabulary list
#     vocab_tfidf = np.array(vectorizer.get_feature_names())
#
#     # Get the tfidf values for each word in the documents (ordered based on the vocabulary list)
#     data_tfidf = tfidf.toarray()
#
#
# # Tested
# def post_processing(vocab, data, num_of_vocab=0):
#     """
#     """
#     # Calculate the Sum/Avg of the tfidf for each word
#     data_tfidf_sum = np.sum(data, axis=0)
#
#     # Get the index of the data_tfidf_sum based on the Sum/Avg sorted values (Descending)
#     argsort = data_tfidf_sum.argsort()[::-1]
#
#     # Resort vocab and data based on data_argsort
#     vocab = vocab[argsort]
#     data = data[:, argsort]
#     # The following line also works
#     # data = data.T[argsort].T
#
#     if num_of_vocab > 0:
#         vocab = vocab[:num_of_vocab + 1]
#         data = data[:, :num_of_vocab + 1]
#
#     return vocab, data
