import os
import sys
import collections
import re
import math
import copy
import string
import random

def stopwords(stop_file_name):
    # returns dictionary of stopwords all with value 1
    # this is just so it is easy to lookup since dictionary lookup time using a key is O(1)
    fp = open(stop_file_name, 'r')
    stops = fp.readlines()
    stopwords = {}
    for stopword in stops:
        stopwords[stopword] = 1
    fp.close()
    return stopwords

def build_vocabulary(dir_path, stop_file_name):
    translator=str.maketrans('','',string.punctuation)
    if stop_file_name == None:
        stops = {}
    else:
        stops = stopwords(stop_file_name)
    vocabulary = set()
    spam_doc_list = []
    ham_doc_list = []
    for label in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, label)
        if os.path.isdir(sub_dir_path):
            for fname in os.listdir(sub_dir_path):
                fpath = os.path.join(sub_dir_path, fname)
                if os.path.isfile(fpath):
                    data = collections.Counter()
                    fp = open(fpath, 'r', encoding='latin-1')
                    lines = fp.readlines()
                    for line in lines:
                        line = line.translate(translator)
                        for word in line.strip().split():
                            word = word.lower()
                            try:
                                # check if word is in stopword dictionary
                                x = stops[word]
                            except:
                                # add if it isn't in dictionary
                                vocabulary.add(word)
                                try:
                                    data[word] += 1
                                except:
                                    data[word] = 1
                    if label == 'spam':
                        spam_doc_list.append((data, 0))
                    else:
                        ham_doc_list.append((data, 1))
    return list(vocabulary), spam_doc_list, ham_doc_list

def initialize_weights(vocabulary):
    weights = {}
    for word in vocabulary:
        # sample every weight from a uniform distribution with values from 0 to 1
        weights[word] = random.uniform(0,1)
    return weights

def train(training_data, learning_rate, epochs, stop_file_name):
    vocabulary, spam_doc_list, ham_doc_list = build_vocabulary(training_data, stop_file_name)
    weights = initialize_weights(vocabulary)
    all_train_data = ham_doc_list + spam_doc_list
    random.shuffle(all_train_data)
    for i in range(int(epochs)):
        # spam class label is 0, ham class label is 1
        for item in all_train_data:
            data = item[0]
            class_label = item[1]
            prediction = predict(data, weights)
            for word in list(data.keys()):
                weights[word] += float(learning_rate) * (class_label - prediction) * data[word]
    return weights

def predict(data, weights):
    score = 0
    for word in list(data.keys()):
        try:
            # score of a sentence is sum of scores of words multiplied by their weights
            # try-except block to handle out of vocabulary words in test set
            score += (weights[word] * data[word])
        except:
            score += 0
    if score >= 0:
        return 1
    return 0

def test(test_data, weights):
    # discard vocabulary built from test data
    _, spam_doc_list, ham_doc_list = build_vocabulary(test_data, None)
    total_docs = len(spam_doc_list) + len(ham_doc_list)
    all_test_data = ham_doc_list + spam_doc_list
    correct = 0
    for item in all_test_data:
        data = item[0]
        class_label = item[1]
        prediction = predict(data, weights)
        if prediction == class_label:
            correct += 1
    accuracy = correct*100/total_docs
    return round(accuracy, 3)

if __name__=='__main__':
    train_folder = sys.argv[1]
    test_folder = sys.argv[2]
    learning_rate = sys.argv[3]
    epochs = sys.argv[4]
    remove_stop = sys.argv[5]

    if remove_stop.lower() == 'yes':
        try:
            stop_name = sys.argv[6]
        except:
            stop_name = 'stopwords.txt'

    else:
        stop_name = None

    weights = train(train_folder, learning_rate, epochs, stop_name)
    accuracy = test(test_folder, weights)
    print(f'Accuracy of perceptron with learning rate {learning_rate} and {epochs} epochs on test set after removing stopwords from {stop_name} file is: {accuracy}')


