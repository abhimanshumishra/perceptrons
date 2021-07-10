import os
import sys
import collections
import re
import math
import copy
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize

labels = ['spam', 'ham']
ps = PorterStemmer()

class Doc(object):
    text = ""
    freq_table = {}
    true_class = ""
    predicted_class = ""

    def __init__(self, text, freq_table, true_class):
        self.text = text
        self.freq_table = freq_table
        self.true_class = true_class

def frequencies(text):
    return dict(collections.Counter(re.findall(r'\w+', text)))

def stopwords(stop_file_name):
    fp = open(stop_file_name, 'r')
    stopwords = fp.readlines()
    fp.close()
    return stopwords

def filter_stopwords(data, stopwords):
    filtered_data = copy.deepcopy(data)
    for i in stopwords:
        for j in filtered_data:
            if i in filtered_data[j].freq_table:
                del filtered_data[j].freq_table[i]
    return filtered_data

def vocabulary(data):
    text = ""
    vocab = []
    for i in data:
        text += data[i].text
    for i in frequencies(text):
        vocab.append(i)
    return vocab

def stem(text):
    ret = ''
    for line in text:
        words = word_tokenize(line.lower())
        for w in words:
            ret += ps.stem(w) + ' '
    return ret

def no_stem(text):
    ret = ''
    for line in text:
        words = word_tokenize(line.lower())
        for w in words:
            ret += w + ' '
    return ret

def preprocess(dir_path, to_stem=True):
    data = {}
    for label in os.listdir(dir_path):
        sub_dir_path = os.path.join(dir_path, label)
        if os.path.isdir(sub_dir_path):
            for fname in os.listdir(sub_dir_path):
                fpath = os.path.join(sub_dir_path, fname)
                if os.path.isfile(fpath):
                    fp = open(fpath, 'r', encoding='latin-1')
                    original_text = fp.readlines()
                    if to_stem:
                        text = stem(original_text)
                    else:
                        text = no_stem(original_text)
                    data.update({fpath: Doc(text, frequencies(text), label)})
    return data

def train(data):
    conditional_probabilities = {}
    priors = {}
    vocab = vocabulary(data)
    no_of_docs = len(data)
    for label in labels:
        docs_in_label = 0
        text_in_label = ""
        for i in data:
            if data[i].true_class == label:
                docs_in_label += 1
                text_in_label += data[i].text
        priors[label] = float(docs_in_label)/float(no_of_docs)
        token_frequencies = frequencies(text_in_label)
        for term in vocab:
            if term in token_frequencies:
                conditional_probabilities.update({label + '_' + term: (float((token_frequencies[term] + 1.0)) / float((len(text_in_label) + len(token_frequencies))))})
            else:
                conditional_probabilities.update({label + '_' + term: (float(1.0) / float((len(text_in_label) + len(token_frequencies))))})
    return conditional_probabilities, priors

def predict(sentence, conditional_probabilities, priors):
    score = {}
    for label in labels:
        score[label] = math.log(float(priors[label]))
        for term in sentence.freq_table:
            if (label + '_' + term) in conditional_probabilities:
                score[label] += float(math.log(conditional_probabilities[label + '_' + term]))
    if score['ham'] > score['spam']:
        return 'ham'
    else:
        return 'spam'

if __name__=='__main__':
    train_folder = sys.argv[1]
    test_folder = sys.argv[2]
    remove_stop = sys.argv[3]
    try:
        to_stem = sys.argv[4]
    except:
        to_stem = 'yes'

    if to_stem.lower() == 'yes':
        train_data = preprocess(train_folder)
        test_data = preprocess(test_folder)
    elif to_stem.lower() == 'no':
        train_data = preprocess(train_folder, False)
        test_data = preprocess(test_folder, False)
    else:
        print('Invalid to_stem argument. Possible values are: yes/no')
        sys.exit(0)

    if remove_stop.lower() == 'no':
        conditional_probabilities, priors = train(train_data)

        correct = 0

        for i in test_data:
            prediction = predict(test_data[i], conditional_probabilities, priors)
            if prediction == test_data[i].true_class:
                correct += 1
        accuracy = 100*correct/len(test_data)
        accuracy = round(accuracy, 3)
        print(f'Accuracy of Naive Bayes on test set without stopword removal with {to_stem} stemming is: {accuracy}')

    elif remove_stop.lower() == 'yes':
        try:
            stop_name = sys.argv[5]
        except:
            stop_name = 'stopwords.txt'
        stops = stopwords(stop_name)

        filtered_train_data = filter_stopwords(train_data, stops)
        filtered_test_data = filter_stopwords(test_data, stops)

        filtered_conditional_probabilities, filtered_priors = train(filtered_train_data)

        correct = 0

        for i in filtered_test_data:
            prediction = predict(filtered_test_data[i], filtered_conditional_probabilities, filtered_priors)
            if prediction == filtered_test_data[i].true_class:
                correct += 1
        accuracy = 100*correct/len(filtered_test_data)
        accuracy = round(accuracy, 3)
        print(f'Accuracy of Naive Bayes on test set after removing stopwords from {stop_name} with {to_stem} stemming file is: {accuracy}')

    else:
        print('Invalid stopword removal argument. Possible values are: yes/no')


