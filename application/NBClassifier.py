import numpy as np
import pandas as pd
from collections import defaultdict
import re

def _preprocess_string(str_arg):
    cleaned_str=re.sub('[^a-z\s]+',' ',str_arg,flags=re.IGNORECASE) #every char except alphabets is replaced
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) #multiple spaces are replaced by single space
    cleaned_str=cleaned_str.lower() #converting the cleaned string to lower case
    return cleaned_str # returning the preprocessed string

class NBClassifier:
    def __init__(self, unique_classes):
        self.classes = unique_classes

    # supplements the training function
    def add_to_bow(self, example, dict_index):
        if isinstance(example, np.ndarray):
            example = example[0]
        for token_word in example.split(): # for every word in preprocessed example
            self.bow_dicts[dict_index][token_word] += 1 #increment in its count

    def train(self, dataset, labels):
        self.examples = dataset
        self.labels = labels
        self.bow_dicts = np.array([defaultdict(lambda:0) for i in range(self.classes.shape[0])])

        if not isinstance(self.examples, np.ndarray):
            self.examples = np.array(self.examples)
        if not isinstance(self.labels, np.ndarray):
            self.labels = np.array(self.labels)

        for cat_index, cat in enumerate(self.classes):
            all_cat_examples = self.examples[self.labels==cat] # filter all examples of category that is cat
            cleaned_examples = [_preprocess_string(cat_example) for cat_example in all_cat_examples]
            cleaned_examples = pd.DataFrame(data=cleaned_examples)
            np.apply_along_axis(self.add_to_bow, 1, cleaned_examples, cat_index)

        prob_classes = np.empty(self.classes.shape[0])
        all_words = []
        cat_word_counts = np.empty(self.classes.shape[0])
        for cat_index, cat in enumerate(self.classes):
            # prior prob for each class
            prob_classes[cat_index] = np.sum(self.labels == cat)/float(self.labels.shape[0])
            # total counts of all the words in each class
            count = list(self.bow_dicts[cat_index].values())
            cat_word_counts[cat_index] = np.sum(np.array(list(self.bow_dicts[cat_index].values()))) + 1
            # words of category
            all_words += self.bow_dicts[cat_index].keys()

        self.vocab = np.unique(np.array(all_words))
        self.vocab_lenght = self.vocab.shape[0]

        denominators = np.array([cat_word_counts[cat_index] + self.vocab_lenght + 1 for cat_index, cat in enumerate(self.classes)])

        self.cats_info = [(self.bow_dicts[cat_index], prob_classes[cat_index], denominators[cat_index]) for cat_index, cat in enumerate(self.classes)]
        self.cats_info = np.array(self.cats_info)

    # estimate posterior prob of given test example
    # returns prob of test example in all classes
    def get_example_prob(self, test_example):
        likelihood_probability = np.zeros(self.classes.shape[0]) # store prob wrt each class
        # finding prob wrt each class of the test example
        for i, cat in enumerate(self.classes):
            for test_token in test_example.split(): # split example to get prob of each word
                test_token_counts = self.cats_info[i][0].get(test_token, 0) + 1
                #print test_token_counts/float(self.cats_info[i][2])
                test_token_prob = test_token_counts/float(self.cats_info[i][2])
                likelihood_probability[i] += np.log(test_token_prob)
        posterior_prob = np.empty(self.classes.shape[0])
        for i, cat in enumerate(self.classes):
            posterior_prob[i] = likelihood_probability[i] + np.log(self.cats_info[i][1])
        return posterior_prob

    # param: test_set of shape (m,)
    # determines the prob of each test example against all classes and predicts the label agaisnt which the class probability is max
    # returns predictions of test example, single prediction againts every test example
    def test(self, test_set):
        predictions = []
        for example in test_set:
            cleaned_ex = _preprocess_string(example)
            post_prob = self.get_example_prob(cleaned_ex)
            # print np.argmax(post_prob)
            predictions.append(self.classes[np.argmax(post_prob)])
        return np.array(predictions)
