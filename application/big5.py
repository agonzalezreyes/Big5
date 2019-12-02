import numpy as np
import pandas as pd
from collections import defaultdict
import re
import NBClassifier as NBC

# split data
def train_test_split(array, split=0.1):
    indices = np.random.permutation(array.shape[0])
    split = int(split * array.shape[0])
    train_indices = indices[split:]
    test_indices = indices[:split]
    train, test = array[train_indices], array[test_indices]
    return train, test

df = pd.read_csv('mypersonality.csv')
status = 'STATUS'
categories = ['cEXT','cNEU','cAGR','cCON','cOPN']

ext_data = df[[categories[0],status]].values
neu_data = df[[categories[1],status]].values
agr_data = df[[categories[2],status]].values
con_data = df[[categories[3],status]].values
opn_data = df[[categories[4],status]].values

ext_data = np.array(ext_data)
ext_data[:,0][ext_data[:,0] == 'n'] = 0
ext_data[:,0][ext_data[:,0] == 'y'] = 1

neu_data = np.array(neu_data)
neu_data[:,0][neu_data[:,0] == 'n'] = 0
neu_data[:,0][neu_data[:,0] == 'y'] = 1

agr_data = np.array(agr_data)
agr_data[:,0][agr_data[:,0] == 'n'] = 0
agr_data[:,0][agr_data[:,0] == 'y'] = 1

con_data = np.array(con_data)
con_data[:,0][con_data[:,0] == 'n'] = 0
con_data[:,0][con_data[:,0] == 'y'] = 1

opn_data = np.array(opn_data)
opn_data[:,0][opn_data[:,0] == 'n'] = 0
opn_data[:,0][opn_data[:,0] == 'y'] = 1

# ----  init extraversion model ----- #
print("*** init extraversion model ***")
ext_train, ext_test = train_test_split(ext_data)
train_ext_labels = ext_train[:,0] # get labels column of 1s and 0s
train_ext_text = ext_train[:,1] # get text of status
nb_ext = NBC.NBClassifier(np.unique(train_ext_labels)) # init with classes = labels

# ----  init neuroticism model ----- #
print("*** init neuroticism model ***")
neu_train, neu_test = train_test_split(neu_data)
train_neu_labels = neu_train[:,0] # get labels column of 1s and 0s
train_neu_text = neu_train[:,1] # get text of status
nb_neu = NBC.NBClassifier(np.unique(train_neu_labels)) # init with classes = labels

# ----  init agreeableness model ----- #
print("*** init agreeableness model ***")
agr_train, agr_test = train_test_split(agr_data)
train_agr_labels = agr_train[:,0] # get labels column of 1s and 0s
train_agr_text = agr_train[:,1] # get text of status
nb_agr = NBC.NBClassifier(np.unique(train_agr_labels)) # init with classes = labels

# ----  init conscientiousness model ----- #
print("*** init conscientiousness model ***")
con_train, con_test = train_test_split(con_data)
train_con_labels = con_train[:,0] # get labels column of 1s and 0s
train_con_text = con_train[:,1] # get text of status
nb_con = NBC.NBClassifier(np.unique(train_con_labels)) # init with classes = labels

# ----  init openness model ----- #
print("*** init openness model ***")
opn_train, opn_test = train_test_split(opn_data)
train_opn_labels = opn_train[:,0] # get labels column of 1s and 0s
train_opn_text = opn_train[:,1] # get text of status
nb_opn = NBC.NBClassifier(np.unique(train_opn_labels)) # init with classes = labels

def train_models():
    # ---- extraversion model ----- #
    print "Starting extraversion model training..."
    nb_ext.train(train_ext_text, train_ext_labels)
    print "... extraversion training model completed!"
    test_ext_labels = ext_test[:,0] # get labels column of 1s and 0s
    test_ext_text = ext_test[:,1] # get text of status
    p_ext_classes = nb_ext.test(test_ext_text) # predictions for test text
    #check how many predictions actually match original test labels
    ext_accuracy = np.sum(p_ext_classes == test_ext_labels) / float(test_ext_labels.shape[0])
    print "- Extraversion Test Set Examples: " + str(test_ext_labels.shape[0])
    print "- Extraversion Test Set Accuracy: " + str(ext_accuracy*100.0)

    # ---- neuroticism model ----- #
    print "Starting neuroticism model training..."
    nb_neu.train(train_neu_text, train_neu_labels)
    print "... neuroticism training model completed!"
    test_neu_labels = neu_test[:,0] # get labels column of 1s and 0s
    test_neu_text = neu_test[:,1] # get text of status
    p_neu_classes = nb_neu.test(test_neu_text) # predictions for test text
    #check how many predictions actually match original test labels
    neu_accuracy = np.sum(p_neu_classes == test_neu_labels) / float(test_neu_labels.shape[0])
    print "- Neuroticism Test Set Examples: " + str(test_neu_labels.shape[0])
    print "- Neuroticism Test Set Accuracy: " + str(neu_accuracy*100.0)

    # ---- agreableness model ----- #
    print "Starting agreableness model training..."
    nb_agr.train(train_agr_text, train_agr_labels)
    print "... agreableness training model completed!"
    test_agr_labels = agr_test[:,0] # get labels column of 1s and 0s
    test_agr_text = agr_test[:,1] # get text of status
    p_agr_classes = nb_agr.test(test_agr_text) # predictions for test text
    #check how many predictions actually match original test labels
    agr_accuracy = np.sum(p_agr_classes == test_agr_labels) / float(test_agr_labels.shape[0])
    print "- Agreableness Test Set Examples: " + str(test_agr_labels.shape[0])
    print "- Agreableness Test Set Accuracy: " + str(agr_accuracy*100.0)

    # ---- conscientiousness model ----- #
    print "Starting conscientiousness model training..."
    nb_con.train(train_con_text, train_con_labels)
    print "... conscientiousness training model completed!"
    test_con_labels = con_test[:,0] # get labels column of 1s and 0s
    test_con_text = con_test[:,1] # get text of status
    p_con_classes = nb_con.test(test_con_text) # predictions for test text
    #check how many predictions actually match original test labels
    con_accuracy = np.sum(p_con_classes == test_con_labels) / float(test_con_labels.shape[0])
    print "- Conscientiousness Test Set Examples: " + str(test_con_labels.shape[0])
    print "- Conscientiousness Test Set Accuracy: " + str(con_accuracy*100.0)

    # ---- openness model ----- #
    print "Starting openness model training..."
    nb_opn.train(train_opn_text, train_opn_labels)
    print "... openness training model completed!"
    test_opn_labels = opn_test[:,0] # get labels column of 1s and 0s
    test_opn_text = opn_test[:,1] # get text of status
    p_opn_classes = nb_opn.test(test_opn_text) # predictions for test text
    #check how many predictions actually match original test labels
    opn_accuracy = np.sum(p_opn_classes == test_opn_labels) / float(test_opn_labels.shape[0])
    print "- Openness Test Set Examples: " + str(test_opn_labels.shape[0])
    print "- Openness Test Set Accuracy: " + str(opn_accuracy*100.0)

def check_ext(tweets):
    p_class = nb_ext.test(np.array(tweets))
    return np.mean(p_class)

def check_neu(tweets):
    p_class = nb_neu.test(np.array(tweets))
    return np.mean(p_class)

def check_agr(tweets):
    p_class = nb_agr.test(np.array(tweets))
    return np.mean(p_class)

def check_con(tweets):
    p_class = nb_con.test(np.array(tweets))
    return np.mean(p_class)

def check_opn(tweets):
    p_class = nb_opn.test(np.array(tweets))
    return np.mean(p_class)

def check_personality(tweets):
    E = round(check_ext(tweets),4)*100.0
    N = round(check_neu(tweets),4)*100.0
    A = round(check_agr(tweets),4)*100.0
    C = round(check_con(tweets),4)*100.0
    O = round(check_opn(tweets),4)*100.0
    dict = {'E': E,'N': N,'A': A,'C': C,'O': O}
    return dict
