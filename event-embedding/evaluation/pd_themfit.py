# coding: utf-8
import os
import sys
import time
import cPickle
import gzip
import random
import pandas as pd
import numpy

from scipy.stats import spearmanr
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

# Configuration of environment
SRC_DIR = os.path.dirname((os.path.dirname(os.path.abspath(__file__))))
sys.path.append(SRC_DIR)

import model_builder
import config
import utils

MODEL_PATH = config.MODEL_VERSION

wnl = WordNetLemmatizer()


def get_filler_prob(inputs, target, model, raw_word_list):
    """ Returns the probability of a target filler for a role, given a set of input roles + fillers
        
    Keyword arguments:
    inputs -- A dictionary of inputs with the role as the key and the filler as the value.
    target -- A singleton dictionary containing the target role as the key and target filler as the value.
    model -- The loaded model with which to make predictions
    raw_word_list -- A dictionary of vocabulary
    """
    #print(inputs)
    raw_word_list.update(inputs)
    #print(raw_word_list)
        
    assert len(raw_word_list) == len(model.role_vocabulary)
        
    t_r = [model.role_vocabulary.get(r, model.unk_role_id) for r in target.keys()]
    t_w = [model.word_vocabulary.get(w, model.unk_word_id) for w in target.values()]
    #print("Target role", t_r)
    #print("Target word", t_w)

    if t_w[0] == model.unk_word_id:
        return None

    input_roles_words = {}
    for r, w in raw_word_list.items():
        input_roles_words[model.role_vocabulary[r]] = utils.input_word_index(model.word_vocabulary, w, model.unk_word_id, warn_unk=False)

    #print input_roles_words, t_r[0]
    input_roles_words.pop(t_r[0])

    x_w_i = numpy.asarray([input_roles_words.values()], dtype=numpy.int64)
    x_r_i = numpy.asarray([input_roles_words.keys()], dtype=numpy.int64)
    y_w_i = numpy.asarray(t_w, dtype=numpy.int64)
    y_r_i = numpy.asarray(t_r, dtype=numpy.int64)

    return model.p_words(x_w_i, x_r_i, y_w_i, y_r_i)[0]

def get_top_predictions(inputs, target, model, raw_word_list, n=5):
    """ Returns the top predicted filler for a target role, given a set of input roles + fillers
        
    Keyword arguments:
    inputs -- A dictionary of inputs with the role as the key and the filler as the value.
    target -- A singleton dictionary containing the target role as the key and target filler as the value.
    model -- The loaded model with which to make predictions
    raw_word_list -- A dictionary of vocabulary
    n -- The number of top predictions that should be retrieved
    """
    #print(inputs)
    raw_word_list.update(inputs)
    #print(raw_word_list)
        
    assert len(raw_word_list) == len(model.role_vocabulary)
        
    t_r = [model.role_vocabulary.get(r, model.unk_role_id) for r in target.keys()]
    t_w = [model.unk_word_id]

    input_roles_words = {}
    for r, w in raw_word_list.items():
        input_roles_words[model.role_vocabulary[r]] = utils.input_word_index(model.word_vocabulary, w, model.unk_word_id, warn_unk=False)

    #print input_roles_words, t_r[0]
    input_roles_words.pop(t_r[0])

    x_w_i = numpy.asarray([input_roles_words.values()], dtype=numpy.int64)
    x_r_i = numpy.asarray([input_roles_words.keys()], dtype=numpy.int64)
    y_w_i = numpy.asarray(t_w, dtype=numpy.int64)
    y_r_i = numpy.asarray(t_r, dtype=numpy.int64)

    predicted_word_indices = model.top_words(x_w_i, x_r_i, y_w_i, y_r_i, n)
    results = []
    reverse_vocabulary = utils.get_reverse_map(model.word_vocabulary)

    for t_w_i in predicted_word_indices:
        t_w = model.word_vocabulary.get(t_w_i, model.unk_word_id)
        y_w_i = numpy.asarray([t_w_i], dtype=numpy.int64)
        p = model.p_words(x_w_i, x_r_i, y_w_i, y_r_i, batch_size=1, verbose=0)[0]
        n = numpy.round(p / 0.005)
        fb = numpy.floor(n)
        hb = n % 2
        lemma = reverse_vocabulary[int(t_w_i)]
        #print u"{:<5} {:7.6f} {:<20} ".format(i+1, float(p), lemma) + u"\u2588" * int(fb) + u"\u258C" * int(hb)
        results.append((lemma, p))

    return results

    
def process_row(predict_role, role_fillers, model, raw_word_list, function="filler_prob", n=5):
    """ Apply get_filler_prob or get_top_predictions to a row in a pandas DF.
        
    Keyword arguments:
    predict_role -- the target role for which the filler will be predicted (default: 'V')
    role_fillers -- a dictionary containing the role labels and role-fillers
    model -- The loaded model with which to make predictions
    raw_words -- A dictionary of vocabulary
    n -- The number of predictions for the "get top predictions" function
    """

    # TARGET GOAL: {"V" : "eat"}
    # INPUTS GOAL: {"A0" : "horse", "A1" : "hay"}

    target = {predict_role : role_fillers[predict_role]}
    role_fillers.pop(predict_role)
    
    if function == 'filler_prob':
        return get_filler_prob(role_fillers, target, model, raw_word_list)
    else:
        return get_top_predictions(role_fillers, target, model, raw_word_list, n)


def pd_themfit(model_name, experiment_name, df, predict_role='V', input_roles="all_args", function="filler_prob", n=5):    
    """ Adds a column to a pandas df with a role filler probability.

    For each row in the pandas df, calculates the probability that a particular role filler will fill a
    particular role, given a set of input roles and fillers (from that row).
        
    Keyword arguments:
    model_name -- The name of the model
    experiment_name -- The name of the model plus the name of the experiment, separated by '_'
    df -- The pandas dataframe. Columns should use dependency labels (nsubj, iobj etc)
    predict_role -- the target role (in propbank labels) for which the filler will be predicted (default: 'V')
    input_roles -- the set of roles (in propbank labels) that should be used as inputs (default: 'all_args')
    """
    
    MODEL_NAME = experiment_name

    description = model_builder.load_description(MODEL_PATH, MODEL_NAME)
    net = model_builder.build_model(model_name, description)
    net.load(MODEL_PATH, MODEL_NAME, description)

    bias = net.set_0_bias()

    # net.model.summary()
    # print net.model.get_layer(name="embedding_2").get_weights()[0]

    # If no <UNKNOWN> role in the role vocabulary, add it.
    if net.role_vocabulary.get("<UNKNOWN>", -1) == -1:
        net.role_vocabulary["<UNKNOWN>"] = len(net.role_vocabulary) - 1

    print("Role vocabulary", net.role_vocabulary)
    print("unk_word_id", net.unk_word_id)
    print("missing_word_id", net.missing_word_id)

    reverse_vocabulary = utils.get_reverse_map(net.word_vocabulary)
    reverse_role_vocabulary = utils.get_reverse_map(net.role_vocabulary)    

    print("Reverse role vocabulary", reverse_role_vocabulary)

    raw_words = dict((reverse_role_vocabulary[r], reverse_vocabulary[net.missing_word_id]) for r in net.role_vocabulary.values())

    if input_roles == 'all_args':
        input_roles = ['A0', 'A1', '<UNKNOWN>', 'V']
        input_roles.remove(predict_role)
  
    def return_non_null_arg(dobj, nsubjpass):
        if pd.isnull(dobj):
            return nsubjpass
        else:
            return dobj

    if ('A1' in input_roles or predict_role == 'A1'):
        if 'nsubjpass' in df.columns:
            df['A1'] = df.apply(lambda x: return_non_null_arg(x['dobj'], x['nsubjpass']),
                          axis=1)
        else:
            df = df.rename(columns={'dobj':'A1'})

    if ('A0' in input_roles) or (predict_role == 'A0'):
        df = df.rename(columns={'nsubj':'A0'})
    if ('<UNKNOWN>' in input_roles) or (predict_role == '<UNKNOWN>'):
        df = df.rename(columns={'iobj':'<UNKNOWN>'})

    all_roles = input_roles
    all_roles.append(predict_role)

    df = df.apply(lambda x: process_row(predict_role = predict_role,
                                        role_fillers = { i : x[i] for i in all_roles},
                                        model = net,
                                        raw_word_list = raw_words,
                                        function = function,
                                        n = n),
                axis = 1)

    return df
