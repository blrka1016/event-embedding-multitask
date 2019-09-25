# coding: utf-8
import os
import sys
import time
import cPickle
import gzip
import random

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


def get_filler_prob(inputs, target, model, raw_words):
    """ Returns the probability of a target filler for a role, given a set of input roles + fillers
        
    Keyword arguments:
    inputs -- A dictionary of inputs with the role as the key and the filler as the value.
    target -- A singleton dictionary containing the target role as the key and target filler as the value.
    model -- The loaded model with which to make predictions
    raw_words -- A dictionary of vocabulary
    """
    raw_words.update(inputs)
        
    assert len(raw_words) == len(model.role_vocabulary)
        
    t_r = [model.role_vocabulary.get(r, model.unk_role_id) for r in target.keys()]
    t_w = [model.word_vocabulary.get(w, model.unk_word_id) for w in target.values()]

    if t_w[0] == model.unk_word_id:
        return None

    input_roles_words = {}
    for r, w in raw_words.items():
        input_roles_words[model.role_vocabulary[r]] = utils.input_word_index(model.word_vocabulary, w, model.unk_word_id, warn_unk=True)

        print input_roles_words, t_r
        input_roles_words.pop(t_r[0])

    x_w_i = numpy.asarray([input_roles_words.values()], dtype=numpy.int64)
    x_r_i = numpy.asarray([input_roles_words.keys()], dtype=numpy.int64)
    y_w_i = numpy.asarray(t_w, dtype=numpy.int64)
    y_r_i = numpy.asarray(t_r, dtype=numpy.int64)

    return model.p_words(x_w_i, x_r_i, y_w_i, y_r_i)

def process_row(predict_role, input_roles, predicate_lemma, nsubj, dobj, iobj, nsubjpass, model, raw_words):
    """ Apply get_filler_prob to a row in a pandas DF.
        
    Keyword arguments:
    predict_role -- the target role for which the filler will be predicted (default: 'V')
    input_roles -- the set of roles that should be used as inputs (default: 'all_args')
    predicate_lemma -- the verb lemma 
    nsubj -- the non-passive subject
    dobj -- the direct object
    iobj -- the indirect object
    nsubjpass -- the passive subject
    model -- The loaded model with which to make predictions
    raw_words -- A dictionary of vocabulary
    """
    ud_map = {
        "V" : "V",
        "nsubj" : "A0",
        "nsubjpass" : "A0",
        "obj" : "A1",
        "iobj" : "A2"
    }

    role_map = {
        "V" : predicate_lemma,
        "nsubj" : nsubj,
        "dobj" : dobj,
        "iobj" : iobj,
        "nsubjpass" : nsubjpass
    }

    # TARGET GOAL: {"V" : "eat"}
    # INPUTS GOAL: {"A0" : "horse", "A1" : "hay"}

    target = {ud_map[predict_role] : role_map[predict_role]}
    inputs = {}
    
    for r in input_roles:
        role = ud_map[r]
        filler = role_map[r]
        if filler is not None:
            inputs[role] = filler

    return get_filler_prob(inputs, target, model, raw_words)

    

def pd_themfit(model_name, experiment_name, df, predict_role='V', input_roles="all_args"):    
    """ Adds a column to a pandas df with a role filler probability.

    For each row in the pandas df, calculates the probability that a particular role filler will fill a
    particular role, given a set of input roles and fillers (from that row).
        
    Keyword arguments:
    model_name -- The name of the model
    experiment_name -- The name of the model plus the name of the experiment, separated by '_'
    df -- The pandas dataframe
    predict_role -- the target role for which the filler will be predicted (default: 'V')
    input_roles -- the set of roles that should be used as inputs (default: 'all_args')
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

    print net.role_vocabulary
    print("unk_word_id", net.unk_word_id)
    print("missing_word_id", net.missing_word_id)

    reverse_vocabulary = utils.get_reverse_map(net.word_vocabulary)
    reverse_role_vocabulary = utils.get_reverse_map(net.role_vocabulary)    

    print reverse_role_vocabulary

    raw_words = dict((reverse_role_vocabulary[r], reverse_vocabulary[net.missing_word_id]) for r in net.role_vocabulary.values())

    df = df.apply(lambda x: process_row(predict_role,
                                        input_roles,
                                        x['Pred.Lemma'],
                                        x['nsubj'],
                                        x['dobj'],
                                        x['iobj'],
                                        x['nsubjpass'],
                                        net,
                                        raw_words),
                axis = 1)

    return df
