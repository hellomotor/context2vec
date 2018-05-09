#!/usr/bin/env python
from __future__ import print_function

import argparse
import re
import numpy as np

from context2vec.common.model_reader import ModelReader

target_exp = re.compile('\[.*\]')


def parse_input(line):
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word):
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos


def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity < 0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity < 0] = 0.0
    return target_similarity * context_similarity


def evaluate_candidates(model_reader, sent, target_pos):
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model
    n_result = 10

    if sent[target_pos] is None:
        target_v = None
    elif sent[target_pos] not in word2index:
        raise ValueError("Target word is out of vocabulary.")
    else:
        target_v = w[word2index[sent[target_pos]]]
    if len(sent) > 1:
        context_v = model.context2vec(sent, target_pos)
        context_v = context_v / np.sqrt((context_v * context_v).sum())
    else:
        context_v = None

    if target_v is not None and context_v is not None:
        similarity = mult_sim(w, target_v, context_v)
    else:
        if target_v is not None:
            v = target_v
        elif context_v is not None:
            v = context_v
        else:
            raise ValueError("Can't find a target nor context.")
        similarity = (w.dot(v) + 1.0) / 2  # Cosine similarity can be negative, mapping similarity to [0,1]

    count = 0
    for i in (-similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        print('{0}: {1}'.format(index2word[i], similarity[i]))
        count += 1
        if count == n_result:
            break


def main(args):
    model_param_file = args.model_file
    model_reader = ModelReader(gpu=0, config_file=model_param_file)
    sent, target_pos = parse_input(args.inputs)
    evaluate_candidates(model_reader, sent, target_pos)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", type=str)
    parser.add_argument("--inputs", type=str)
    args = parser.parse_args()
    if not (args.model_file and args.inputs):
        parser.print_usage()
        parser.exit()
    return args


if __name__ == '__main__':
    main(parse_arguments())
