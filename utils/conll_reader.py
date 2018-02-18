# -*- coding:utf-8 -*-
from __future__ import print_function
import codecs

POSTAG_DICT = {'SP':0,
'BA':1,
'FW':2,
'DER':3,
'DEV':4,
'MSP':5,
'ETC':6,
'JJ':7,
'DT':8,
'DEC':9,
'VE':10,
'LB':11,
'LC':12,
'NN':13,
'PU':14,
'URL':15,
'NR':16,
'DEG':17,
'PN':18,
'FRAG':19,
'VA':20,
'VC':21,
'AD':22,
'CC':23,
'M':24,
'CD':25,
'P':26,
'AS':27,
'IJ':28,
'VV':29,
'CS':30,
'X':31,
'ON':32,
'NOI':33,
'NT':34,
'OD':35,
'SB':36
}

class ConllEntry(object):
    def __init__(self, idx, token, postag):
        self.idx = idx
        self.token = token
        self.postag = postag
        self.pos_idx = POSTAG_DICT[self.postag] if postag != None else 0
        self.parent_id = 0
        self.fields = {}

    def __unicode__(self):
        extra_fields = []
        for (k,v) in self.fields.items():
            extra_fields.append(u'{0}:{1}'.format(k, v))
        return u'{0}\t{1}\t{2}\t{3}'.format(str(self.idx), self.token, str(self.parent_id), '\t'.join(extra_fields))

def posidx2postag(idxseq):
    idx2postag = {}
    for (k, v) in POSTAG_DICT.items():
        idx2postag[v] = k
    # return [POSTAG_DICT.keys()[idx] for idx in idxseq]
    return [idx2postag[i] for i in idxseq]

def read_conll(sentence):
    lines = sentence.split('\n')
    for l in lines:
        parts = l.split()


def load_conll(path):
    sentences = []
    with codecs.open(path, mode='r', encoding='utf-8') as f:
        buff = []
        for l in f:
            if len(l.strip()) > 0:
                buff.append(l)
            else:
                conll_sentence = []
                for s in buff:
                    parts = s.split()
                    conll_entry = ConllEntry(int(parts[0]), parts[1], parts[3])
                    conll_entry.parent_id = int(parts[6])
                    conll_sentence.append(conll_entry)

                sentences.append(conll_sentence)
                buff = []

    return sentences

def postags(sentences):
    postag_set = set()
    for entries in sentences:
        for e in entries:
            postag_set.add(e.postag)
    return postag_set

def vocab(sentences):
    vocab_set = set()
    for entries in sentences:
        for e in entries:
            vocab_set.add(e.token)
    return vocab_set

def vocab_dict(path):
    word2idx = {}
    idx = 0
    with codecs.open(path, encoding='utf-8', mode= 'r') as f:
        for l in f:
            word2idx[l.strip()] = idx
            idx += 1
    return word2idx

def char_dict(path):
    char2idx = {}
    idx = 0
    with codecs.open(path, encoding='utf-8', mode='r') as f:
        for l in f:
            char2idx[l.strip()] = idx
            idx += 1
    return char2idx

def charset(sentences):
    char_set = set()
    for entries in sentences:
        for e in entries:
            for ch in e.token:
                char_set.add(ch)
    return char_set

if __name__ == '__main__':
    sentences = load_conll('../data/treebank.conll')
    # postag_set = postags(sentences)
    # idx = 0
    # for p in postag_set:
    #     print('\'{0}\':{1},'.format(p, idx))
    #     idx += 1

    # vocab_set = vocab(sentences)
    # with codecs.open('../data/vocab', mode='w', encoding='utf-8') as f:
    #     for v in vocab_set:
    #         print(v, file=f)

    # char_set = charset(sentences)
    # with codecs.open('../data/char', mode='w', encoding='utf-8') as f:
    #     for c in char_set:
    #         print(c, file=f)

    print(len(sentences))