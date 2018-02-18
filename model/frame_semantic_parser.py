# -*- coding:utf-8 -*-
from dynet import *
import numpy as np
import random
import datetime as dt
from utils.decoder import *

class Entry(object):
    def __init__(self):
        self.embedding = None
        self.lstms = [None, None]
        self.headfov = None
        self.modfov = None

class Node(object):
    def __init__(self, id):
        self.id = id
        self.parent_id = 0
        self.children = []
        self.slot_combinations = []
        self.depth = 0

def all_structures_without_head(start, end):
    structures = []
    for head_count in xrange(1, end - start + 1):
        # C(end - start + 1, head_count)
        stick_count = head_count - 1
        ongoings = [[0 for _ in xrange(end - start - 1)]]
        for _ in xrange(stick_count):
            next_states = []
            for prev_context in ongoings:
                available_max = 0
                for (idx, v) in enumerate(prev_context):
                    if v != 0:
                        break
                    available_max = idx + 1
                if available_max == 0:
                    continue
                for i in xrange(available_max):
                    next_state = [t for t in prev_context]
                    next_state[i] = 1
                    next_states.append(next_state)
            ongoings = next_states
        for state in ongoings:
            #for each state, assume a head and get all_structures
            ranges = []
            st = start
            for (i, v) in enumerate(state):
                if v == 1:
                    ranges.append((st, start + i + 1))
                    st = start + i + 1
            ranges.append((st, end))
            range_structures = {}
            for (range_idx, range) in enumerate(ranges):
                current_range_structures = []
                range_structures[range_idx] = current_range_structures
                for head in xrange(range[0], range[1]):
                    sub_structures = all_structures_with_head(range[0], range[1], head)
                    for s in sub_structures:
                        current_range_structures.append(s)
            #do all combination
            combinations = [[]]
            for (range_idx, parents_id) in range_structures.items():
                next_steps = []
                for prev_path in combinations:
                    for v in parents_id:
                        temp = [t for t in prev_path]
                        temp.append(v)
                        next_steps.append(temp)
                combinations = next_steps

            for comb in combinations:
                current_structure = []
                for (range_idx, v) in enumerate(comb):
                    current_structure.extend(v)
                structures.append(current_structure)
    # [[parent_id1, parent_id2, ...], ...]
    return structures

def all_structures_with_head(start, end, center_idx):
    left_structures = all_structures_without_head(start, center_idx)
    right_structures = all_structures_without_head(center_idx + 1, end)
    total_structures = [s for s in left_structures]
    total_structures.extend(right_structures)
    for s in total_structures:
        for i in xrange(len(s)):
            if s[i] == -1:
                s[i] = center_idx
    if len(left_structures) == 0:
        left_structures.append([])
    if len(right_structures) == 0:
        right_structures.append([])
    #a x b
    structures = []
    for left_structure in left_structures:
        for right_structure in right_structures:
            structure = []
            structure.extend(left_structure)
            structure.extend([-1])
            structure.extend(right_structure)
            structures.append(structure)
    return structures

class FrameSemanticParser(object):
    CONCEPT = u'concept'

    def __init__(self, vocab_dict, char_dict, options):
        #every slot has it's own attention
        #every frame
        self.model = Model()
        random.seed(1)
        embedding_dim = options.embedding_dim
        char_embedding_dim = options.char_embedding_dim
        word_category_dim = options.word_category_dim
        postag_dim = options.postag_dim
        postag_label_sz = options.postag_label_size
        assert word_category_dim % 2 == 0
        lstm_hidden_dim = options.lstm_hidden_dim
        layers = options.lstm_layers
        vocab_capacity = options.vocab_capacity
        char_capacity = options.char_capacity
        attn_hidden = options.attn_hidden
        token_conv_size = options.token_conv_size

        self.word2idx = vocab_dict
        self.word2idx = self.__reorder_dict(vocab_dict)
        self.word_embedding_dim = embedding_dim
        self.token_conv_size = token_conv_size
        self.char2idx = char_dict
        self.char2idx = self.__reorder_dict(char_dict)
        self._char_embedding = self.model.add_lookup_parameters((char_capacity, char_embedding_dim))
        self.token_conv_kernels = [self.model.add_parameters((1, i + 1, char_embedding_dim, embedding_dim)) for i
                                   in xrange(token_conv_size)]

        self.postag_encoder = [VanillaLSTMBuilder(1, word_category_dim + embedding_dim, word_category_dim, self.model),
                               VanillaLSTMBuilder(1, word_category_dim + embedding_dim, word_category_dim, self.model)]
        #add l1 loss constraints
        self.postag_W1 = self.model.add_parameters((postag_label_sz, word_category_dim * 2))
        self.postag_b1 = self.model.add_parameters((postag_label_sz))

        self._word_embedding = self.model.add_lookup_parameters((vocab_capacity, embedding_dim))
        self._category_embedding = self.model.add_lookup_parameters((vocab_capacity, word_category_dim))
        self._postag_matrix = self.model.add_parameters((postag_dim, postag_label_sz))


        #multi-layer lstm
        self.first_layer = [VanillaLSTMBuilder(1, embedding_dim + word_category_dim, lstm_hidden_dim, self.model),
                         VanillaLSTMBuilder(1, embedding_dim + word_category_dim, lstm_hidden_dim, self.model)]
        self.builders = [[VanillaLSTMBuilder(1, 2 * lstm_hidden_dim, lstm_hidden_dim, self.model),
                         VanillaLSTMBuilder(1, 2 * lstm_hidden_dim, lstm_hidden_dim, self.model)] for _ in xrange(layers)]

        self.hidLayerFOH = self.model.add_parameters((attn_hidden, lstm_hidden_dim * 2))
        self.hidLayerFOM = self.model.add_parameters((attn_hidden, lstm_hidden_dim * 2))
        self.hidBias = self.model.add_parameters((attn_hidden))

        self.U1 = self.model.add_parameters((attn_hidden, attn_hidden))
        self.U2 = self.model.add_parameters((attn_hidden))

    def __reorder_dict(self, dict_to_process):
        vv = {}
        vv['**oov**'] = 0
        for (k,v) in dict_to_process.items():
            vv[k] = v + 1
        return vv

    def  __getExpr(self, sentence, i, j):
        if sentence[i].headfov is None:
            sentence[i].headfov = rectify(self.hidLayerFOH.expr() * concatenate([sentence[i].lstms[0], sentence[i].lstms[1]]))
        if sentence[j].modfov is None:
            sentence[j].modfov  = rectify(self.hidLayerFOM.expr() * concatenate([sentence[j].lstms[0], sentence[j].lstms[1]]))

        output = transpose(sentence[i].headfov) * self.U1.expr() * sentence[j].modfov
                 #+ transpose(sentence[i].headfov) * self.U2.expr()
        return output

    def __evaluate(self, sentence):
        exprs = [ [self.__getExpr(sentence, i, j) for j in xrange(len(sentence))] for i in xrange(len(sentence)) ]
        scores = np.array([ [output.scalar_value() for output in exprsRow] for exprsRow in exprs ])

        return scores, exprs

    def seq_label(self, tokens):
        category_embeddings = []
        for t in tokens:
            word_category = concatenate([self.token_embedding(t), self.word_category(t)])
            entry = Entry()
            entry.embedding = word_category
            category_embeddings.append(entry)

        #bi-lstm
        forward = self.postag_encoder[0].initial_state()
        backward = self.postag_encoder[1].initial_state()
        f_vec = []
        b_vec = []
        for v, rv in zip(category_embeddings, reversed(category_embeddings)):
            forward = forward.add_input(v.embedding)
            backward = backward.add_input(rv.embedding)
            f_vec.append(forward.output())
            b_vec.append(backward.output())

        pos_embedding = []
        for f_v, b_v in zip(f_vec, reversed(b_vec)):
            v = concatenate([f_v, b_v])
            postag_embedding = self._postag_matrix.expr() * softmax(self.postag_W1.expr() * v + self.postag_b1.expr())
            pos_embedding.append(postag_embedding)
        return pos_embedding

    def word_category(self, token):
        if token in self.word2idx:
            idx = self.word2idx[token]
        else:
            idx = self.word2idx['**oov**']
        return self._category_embedding[idx]

    def char_emb(self, token):
        char_emb_seq = []
        for c in token:
            if c in self.char2idx:
                char_emb_seq.append(self._char_embedding[self.char2idx[c]])
            else:
                char_emb_seq.append(self._char_embedding[self.char2idx['**oov**']])
        return char_emb_seq

    def token_embedding(self, token):
        conv_kernels = []
        for kernel_idx in xrange(self.token_conv_size):
            if len(token) >= kernel_idx + 1:
                conv_kernels.append(self.token_conv_kernels[kernel_idx])

        char_emb_seq = transpose(concatenate_cols(self.char_emb(token)))
        shape, _ = char_emb_seq.dim()
        char_emb_seq = reshape(char_emb_seq, (1, shape[0], shape[1]))
        pool_emb_seq = []
        for kernel in conv_kernels:
            conv_emb = conv2d(char_emb_seq, kernel.expr(), [1, 1], is_valid=True)
            pooled_emb = kmax_pooling(conv_emb, 1)
            pool_emb_seq.append(pooled_emb)
        feature_emb = esum(pool_emb_seq)
        feature_emb = reshape(feature_emb, (self.word_embedding_dim,))

        if token in self.word2idx:
            word_embedding = self._word_embedding[self.word2idx[token]]
            emb = esum([word_embedding, feature_emb])
        else:
            emb = feature_emb
        return emb

    def encode(self, tokens):
        embeddings = []
        word_categories = self.seq_label(tokens)
        for (i, t) in enumerate(tokens):
            word_embedding = self.token_embedding(t)
            postag_embedding = word_categories[i]
            cat = concatenate([word_embedding, postag_embedding])
            entry = Entry()
            entry.embedding = cat
            embeddings.append(entry)

        forward = self.first_layer[0].initial_state()
        backward = self.first_layer[1].initial_state()

        for entry, rev_entry in zip(embeddings, reversed(embeddings)):
            forward = forward.add_input(entry.embedding)
            backward = backward.add_input(rev_entry.embedding)

            entry.lstms[0] = forward.output()
            rev_entry.lstms[1] = backward.output()

        for e in embeddings:
            e.embedding = concatenate(e.lstms)

        for lstms in self.builders:
            forward = lstms[0].initial_state()
            backward = lstms[1].initial_state()

            for entry, rev_entry in zip(embeddings, reversed(embeddings)):
                forward = forward.add_input(entry.embedding)
                backward = backward.add_input(rev_entry.embedding)

                entry.lstms[0] = forward.output()
                entry.lstms[1] = backward.output()

            for e in embeddings:
                e.embedding = concatenate(e.lstms)

        scores, exprs = self.__evaluate(embeddings)
        return scores, exprs

    #only support single frame atm
    def pos_loss(self, tokens, token_indexs, head_idx):
        token_seq = ['root']
        token_seq.extend(tokens)
        scores, exprs = self.encode(token_seq)
        # head_scores = []
        # total_hit = 0
        idx_in_token_idx = token_indexs.index(head_idx)
        structures = all_structures_with_head(0, len(token_indexs), idx_in_token_idx)
        #calculate sum of all possibility of structures
        total_poss = []
        token_poss = {}
        for idx in token_indexs:
            token_poss[idx + 1] = softmax(concatenate(exprs[idx + 1]))
        max_hit = 0
        for structure in structures:
            p = []
            hit = 0
            for i, parent_id in enumerate(structure):
                if parent_id != -1:
                    p.append(token_poss[token_indexs[i] + 1][token_indexs[parent_id] + 1])
                    if np.argmax(scores[token_indexs[i] + 1]) == token_indexs[parent_id] + 1:
                        hit += 1
            if hit > max_hit:
                max_hit = hit
            poss = reduce(lambda a,b: a * b, p) if len(p) > 0 else None
            if poss != None:
                total_poss.append(poss)
        sum_p = esum(total_poss) if len(total_poss) > 0 else None
        return -log(sum_p + 1e-6) if sum_p != None else None, max_hit

    def conll_loss(self, conll_entries):
        #conll_entries
        #id, token, parent_id, concepts
        tokens = ['root']
        for e in conll_entries:
            t = e.token
            tokens.append(t)
        scores, exprs = self.encode(tokens)
        loss = []
        hit = 0
        for e in conll_entries:
            scores_i = softmax(concatenate(exprs[e.idx]))
            predicated_head = np.argmax(scores[e.idx])
            if predicated_head == e.parent_id:
                hit += 1
            loss.append(-log(scores_i[e.parent_id] + 1e-6))
        return esum(loss) if len(loss) > 0 else None, hit

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model.populate(path)

    def similarity(self, t1, t2):
        if self.w2vmodel != None and t1 in self.w2vmodel.wv and t2 in self.w2vmodel.wv:
            return self.w2vmodel.wv.similarity(t1, t2)
        return 0

    def __depth(self, node, nodes):
        depth = 0
        current_node = node
        while current_node.parent_id > 0:
            depth += 1
            current_node = nodes[current_node.parent_id]
        return depth

    def clear_nodes_slot_info(self, nodes):
        for n in nodes:
            n.slot_combinations = []

    def in_same_clique(self, nodes, matrix):
        count = 0
        entries_id = set()
        for n in nodes:
            entries_id.add(n.id)
        for n in nodes:
            if matrix[n.id] in entries_id:
                count += 1
        return count == len(nodes) - 1 and max(entries_id) - min(entries_id) == len(nodes) - 1

    def predicate_dependency(self, tokens):
        token_with_root = ['root']
        for t in tokens:
            token_with_root.append(t)
        scores, exprs = self.encode(token_with_root)
        # decode proj
        heads = parse_proj(np.transpose(scores))
        return heads

if __name__ == '__main__':
    #test
    all_structures = all_structures_with_head(0, 5, 2)
    for s in all_structures:
        print(s)

