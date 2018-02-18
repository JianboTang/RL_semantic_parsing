# This file contains routines from Lisbon Machine Learning summer school.
# The code is freely distributed under a MIT license. https://github.com/LxMLS/lxmls-toolkit/
import numpy as np
import bottleneck
from heapq import heappush, heappop
import sys
from dynet import *
import edmonds
from collections import defaultdict, namedtuple
from operator import itemgetter
import kmst

def unsupervised_error(scores, expr):
    topk = kbest(scores)
    nr, nc = np.shape(scores)
    matrixs = []
    for heads in topk:
        matrix = np.array([[0 for _ in xrange(nr)] for _ in xrange(nc)])
        for mod, h in enumerate(heads[1:]):
            matrix[mod + 1][h] = 1
        matrixs.append(matrix)
    return loss(scores, matrixs, expr)

def kbest(scores, k = 2):
    #return head info
    nr, nc = np.shape(np_softmax(scores))
    t_score = np.transpose(scores)
    mst = kmst.EdmondMST()
    G = mst.tran_score(t_score)
    k_bests = mst.get_kbest(G, k)
    heads = []
    for r in k_bests:
        h = [-1] * nr
        for e in r.values():
            h[e.org_dst] = e.org_src
        heads.append(h)
    return heads

def head_matrix(conll_sentence):
    sz = len(conll_sentence)
    matrix = np.array([[0 for _ in xrange(sz)] for _ in xrange(sz)])
    for i, entry in enumerate(conll_sentence):
        matrix[i, entry.parent_id] = 1
    return matrix

def softmax_expr(expr):
    nr = len(expr)
    s_expr = [None] * nr
    for i in xrange(nr):
        s_expr[i] = softmax(concatenate(expr[i]))
        # print(s_expr[i].npvalue())
    return s_expr

def np_softmax(scores):
    nr, nc = np.shape(scores)
    for i in xrange(nr):
        x = scores[i, :]
        e_x = np.exp(x - np.max(x))
        scores[i,:] = e_x / e_x.sum()
    return scores

def linear_regularization_on_row(matrix):
    nr, nc = np.shape(matrix)
    for i in xrange(nr):
        x = matrix[i, :]
        if np.max(x) != 0:
            matrix[i,:] = x / np.max(x)
        else:
            for j in xrange(nr):
                matrix[i,j] = 1.0

    return matrix

def parse_head(scores):
    head = np.argmax(scores, -1)
    return head

def loss(scores, possible_mats, expr, ignore_root = False):
    #sum up scores according to possible matrix
    # matrix in possible_mats is either 0 or 1

    scores = np_softmax(scores)
    nr, nc = np.shape(scores)
    if nr != nc:
        raise Exception('column != row')
    path_total = []
    paths = []
    for mat in possible_mats:
        shape = np.shape(mat)
        score = 0
        path = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mat[i,j] == 1:
                    score += scores[i,j]
                    path.append((i,j))
        path_total.append(score)
        paths.append(path)

    r_score = np.array([[0.0 for _ in xrange(nr)] for _ in xrange(nc)])
    for i, path_score in enumerate(path_total):
        for s, e in paths[i]:
            r_score[s,e] += scores[s,e] / path_score


    r_score = linear_regularization_on_row(r_score)
    loss = None
    expr = softmax_expr(expr)
    for row_idx in range(nr):
        pos = None
        neg = None
        for i in range(nc):
            if r_score[row_idx, i] > 0:
                if pos:
                    pos += expr[row_idx][i] * r_score[row_idx, i]
                else:
                    pos = expr[row_idx][i] * r_score[row_idx, i]
            else:
                if neg:
                    neg += log(-expr[row_idx][i] + 1.0001)
                else:
                    neg = log(-expr[row_idx][i] + 1.0001)
            # print("r_score: {0}".format(r_score[row_idx, i]))
            # print("expr[row_idx][i] = {0}".format(expr[row_idx][i].value()))
        if loss:
            loss += -(log(pos + 0.0001) + 0 if neg == None else neg)
        else:
            loss = -(log(pos + 0.0001) + 0 if neg == None else neg)
        # print("loss: {0}".format(loss.value()))

        # comp = concatenate([pos, neg])
        # pos -> 1

    return loss


def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    incomplete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1). 
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in xrange(1,N+1):
        for s in xrange(N-k+1):
            t = s+k
            
            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s]# + (0.0 if gold is not None and gold[s]==t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t]# + (0.0 if gold is not None and gold[t]==s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)
        
    value = complete[0][N][1]
    heads = [-1 for _ in range(N+1)] #-np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in xrange(1,N+1):
        h = heads[m]
        value_proj += scores[h,m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the 
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return