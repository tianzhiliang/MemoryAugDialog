import sys
import os
import numpy as np
from numpy import dot
from numpy.linalg import norm

import kgdlg.utils.operation_utils as operation_utils

from sklearn import metrics
import sklearn.metrics.pairwise
import math 
import gensim.models as g
import codecs

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000
filter_totally_matched_sample = True

#use_vector_file = True
use_vector_file = False

def load_samples(file):
    f = open(file, "r")
    d = []
    for line in f:
        line = line.strip()
        d.append(line)
    f.close()
    return d

def load_vectors(file):
    f = open(file, "r")
    d = []
    for line in f:
        vec = line.strip().split()
        d.append(list(map(float, vec)))
    f.close()
    return d

def get_vectors(test_file, vector_file, model_path):
    #load model
    model = g.Doc2Vec.load(model_path)
    test_docs = [ x.strip().split() for x in codecs.open(test_file, "r", "utf-8").readlines() ]
     
    #infer test vectors
    output = open(vector_file, "w")
    vectors = []
    vectors_in_np = []
    for d in test_docs:
        vector = model.infer_vector(d, alpha=start_alpha, steps=infer_epoch)
        #print("vector:", vector)
        vectors.append(vector.tolist())
        vectors_in_np.append(vector)

    for v in vectors:
        output.write(" ".join(list(map(str, v))) + "\n")
        #output.write(" ".join([str(x) for x in v]) + "\n")
    #output.flush()
    output.close()

    return vectors, vectors_in_np

def get_sim_and_neighbor(matrix, vector, topK):
    res = []
    for i, m in enumerate(matrix):
        res.append([i, operation_utils.cos_sim_manually(m, vector)])
    res_sorted = sorted(res, key=lambda x:x[1], reverse=True)
    res_topK = res_sorted[:topK]
    return res_topK, res_sorted

def near_neighbor(samples, vectors, topK):
    for i, sample in enumerate(samples):
        topK_ids, sims = get_sim_and_neighbor(vectors, vectors[i], topK)
        print("SampleID:" + str(i) + "\tSample:\t" + sample)
        for (id, sim) in topK_ids:
            print(samples[id] + "\t" + str(sim) + "\t" + str(id))
        print("")

def test_cases():
    sim_type = "cosine"
    a = np.array([[1,2],[3,4]])
    b = np.array([[1.1,2],[-3,-4],[5,5],[1,2]])
    c = np.array([[1,2],[3,4]])
    d = np.array([[1.1,3],[2,4]])
    cos = operation_utils.cos_sim_by_np(c, d)
    print("c:",c)
    print("d:",d)
    print("cos:", cos)
    sklearn.metrics.pairwise.cosine_similarity(a,b)
    sim_matrix = get_vectorwise_sim_of_two_matrix(a, b, sim_type)
    max_sim_col, max_sim = get_vectorwise_nearest_neighbor_of_two_matrix(a, b, sim_type)
    print("a:", a)
    print("b:", b)
    print("sim_matrix:", sim_matrix)
    print("max_sim_col:", max_sim_col)
    print("max_sim:", max_sim)

#test_cases()
