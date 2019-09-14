import math
import os
import sys
import time
import random

import numpy as np
from numpy import dot
from numpy.linalg import norm

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn import metrics
import sklearn.metrics.pairwise

def matrix_weight_sum_by_batch_vectors(m, vecs):
    ''' matrix m(dim1, dim2) weighted sum up by BatchVector vecs(batchsize, dim1) get (batchsize, dim2) '''
    
    #print("m:", m.shape)
    #print("vecs:", vecs.shape)
    batch_size = vecs.shape[0]
    m_batch = m.repeat(batch_size, 1, 1) # can be optimized for reducing the GPU memory
    m_batch_t = m_batch.transpose(0, 2)
    vecs_t = vecs.transpose(0, 1)
    mv_t = m_batch_t * vecs_t
    mv = mv_t.transpose(0, 2)
    mv_sum = torch.sum(mv, dim=1)
    return mv_sum

def cos_sim(Ma, Mb):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        Ma(dim1, dim_vec), Mb(dim2, dim_vec) are matrix 
        dim1 can be not equal with dim2
        size_a and size_b can be 1
        get cos sim of every Ma[i] and Mb[j]
        result[i,j] = cos(Ma[i], Mb[j]) for every i,j

        a and b can also be a 1-D torch.tensor() vector

        input a and b must be in torch data (but not python list or numpy.array)
    '''

    #print("a:", a)
    #print("b:", b)

    if 1 == len(Ma.shape):
        assert len(Ma.shape) == len(Mb.shape)
        a_duplicate = Ma.repeat(1,1)
        b_duplicate = Mb.repeat(1,1)
    else:
        vec_dim = Ma.shape[-1]
        assert Mb.shape[-1] == vec_dim
        a_size = Ma.shape[0]
        b_size = Mb.shape[0]

        print("Ma:", Ma)
        Ma = torch.tensor(Ma)
        a_duplicate = torch.t(Ma).repeat(b_size,1,1).transpose(1,2).transpose(0,1)
        #print("a_duplicate:", a_duplicate)
        #print("a_duplicate.shape:",a_duplicate.shape)
        b_duplicate = Mb.repeat(a_size, 1)
#b_duplicate = Mb.repeat(a_size, 1, 1)
        #print("b_duplicate:", b_duplicate)
        #print("b_duplicate.shape:", b_duplicate.shape)
    cos = F.cosine_similarity(a_duplicate, b_duplicate, dim=-1)
    #print("cos:", cos)
    return cos

def cos_sim_by_sklearn(a, b):
    '''
        a, b are vectors
    '''
    return cos_sim_batch_by_sklearn([a], [b]).transpose()[0].transpose()

def cos_sim_batch_by_sklearn(Ma, Mb):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        Ma(dim1, dim_vec), Mb(dim2, dim_vec) are matrix 
        dim1 can be not equal with dim2
        get cos sim of every Ma[i] and Mb[j]
        result[i,j] = cos(Ma[i], Mb[j]) for every i,j
    '''
    return sklearn.metrics.pairwise.cosine_similarity(Ma, Mb)

def cos_sim_manually(A, B):
    aa, bb, ab = [0,0,0]
    for a, b in zip(A, B):
        aa += a * a
        bb += b * b
        ab += a * b
    aabb = aa*bb
    if aabb == 0:
        return 0
    aabb = math.sqrt(aabb)
    return ab/aabb

def float_equal(a, b, neighbourhood_of_filter=0.001):
    if abs(a-b) <= neighbourhood_of_filter:
    #if abs(a-b) <= 1e-5:
    #if abs(a-b) <= 1e-7:
        return True
    return False

def get_vectorwise_sim_of_two_matrix(Ma, Mb, sim_type):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get cos sim of every Ma[i] and Mb[j]
        result[i,j] = cos(Ma[i], Mb[j]) for every i,j

        Ma, Mb, and res are in torch.tensor type
    '''

    if "dot_product" == sim_type:
        Mb_t = torch.t(Mb)
        res = torch.mm(Ma, Mb_t)
    elif "cosine" == sim_type:
        res = cos_sim(Ma, Mb)

    return res

def get_vectorwise_sim_of_two_matrix_by_np(Ma, Mb, sim_type):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get cos sim of every Ma[i] and Mb[j]
        result[i,j] = cos(Ma[i], Mb[j]) for every i,j

        Ma, Mb, and res are in numpy.array type
    '''

    if "dot_product" == sim_type:
        Mb_t = Mb.transpose()
        res = np.dot(Ma, Mb_t)
    elif "cosine" == sim_type:
        res = cos_sim_batch_by_sklearn(Ma, Mb)

    return res

def select_first_k_with_filter(m, m_index, k, filter=1.0, neighbourhood_of_filter=0.001, in_numpy_type=False):
    """ matrix m(dim1, dim2) m_index(dim1, dim2)
        new_m is in torch.rensor type; new_m_index is in numpy.array type
        output: result(dim1, k) without any cell result[i][j] == filter
    """

    new_m = []
    new_m_index = []
    for vec, idxs in zip(m, m_index):
        new_vec = []
        new_idxs = []
        cnt = 0
        for v, i in zip(vec, idxs):
            if not float_equal(v, filter):
                new_vec.append(v)
                new_idxs.append(i)
                cnt += 1
                if cnt >= k:
                    break
        new_m.append(new_vec)
        new_m_index.append(new_idxs)

    if in_numpy_type:
        new_m = np.array(new_m)
    else:
        new_m = torch.tensor(new_m)
    new_m_index = np.array(new_m_index)

    return new_m, new_m_index

def get_vectorwise_nearest_neighbor_of_two_matrix(Ma, Mb, sim_type, filter_totally_matched_sample = True, neighbourhood_of_filter = 0.001):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get top1 neighbors of every Ma[i], every Mb[j] is the candidate of neighbors
        result[i] = nearest neighbor of Ma[i] for every i

        Ma, Mb, and output(max_sim) are in torch.tenser type; max_sim_col is in numpy.array type
    '''

    if filter_totally_matched_sample and "cosine" == sim_type: # dot product cannot be filtered
        topk_indexs, topk_values = \
            get_vectorwise_neighbors_of_two_matrix(Ma, Mb, 1, sim_type, \
                filter_totally_matched_sample, neighbourhood_of_filter)
        #print("in get_vectorwise_nearest_neighbor_of_two_matrix topk_indexs:", topk_indexs)
        #print("in get_vectorwise_nearest_nei topk_values:", topk_values)
        max_sim_col = topk_indexs.transpose()[0].transpose()  # numpy.array (for:1. do not need gradient 2. may be used in fetch_vec_from_matrix_by_index)
        max_sim = torch.t(topk_values)[0] # torch.tensor (for:1. need gradient)
        #print("in get_vectorwise_nearest_neighbor_of_two_matrix max_sim_col:", max_sim_col)
        #print("in get_vectorwise_nearest_nei max_sim:", max_sim)
    else: 
        sim_matrix = get_vectorwise_sim_of_two_matrix(Ma, Mb, sim_type)
        max_sim, max_sim_col = sim_matrix.max(dim=1)
        max_sim_col = max_sim_col.detach().cpu().numpy()
    
    return max_sim_col, max_sim 

def get_vectorwise_neighbors_of_two_matrix(Ma, Mb, topK, sim_type, filter_totally_matched_sample = True, neighbourhood_of_filter = 0.001):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get topK neighbors of every Ma[i], every Mb[j] is the candidate of neighbors
        result[i,j] = j-th neighbor of Ma[i] for every i

        Ma, Mb, and output(topk_values) are in torch.tensor type, topk_indexs is in numpy.array type
    '''

    #sim_matrix = get_vectorwise_sim_of_two_matrix(Ma, Mb, sim_type)
    sim_matrix = get_vectorwise_sim_of_two_matrix_by_np(Ma, Mb, sim_type)
    #print("in get_vectorwise_neighbors_of_two_matrix sim_matrix:", sim_matrix)

    if filter_totally_matched_sample and "cosine" == sim_type: # dot product cannot be filtered
        sim_matrix = torch.tensor(sim_matrix)
        topk_sorted = sim_matrix.sort(dim=1, descending=True)
        topk_values, topk_indexs = topk_sorted
        #print("in get_vectorwise_neighbors_of_two_matrix topk_indexs:", topk_indexs)
        #print("in get_vectorwise_neighbors_of_two_matrix topk_values:", topk_values)

        topk_indexs_np = topk_indexs.detach().cpu().numpy()
        topk_values_filtered, topk_indexs_filtered = select_first_k_with_filter( \
                topk_values, topk_indexs_np, topK, filter=1.0, \
                neighbourhood_of_filter=neighbourhood_of_filter, in_numpy_type=False)
        return topk_indexs_filtered, topk_values_filtered
    else:
        topk_sorted = sim_matrix.sort(dim=1, descending=True)
        topk_values, topk_indexs = topk_sorted
        #print("in get_vectorwise_neighbors_of_two_matrix topk_indexs:", topk_indexs)
        #print("in get_vectorwise_neighbors_of_two_matrix topk_values:", topk_values)
        if 1 == topK:
            topk_values = torch.t(torch.t(topk_values)[:topK])
            topk_indexs = torch.t(torch.t(topk_indexs)[:topK])
            #print("2 in get_vectorwise_neighbors_of_two_matrix topk_indexs:", topk_indexs)
            #print("2 in get_vectorwise_neighbors_of_two_matrix topk_values:", topk_values)
        else:
            topk_values = torch.t(torch.t(topk_values)[:topK])
            topk_indexs = torch.t(torch.t(topk_indexs)[:topK])
        topk_indexs_np = topk_indexs.detach().cpu().numpy()

        return topk_indexs_np, topk_values

def fetch_vec_from_matrix_by_index(m, indexs):
    '''
        matrix m(dim1, dim_vec)
        list(1-D array) index(dim2)     
            index must be in int list or int numpy.array !!!
        return:  matrix(a list of vectors) res(dim2, dim_vec)
    '''
    #print("m:",m)
    #print("indexs:",indexs)
    #print("m.shape:",m.shape)
    try:
        res = m[indexs]
    except:
        indexs_tensor = torch.tensor(indexs)
        #print("indexs_tensor:",indexs_tensor)
        #print("indexs_tensor.shape:",indexs_tensor.shape)
        res = m[indexs_tensor]
    return res

def get_vectorwise_nearest_neighbor_of_two_matrix_by_np(Ma, Mb, sim_type, filter_totally_matched_sample = True, neighbourhood_of_filter = 0.001):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get top1 neighbors of every Ma[i], every Mb[j] is the candidate of neighbors
        result[i] = nearest neighbor of Ma[i] for every i

        Ma, Mb, and output(max_sim_col, max_sim) are in numpy.array type
    '''

    if not filter_totally_matched_sample:
        sim_matrix = get_vectorwise_sim_of_two_matrix_by_np(Ma, Mb, sim_type)
        max_sim_col = np.argmax(sim_matrix, axis=1)
        max_sim = np.max(sim_matrix, axis=1)
    else:
        topk_indexs, topk_values = \
            get_vectorwise_neighbors_of_two_matrix(Ma, Mb, 1, sim_type, \
                filter_totally_matched_sample, neighbourhood_of_filter)
        max_sim_col = topk_indexs.transpose()[0].transpose()
        max_sim = topk_values.transpose()[0].transpose()
    
    return max_sim_col, max_sim 

def get_vectorwise_neighbors_of_two_matrix_by_np(Ma, Mb, topK, sim_type, filter_totally_matched_sample = True, neighbourhood_of_filter = 0.001):
    ''' matrix Ma(size_a, dim_vec) matrix Mb(size_b, dim_vec)
        get topK neighbors of every Ma[i], every Mb[j] is the candidate of neighbors
        result[i,j] = j-th neighbor of Ma[i] for every i

        Ma, Mb, and output(topk_indexs, topk_values) are in numpy.array type
    '''

    sim_matrix = get_vectorwise_sim_of_two_matrix_by_np(Ma, Mb, sim_type)
    if filter_totally_matched_sample and "cosine" == sim_type: # dot product cannot be filtered
        topk_indexs = np.argsort(sim_matrix)
        topk_values = np.sort(sim_matrix)

        topk_indexs = topk_indexs.transpose()[::-1].transpose() # reverse
        topk_values = topk_values.transpose()[::-1].transpose() # reverse

        topk_values, topk_indexs = select_first_k_with_filter( \
                topk_values, topk_indexs, topK, filter=1.0, \
                neighbourhood_of_filter=neighbourhood_of_filter, in_numpy_type=True)
    else: 
        topk_indexs = np.argsort(sim_matrix).transpose()[(0-topK):].transpose()
        topk_values = np.sort(sim_matrix).transpose()[(0-topK):].transpose()

        topk_indexs = topk_indexs.transpose()[::-1].transpose() # reverse
        topk_values = topk_values.transpose()[::-1].transpose() # reverse

    return topk_indexs, topk_values

def test_cosine_similarity():
    a=torch.tensor([[1,2,3.0],[0,0,0.0]])
    b=torch.tensor([[1,2,3.1],[-1,-2,-3.0]])
    a2=torch.tensor([[1,2,3.0],[0,0,0]])
    b2=torch.tensor([[1,2,3.1],[-1,-2,-3.0],[5,5,5.0],[6,6,6.0]])
    a3=torch.tensor([1,2,3.0])
    b3=torch.tensor([1,2,3.1])
    a31=torch.tensor([[1,2,3.0]])
    b31=torch.tensor([[1,2,3.1]])

    ar=np.random.rand(5,10)
    br=np.random.rand(15,10)
    art=torch.tensor(ar)
    brt=torch.tensor(br)

    cos(a,b)
    cos(a2,b2)
    cos(art,brt)
    print("sklearn cos:", sklearn.metrics.pairwise.cosine_similarity(ar,br))
    cos(a3,b3)
    try:
        cos(a3,b3)
    except:
        print("cos(a3,b3) failed")
    try:
        cos([a3],[b3])
    except:
        print("cos(a3,b3) failed")
    cos(a31,b31)

def test_select_first_k_with_filter():
    print("Start to test select_first_k_with_filter")
    topk=3
    art=torch.tensor(np.random.rand(5,10))
    value=torch.tensor(np.random.rand(15,10))
    #art=art.cuda()
    #value=value.cuda()
    print("art:", art)
    artsorted = art.sort(dim=1)
    print("artsorted:", artsorted)
    m, m_index = artsorted
    m_index = m_index.detach().cpu().numpy()
    res_m, res_index = select_first_k_with_filter(m, \
        m_index, topk, filter=1.0, neighbourhood_of_filter=0.001, in_numpy_type = False)
    print("res_m:", res_m)
    print("res_index:", res_index)
    res_fetch_vec = fetch_vec_from_matrix_by_index(value, res_index)
    print("value:", value)
    print("res_fetch_vec:", res_fetch_vec)
    print("test select_first_k_with_filter and fetch_vec_from_matrix_by_index done\n\n")

def test_get_vectorwise_nearest_neighbor_of_two_matrix():
    print("Start to test get_vectorwise_nearest_neighbor_of_two_matrix")
    art=torch.tensor(np.random.rand(5,10))
    print("art:", art)
    value=torch.tensor(np.random.rand(15,10))
    print("value:", value)
    print("\nOption1: cosine filter_totally_matched_sample=True")
    max_sim_col, max_sim = get_vectorwise_nearest_neighbor_of_two_matrix(art, \
        value, "cosine", filter_totally_matched_sample=True, neighbourhood_of_filter=0.001)  
    print("max_sim_col:", max_sim_col)
    print("max_sim:", max_sim)
    res_fetch_vec = fetch_vec_from_matrix_by_index(value, max_sim_col)
    print("res_fetch_vec:",res_fetch_vec)

    print("\nOption2: cosine filter_totally_matched_sample=False")
    max_sim_col, max_sim = get_vectorwise_nearest_neighbor_of_two_matrix(art, \
        value, "cosine", filter_totally_matched_sample=False, neighbourhood_of_filter=0.001)  
    print("max_sim_col:", max_sim_col)
    print("max_sim:", max_sim)
    res_fetch_vec = fetch_vec_from_matrix_by_index(value, max_sim_col)
    print("res_fetch_vec:",res_fetch_vec)

    print("\nOption3: dot product filter_totally_matched_sample=False")
    max_sim_col, max_sim = get_vectorwise_nearest_neighbor_of_two_matrix(art, \
        value, "dot_product", filter_totally_matched_sample=False, neighbourhood_of_filter=0.001)  
    print("max_sim_col:", max_sim_col)
    print("max_sim:", max_sim)
    res_fetch_vec = fetch_vec_from_matrix_by_index(value, max_sim_col)
    print("res_fetch_vec:",res_fetch_vec)

    print("\nOption4: dot product filter_totally_matched_sample=True")
    max_sim_col, max_sim = get_vectorwise_nearest_neighbor_of_two_matrix(art, \
        value, "dot_product", filter_totally_matched_sample=True, neighbourhood_of_filter=0.001)  
    print("max_sim_col:", max_sim_col)
    print("max_sim:", max_sim)
    res_fetch_vec = fetch_vec_from_matrix_by_index(value, max_sim_col)
    print("res_fetch_vec:",res_fetch_vec)

def test_cases():
    test_select_first_k_with_filter()
    test_get_vectorwise_nearest_neighbor_of_two_matrix()
    #test_cosine_similarity()

#test_cases()


