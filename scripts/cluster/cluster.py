from sklearn.cluster import KMeans
import numpy as np
import sys

def load_vector(file):
    f = open(file, "r")
    vectors = []
    dict = {}
    rdict = []
    count = 0

    for line in f:
        slots = line.strip().split("\t")
        if len(slots) != 2:
            continue
        word_str, vec_str = slots
        words = word_str.split()
        vec = list(map(float, vec_str.split()))
   
        vectors.append(vec) 
        rdict.append(word_str)  
        dict[word_str] = count
        count += 1
    return vectors, dict, rdict  

def load_vector_only_vector(file):
    f = open(file, "r")
    vectors = []
    dict = {}
    rdict = []
    count = 0

    for line in f:
        line = line.strip()
        vec = list(map(float, line.split()))
   
        vectors.append(vec) 
        count += 1
    return vectors

def load_vector_with_type(file, type):
    f = open(file, "r")
    vectors = []
    dict = {}
    rdict = []
    count = 0

    for line in f:
        slots = line.strip().split("\t")
        if len(slots) != 3:
            continue
        if slots[0] != type:
            continue
        word_str, vec_str = slots[1:]
        words = word_str.split()
        vec = list(map(float, vec_str.split()))
   
        vectors.append(vec) 
        rdict.append(word_str)  
        dict[word_str] = count
        count += 1
    return vectors, dict, rdict  

def cluster(vectors, cluster_num):
    p = KMeans(n_clusters=cluster_num, init="k-means++", n_init=10, max_iter=30, tol=0.0001, precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm="auto")
    model = p.fit(vectors)
    #print(model.labels_)
    return model

def output(model, rdict, bysample = False, bycluster = True):
    labels = model.labels_
    assert len(labels) == len(rdict)
    labels_with_sent = []
    for label, sent in zip(labels, rdict):
        if bysample:
            print(str(label) + "\t" + sent)
        if bycluster:
            labels_with_sent.append([label, sent])
    if bycluster:
        labels_with_sent = sorted(labels_with_sent, key = lambda x:x[0])
        for label, sent in labels_with_sent:
            print(str(label) + "\t" + sent)

#vectors, dict, rdict = load_vector_with_type(sys.argv[1], sys.argv[2])
vectors = load_vector_only_vector(sys.argv[1])
#vectors = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
model = cluster(vectors, int(sys.argv[2]))
#model = cluster(vectors, int(sys.argv[3]))
#output(model, rdict)
