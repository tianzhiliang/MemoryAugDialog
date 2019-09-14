import sys,os,random,math
import numpy as np


def load_vector_with_type(file, type, sample_size):
    f = open(file, "r")
    vectors = []
    sample2id = {}
    samples_list = []
    count = 0

    for line in f:
        slots = line.strip().split("\t")
        if len(slots) != 3:
            continue
        if slots[0] != type:
            continue
        words_str, vec_str = slots[1:]
        words = words_str.split()

        try:
            stop_index = words.index("</s>")
            words = words[:stop_index]
        except:
            pass
        words_str = " ".join(words)

        vec = list(map(float, vec_str.split()))
        vectors.append(vec) 
        samples_list.append(words_str)  
        sample2id[words_str] = count
        count += 1
        if count >= sample_size:
            break

    print("load_vector_with_type done. size:", len(samples_list))

    vectors = np.array(vectors)
    return vectors, sample2id, samples_list  

def load_sample2cluster(file):
    f = open(file, "r")
    cluster2samples = [[] for i in range(1000)]
    samples2cluster = {}
    samples2clusterlist = []
    for i, line in enumerate(f):
        line = line.strip()
        cid = int(line)
        sid = int(i)
        cluster2samples[cid].append(sid)
        samples2cluster[sid] = cid
        samples2clusterlist.append(cid)

    f.close()
    return cluster2samples, samples2cluster, samples2clusterlist

def print_query_under_cluster(cluster2samples, samples_list):
    for i, samples in enumerate(cluster2samples):
        print("cluster:\t" + str(i))
        for sid in samples:
            print(str(sid) + "\t" + samples_list[sid])

def main():
    vectors, sample2id, samples_list = load_vector_with_type(sys.argv[1], "condition", int(sys.argv[3]))
    cluster2samples, samples2cluster, samples2clusterlist = load_sample2cluster(sys.argv[2])
    print_query_under_cluster(cluster2samples, samples_list)

main()
