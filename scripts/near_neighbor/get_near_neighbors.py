from sklearn.cluster import KMeans
import numpy as np
import sys

#from near_neighbors import near_neighbor
import kgdlg.utils.operation_utils as operation_utils
import kgdlg.utils.print_utils as print_utils
#from near_neighbors import get_vectorwise_nearest_neighbor_of_two_matrix
#from near_neighbors import get_vectorwise_neighbors_of_two_matrix
import sklearn.metrics.pairwise

is_near_neighbor = True

def load_vector(file):
    f = open(file, "r")
    vectors = []
    sample2id = {}
    samples_list = []
    count = 0

    for line in f:
        slots = line.strip().split("\t")
        if len(slots) != 2:
            continue
        words_str, vec_str = slots
        words = words_str.split()
        vec = list(map(float, vec_str.split()))
   
        vectors.append(vec) 
        samples_list.append(words_str)  
        sample2id[words_str] = count
        count += 1
    return vectors, sample2id, samples_list  

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

def load_vector_with_type_pairwise(file, type, sample_size):
    f = open(file, "r")
    src_vecs = []
    tgt_vecs = []
    sample2id = {}
    samples_list = []
    count = 0

    for line in f:
        slots = line.strip().split("\t")
        if len(slots) != 5:
            continue
        if slots[0] != type:
            continue
        src_str, src_vec, tgt_str, tgt_vec = slots[1:]

        src = src_str.split()
        tgt = tgt_str.split()
        src = src[:src.index("</s>")]
        try:
            tgt = tgt[:tgt.index("</s>")]
        except:
            pass
        src_str = " ".join(src)
        tgt_str = " ".join(tgt)

        src_vec = list(map(float, src_vec.split()))
        tgt_vec = list(map(float, tgt_vec.split()))
        src_vecs.append(src_vec) 
        tgt_vecs.append(tgt_vec) 
        samples_list.append([src_str, tgt_str])  
        count += 1
        if count >= sample_size:
            break

    print("load_vector_with_type done. size:", len(samples_list))

    src_vecs = np.array(src_vecs)
    tgt_vecs = np.array(tgt_vecs)
    return src_vecs, tgt_vecs, samples_list

def load_src_tgt_mapping(file, sample_size=-1):
    f = open(file, "r")
    src2tgt = {}
    tgt2src = {}
    src_tgt_list = []

    cnt = 0
    for line in f:
        slots = line.strip().split("\t")
        if 2 != len(slots):
            continue
        src, tgt = slots
        if not src in src2tgt:
            src2tgt[src] = []
        if not tgt in tgt2src:
            tgt2src[tgt] = []
        src2tgt[src].append(tgt)
        tgt2src[tgt].append(src)
        src_tgt_list.append([src, tgt])
        cnt += 1
        if cnt >= sample_size and -1 != sample_size:
            break
    f.close()

    print("load_src_tgt_mapping done. source num:", len(src2tgt.keys()), \
        "target num:", len(tgt2src.keys()), "pair num:", len(src_tgt_list))
    return src2tgt, tgt2src, src_tgt_list

def uniq(list):
    uniqlist = []
    last = ""
    for id, sample in list:
        if sample == last:
            continue
        uniqlist.append([id, sample])
        last = sample
    return uniqlist

def uniq_on_samples(vectors, samples_list):
    samples_list_id = [(id, sample) for id, sample in enumerate(samples_list)]
    samples_sorted = sorted(samples_list_id, key=lambda x:x[1])
    samples_uniq_with_id = uniq(samples_sorted)

    new_vectors = []
    new_vectors = [vectors[i] for i, sample in samples_uniq_with_id]

    sample2id = {}
    for i, sample in samples_uniq_with_id:
        sample2id[sample] = i

    samples_uniq = [sample for i, sample in samples_uniq_with_id]

    return new_vectors, sample2id, samples_uniq

def cluster(vectors, cluster_num):
    p = KMeans(n_clusters=cluster_num, init="k-means++", n_init=10, max_iter=30, tol=0.0001, precompute_distances="auto", verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm="auto")
    model = p.fit(vectors)
    #print(model.labels_)
    return model

def output(model, samples_list, bysample = False, bycluster = True):
    labels = model.labels_
    assert len(labels) == len(samples_list)
    labels_with_sent = []
    for label, sent in zip(labels, samples_list):
        if bysample:
            print(str(label) + "\t" + sent)
        if bycluster:
            labels_with_sent.append([label, sent])
    if bycluster:
        labels_with_sent = sorted(labels_with_sent, key = lambda x:x[0])
        for label, sent in labels_with_sent:
            print(str(label) + "\t" + sent)

#def numpy_array_to_str(a, precision = 6):
#    return " ".join(list(map(str, print_utils.round_for_list(a.tolist(), precision))))

def filter_symbols(s, symbols):
    ss = s.split()
    res = []
    for w in ss:
        if not w in symbols:
            res.append(w)
    return " ".join(res)

def get_nearest_neighbor_and_print(train_vecs, test_vecs, \
        train_samples, test_samples, output_file, src2tgt):
    sim_type = "cosine" # dot_product is not good
    max_sim_index, max_sim = operation_utils.get_vectorwise_nearest_neighbor_of_two_matrix( \
        test_vecs, train_vecs, sim_type)

    f = open(output_file, "w")
    for i, (index, sim, vec, sample) \
        in enumerate(zip(max_sim_index, max_sim, test_vecs, test_samples)):
            if train_samples[index] in src2tgt:
                tgts = "\t".join(src2tgt[train_samples[index]])
            else:
                tgts = ""
            res = sample + "\t" + print_utils.numpy_array_to_str(vec) + "\t" \
                + "\t" + print_utils.numpy_array_to_str(train_vecs[index]) + "\t" \
                + str(index) + "\t" + str(sim) + "\t" \
                + str(train_samples[index]) + "\t" + str(tgts) + "\n"
            f.write(res)
    f.close()

def pair_src_and_tgt_of_nearest_neighbor(train_src_vecs, train_tgt_vecs,\
        test_src_vecs, test_tgt_vecs, train_samples, test_samples, \
        output_file, search_by_source_or_target):
    sim_type = "cosine" # dot_product is not good

    if "source" == search_by_source_or_target:
        max_sim_index, max_sim = operation_utils.get_vectorwise_nearest_neighbor_of_two_matrix( \
                test_src_vecs, train_src_vecs, sim_type)
    elif "target" == search_by_source_or_target:
        max_sim_index, max_sim = operation_utils.get_vectorwise_nearest_neighbor_of_two_matrix( \
                test_tgt_vecs, train_tgt_vecs, sim_type)

    f = open(output_file, "w")
    for i, (index, sim, test_src_v, test) \
        in enumerate(zip(max_sim_index, max_sim, test_src_vecs, test_samples)):
            print("test_src_v shape:", test_src_v.shape)
            print("train_src_vecs[index] shape:", train_src_vecs[index].shape, "index:", index)
            sim2 = sklearn.metrics.pairwise.cosine_similarity([test_src_v], [train_src_vecs[index]])
            f.write(test[0] + "\t" + print_utils.numpy_array_to_str(test_src_v) + "\t" \
                + test[1] + "\t" + print_utils.numpy_array_to_str(train_tgt_vecs[index]) \
                + "\t" + str(index) + "\t" + str(sim) \
                + "\t" + str(sim2) + "\t" + print_utils.numpy_array_to_str(train_src_vecs[index]) \
                + "\t" + "\t".join(train_samples[index]) + "\n")
    f.close()

def get_topK_near_neighbor_and_print(train_vecs, test_vecs, train_samples, test_samples, \
        output_file, topK):
    sim_type = "cosine" # dot_product is not good
    topk_indexs, topk_values = operation_utils.get_vectorwise_neighbors_of_two_matrix( \
        test_vecs, train_vecs, topK, sim_type)


    f = open(output_file, "w")
    for i, (k_index, k_sim, vec, sample) \
        in enumerate(zip(topk_indexs, topk_values, test_vecs, test_samples)):
            output = [str(i) + "\t" + str(s) + "\t" + train_samples[i] for i, s in zip(k_index, k_sim)]
            f.write(sample + "\t" + print_utils.numpy_array_to_str(vec) + "\t" \
                    "\n" + "\n".join(output) + "\n")
    f.close()

def pair_src_and_tgt_of_topK_near_neighbor(train_src_vecs, train_tgt_vecs,\
        test_src_vecs, test_tgt_vecs, train_samples, test_samples, \
        output_file, topK, search_by_source_or_target):
    sim_type = "cosine" # dot_product is not good

    if "source" == search_by_source_or_target:
        topk_indexs, topk_values = operation_utils.get_vectorwise_neighbors_of_two_matrix( \
                test_src_vecs, train_src_vecs, topK, sim_type)
    elif "target" == search_by_source_or_target:
        topk_indexs, topk_values = operation_utils.get_vectorwise_neighbors_of_two_matrix( \
                test_tgt_vecs, train_tgt_vecs, topK, sim_type)

    f = open(output_file, "w")
    for i, (k_index, k_sim, test_src_v, test) \
        in enumerate(zip(topk_indexs, topk_values, test_src_vecs, test_samples)):
            output = [str(i) + "\t" + str(s) + "\tSrc:\t" + train_samples[i][0] + "\tTarget:\t" + train_samples[i][1] for i, s in zip(k_index, k_sim)]
            #output = [str(i) + "\t" + str(s) + "\t" + "\t".join(train_samples[i]) for i, s in zip(k_index, k_sim)]
            f.write(test[0] + "\t" + print_utils.numpy_array_to_str(test_src_v) + "\t" \
                + test[1] + "\n" + "\n".join(output) + "\n")
    f.close()

def main():
    training_z_file = sys.argv[1]
    testing_file = sys.argv[2]
    output_file = sys.argv[3]
    training_sample_size = int(sys.argv[4])
    testing_sample_size = int(sys.argv[5])
    topK = int(sys.argv[6])
    search_by_source_or_target = "source" # source / target [default: source]
    search_by_source_or_target = sys.argv[7]
    is_pairwise_data = int(sys.argv[8]) # 1:pairwise_data 0:single_data
    type_name = sys.argv[9] # src_tgt, condition, input

    src_tgt_mapping_file = ""
    src2tgt = None
    if not "" == src_tgt_mapping_file:
        #sample_size = training_sample_size
        #if testing_sample_size > training_sample_size:
        #    sample_size = testing_sample_size

        src2tgt, tgt2src, src_tgt_list = \
            load_src_tgt_mapping(src_tgt_mapping_file)

    if 1 == is_pairwise_data:
        train_src_vecs, train_tgt_vecs, train_samples \
            = load_vector_with_type_pairwise(training_z_file, type_name, training_sample_size)
        test_src_vecs, test_tgt_vecs, test_samples \
            = load_vector_with_type_pairwise(testing_file, type_name, testing_sample_size)

        if is_near_neighbor:
            if 1 == topK:
                pair_src_and_tgt_of_nearest_neighbor(train_src_vecs, \
                    train_tgt_vecs, test_src_vecs, test_tgt_vecs, train_samples, \
                    test_samples, output_file, search_by_source_or_target)
            else:
                pair_src_and_tgt_of_topK_near_neighbor(train_src_vecs, \
                    train_tgt_vecs, test_src_vecs, test_tgt_vecs, train_samples, \
                    test_samples, output_file, topK, search_by_source_or_target)
        else:
            K = int(sys.argv[3])
            model = cluster(vectors, L)
            output(model, samples_list)
    elif 0 == is_pairwise_data:
        train_vecs, train_sampledict, train_samples = load_vector_with_type( \
                training_z_file, type_name, training_sample_size)
        test_vecs, test_sampledict, test_samples = load_vector_with_type( \
                testing_file, type_name, testing_sample_size)

        train_samples = [filter_symbols(sample, ["<s>","</s>"]) for sample in train_samples]
        test_samples = [filter_symbols(sample, ["<s>","</s>"]) for sample in test_samples]

        if is_near_neighbor:
            get_topK_near_neighbor_and_print(train_vecs, test_vecs, \
                train_samples, test_samples, output_file, topK)
#            if 1 == topK:
#get_nearest_neighbor_and_print(train_vecs, test_vecs, \
#                    train_samples, test_samples, output_file, src2tgt)
#            else:

main()
