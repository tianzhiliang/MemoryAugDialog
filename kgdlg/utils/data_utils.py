import kgdlg
from torch.autograd import Variable
import torch
# Pad a with the PAD symbol

def pad_seq(seq, max_length, padding_idx):
    seq += [padding_idx for i in range(max_length - len(seq))]
    return seq
    
def get_src_input_seq(seq):
    seq = seq
    return seq

def seq2indices(seq, word2index, max_len=None):
    seq_idx = []
    words_in = seq.split(' ')
    if max_len is not None:
        words_in = words_in[:max_len]
    for w in words_in:
        seq_idx.append(word2index[w])

    return seq_idx

def batch_seq2var(batch_src_seqs, word2index, use_cuda=True):
    src_seqs = [seq2indices(seq, word2index) for seq in batch_src_seqs]
    src_seqs = sorted(src_seqs, key=lambda p: len(p), reverse=True)
    src_inputs = [get_src_input_seq(s) for s in src_seqs]
    src_input_lengths = [len(s) for s in src_inputs]
    paded_src_inputs = [pad_seq(s, max(src_input_lengths), word2index[kgdlg.IO.PAD_WORD]) for s in src_seqs]    
    src_input_var = Variable(torch.LongTensor(paded_src_inputs), volatile=True).transpose(0, 1)
    if use_cuda:
        src_input_var = src_input_var.cuda() 
    return src_input_var, src_input_lengths

def ids2str(ids):
    return " ".join(list(map(str, ids)))

def indices2words(idxs, index2word):
    words_list = [index2word[idx] for idx in idxs]
    return words_list

def wordID2word(id, id2word):
    if 0 <= id and id < len(id2word):
        return id2word[id]
    else:
        return ""

def wordIDs2words(ids, id2word):
    return [wordID2word(id, id2word) for id in ids]

def wordIDsList2wordsList(ids_list, id2word):
    '''
        input: ids_list [sentence_number, sequence_length]
            means a list of sentences, each sentence contains multiple words
        output: words_list [sentence_number, sequence_length]
            every words_list[i][j] is j-th word in i-th sentence
    '''
    return [wordIDs2words(ids, id2word) for ids in ids_list]

def wordIDsList2wordstrList(ids_list, id2word):
    '''
        input: ids_list [sentence_number, sequence_length]
            means a list of sentences, each sentence contains multiple words
        output: words_list [sentence_number]
            every words_list[i] is i-th sentence in string format
    '''
    return [" ".join(list(map(str, wordIDs2words(ids, id2word)))) for ids in ids_list]

'''
def expend_small_to_large(a, b, axis):
    shape_a = a.shape
    shape_b = b.shape
    len_a = shape_a[axis]
    len_b = shape_b[axis]

    if len_a > len_b:
        small = b
        large = a
    else:
        small = a
        large = b

    shape_small = small.shape
    shape_large = large.shape

    shape_small[axis] = shape_large[axis]
    if shape_large != shape_small:
        print("expend_small_to_large error. after expend shape_large:", shape_large, "shape_small:", shape_small)
'''

def char_is_digital(ch):
    ch_asiic = ord(ch)-48
    if 0 <= ch_asiic and ch_asiic <= 9:
        return True
    else:
        return False

def get_epochid_by_filename(filename, suffix = ".pkl"):
    name_body = filename[:filename.find(suffix)]
    name_body_reversed = name_body[::-1]
    epochid_reverse = ""
    for ch in name_body_reversed:
        if not char_is_digital(ch):
            break
        epochid_reverse += ch
    epochid = int(epochid_reverse[::-1])
    return epochid

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

def save_2Dnumpy_as_list_to_txt(f, vecs):
    outdim = len(vecs)
    if 0 == len(vecs):
        return [len(vecs), 0]

    indim = len(vecs[0])
    f.write(str(outdim) + "\t" + str(indim) + "\n")
    for vec in vecs:
        f.write(" ".join(list(map(str, vec.tolist()))) + "\n")
    return [len(vecs), len(vecs[0])]

def test_cases():
    print(get_epochid_by_filename("checkpoint_epoch_by_layers1.pkl"))
    print(get_epochid_by_filename("checkpoint_epoch_by_layers150.pkl"))
    print(get_epochid_by_filename("checkpoint_epoch150ww4.pkl"))
    print(get_epochid_by_filename("checkpoint_epoch150ww45.txt", ".txt"))

#test_cases()
