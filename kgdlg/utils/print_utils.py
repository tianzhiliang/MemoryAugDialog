import torch
import sys

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def myMatrixDivVector(matrix, vector):
    """
       matrix(N,M) / vector(N) = matrix(N,M)
       for each i,j: 
           matrix_result[i][j] = matrix_source[i][j] / vector[i]
    """
    duplicate_size = matrix.size()[-1]
    vector_duplicate = vector.repeat(duplicate_size, 1).permute(1, 0)
    matrix = matrix / vector_duplicate
    return matrix

def sum_with_axis(input, axes, keepdim=False):
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input

def numpy_array_to_str(a, precision=6):
    return " ".join(list(map(str, round_for_list(a.tolist(), precision))))

def numpy_array_to_str_with_index(a, precision=6):
    a = round_for_list(a, precision)
    a_with_index = [str(i) + ":" + str(aa) for i, aa in enumerate(a)]
    return " ".join(a_with_index)

def print_nn_module_model(model, debug_level=1, print_transpose=False, precision=6):
    """
        Input model must be two-level nn.Module model
        debug_level: 1: structure (name, size, type requires_grad) [default]
                     2: 1 + a few parameters
    """

    if debug_level < 1:
        return

    for i, m in enumerate(model._modules.keys()):
        print("i:", i, "module name:", m)
        for j, l in enumerate(model._modules[m]._modules.keys()):
            print("\tj:", j, "layer name:", l, "layer:", \
                model._modules[m]._modules[l])
            for k, p in enumerate(model._modules[m]._modules[l].parameters()):
                print("\t\tk:", k, "parameter shape:", p.shape, \
                    "parameter requires_grad:", p.requires_grad)
            print("")
        print("")
    print("")

    if debug_level >= 2:
        for i, child in enumerate(model.children()):
            print("i:", i, "child:", child)
            for j, p in enumerate(child.parameters()):
                print("j:", j, "parameters:", p)
            print("")
        print("")

    def _print_1D_or_2D_tensor(t, with_embedding = True, print_transpose = False, precision = 4):
        tshape = t.shape
        if 0 == len(tshape):
            return
        elif len(tshape) == 2:
            if tshape[0] > tshape[1] and tshape[0] >= 15000: # Notice, hard code!!!
                if not with_embedding:
                    return
            if print_transpose:
                t = torch.t(t)
            for tt in t:
                print("\t\t\t" + numpy_array_to_str(tt.detach().numpy(), precision))
        elif len(tshape) == 1:
            print("\t\t\t" + numpy_array_to_str(t.detach().numpy(), precision))
        
    if debug_level >= 3:
        for i, m in enumerate(model._modules.keys()):
            print("i:", i, "module name:", m)
            for j, l in enumerate(model._modules[m]._modules.keys()):
                print("\tj:", j, "layer name:", l, "layer:", \
                    model._modules[m]._modules[l])
                for k, p in enumerate(model._modules[m]._modules[l].parameters()):
                    print("\t\tk:", k, "parameter shape:", p.shape, \
                        "parameter requires_grad:", p.requires_grad)
                    if debug_level == 3:
                        _print_1D_or_2D_tensor(p, False, print_transpose, precision)
                    elif debug_level >= 4:
                        _print_1D_or_2D_tensor(p, True, print_transpose, precision)
                print("")
            print("")
        print("")

def print_matrix(x, filestream=sys.stderr):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    ''' 
    There are some attempts. x.datacpu().numpy() also works.
    print("x size:", x.size()[0])
    print("x:", x)
    print("x data:", x[0].data) # do not change
    print("x[0][0] data0:", x[0][0].data[0]) # do not change
    print("x[0][0].data.cpu():", x[0][0].data.cpu()) # do not change much
    print("x[0][0].data.cpu().numpy():", x[0][0].data.cpu().numpy()) # dealed!
    print("x[0].data.cpu().numpy():", x[0].data.cpu().numpy()) # dealed!
    print("x.data.cpu().numpy():", x.data.cpu().numpy()) # dealed!
    print("x.data.tolist:", x.data.tolist()) # dealed!
    print("x.data.tolist[0]:", x.data.tolist()[0]) # dealed!
    '''
    print("x:", x)
    print("x.data.tolist:", x.data.tolist()) # dealed!
    print("raw:", raw, "col:", col, "len(x_list)", len(x_list), "len(x_list[0]) and 1:", len(x_list[0]), len(x_list[1]))
    for i in range(raw):
        assert len(x_list[i]) == col
        print(" ".join(list(map(str, x_list[i]))), file=filestream)
    print("", file=filestream)

def round_for_list(x, precision):
    return [round(data, precision) for data in x]

def print_matrix_with_ids(x, ids, precision, filestream=sys.stderr):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    ids_list = ids.data.tolist()
    #print("text in print_matrix_with_text:", text)
    #print("len(text) in print_matrix_with_text:", len(text))
    #print("len(text[3]) in print_matrix_with_text:", len(text[3]))
    #print("len(x[3]) in print_matrix_with_text:", len(x[3]))
    #print("len(x) in print_matrix_with_text:", len(x))
    for i in range(raw):
        #print("col:", col, "len(x_list[i]):", len(x_list[i]))
        assert len(x_list[i]) == col
        print(" ".join(list(map(str, ids_list[i]))) + "\t" + " ".join(list(map(str, round_for_list(x_list[i], precision)))), file=filestream)
    print("", file=filestream)

def ids2words(ids, dict):
    return [dict[id] for id in ids]

def print_matrix_with_text(x, ids, dict, precision, filestream=sys.stderr, mark=""):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    ids_list = ids.data.tolist()
    for i in range(raw):
        #print("col:", col, "len(x_list[i]):", len(x_list[i]))
        assert len(x_list[i]) == col
        if "" != mark:
            mark_str = mark + "\t"
        print(mark_str + " ".join(list(map(str, ids2words(ids_list[i], dict)))) + "\t" + " ".join(list(map(str, round_for_list(x_list[i], precision)))), file=filestream)
    print("", file=filestream)

def print_matrix_with_text_with_addition_text(x, ids, dict, precision, \
        add_text, add_dict, filestream=sys.stderr, mark=""):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    ids_list = ids.data.tolist()
    add_text_list = add_text.data.tolist()
    for i in range(raw):
        #print("col:", col, "len(x_list[i]):", len(x_list[i]))
        assert len(x_list[i]) == col
        if "" != mark:
            mark_str = mark + "\t"
        else:
            mark_str = ""
        print(mark_str + " ".join(list(map(str, ids2words(ids_list[i], dict)))) \
            + "\t" + " ".join(list(map(str, round_for_list(x_list[i], precision)))) \
            + "\t" + " ".join(list(map(str, ids2words(add_text_list[i], add_dict)))) \
            , file=filestream)
    print("", file=filestream)

def print_matrix_pair_with_text(xs, ids, xt, idt, sdict, tdict, precision, filestream=sys.stderr, mark=""):
    raw = xs.size()[0]
    col = xs.size()[1]
    assert raw == xt.size()[0] 
    assert col == xt.size()[1]
    xs_list = xs.data.tolist()
    ids_list = ids.data.tolist()
    xt_list = xt.data.tolist()
    idt_list = idt.data.tolist()

    for i in range(raw):
        #print("col:", col, "len(x_list[i]):", len(x_list[i]))
        assert len(xs_list[i]) == col
        assert len(xt_list[i]) == col
        if "" != mark:
            mark_str = mark + "\t"
        print(mark_str + \
            " ".join(list(map(str, ids2words(ids_list[i], sdict))))\
            + "\t" + " ".join(list(map(str, \
            round_for_list(xs_list[i], precision)))) + "\t" + \
            " ".join(list(map(str, ids2words(idt_list[i], tdict))))\
            + "\t" + " ".join(list(map(str, \
            round_for_list(xt_list[i], precision)))) \
            , file=filestream)
    print("", file=filestream)

def save_latent_Z_with_text(src_txt, src_z, tgt_txt, tgt_z, \
        opt, precision=4, filestream=sys.stderr, mark=""):
    if 0 == opt.save_z_and_sample:
        return

    if 1 == opt.save_z_and_sample:
        if src_txt is None or src_z is None:
            return
        src_txt_t = torch.transpose(src_txt, 0, 1)
        print_matrix_with_text(src_z, src_txt_t, \
            opt.variable_src_dict.itos, precision, filestream, mark)
    elif 2 == opt.save_z_and_sample:
        if tgt_txt is None or tgt_z is None:
            return
        src_tgt_t = torch.transpose(tgt_txt, 0, 1)
        print_matrix_with_text(tgt_z, tgt_txt_t, \
            opt.variable_tgt_dict.itos, precision, filestream, mark)
    elif 3 == opt.save_z_and_sample:
        if src_txt is None or src_z is None \
            or tgt_txt is None or tgt_z is None:
            return
        src_txt_t = torch.transpose(src_txt, 0, 1)
        tgt_txt_t = torch.transpose(tgt_txt, 0, 1) 
        print_matrix_pair_with_text(src_z, src_txt_t, \
            tgt_z, tgt_txt_t, opt.variable_src_dict.itos, \
            opt.variable_tgt_dict.itos, precision, filestream, mark)
    elif 4 == opt.save_z_and_sample:
        if src_txt is None or src_z is None \
            or tgt_txt is None:
            return
        src_txt_t = torch.transpose(src_txt, 0, 1)
        tgt_txt_t = torch.transpose(tgt_txt, 0, 1) 
        print_matrix_with_text_with_addition_text(src_z, src_txt_t, \
            opt.variable_src_dict.itos, precision, \
            tgt_txt_t, opt.variable_tgt_dict.itos, filestream, mark)
    elif 5 == opt.save_z_and_sample:
        if src_txt is None or tgt_z is None \
            or tgt_txt is None:
            return
        src_txt_t = torch.transpose(src_txt, 0, 1)
        tgt_txt_t = torch.transpose(tgt_txt, 0, 1) 
        print_matrix_with_text_with_addition_text(tgt_z, src_txt_t, \
            opt.variable_src_dict.itos, precision, \
            tgt_txt_t, opt.variable_tgt_dict.itos, filestream, mark)


