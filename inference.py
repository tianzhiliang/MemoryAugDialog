import torch
import argparse
import kgdlg
import json
from torch import cuda
import progressbar
import kgdlg.utils.misc_utils as utils
import os

def indices_lookup(indices,fields):

    words = [fields['tgt'].vocab.itos[i] for i in indices]
    sent = ' '.join(words)
    return sent


def batch_indices_lookup(batch_indices,fields):

    batch_sents = []
    for sent_indices in batch_indices:
        sent = indices_lookup(sent_indices,fields)
        batch_sents.append(sent)
    return batch_sents



def inference_file(translator, 
                   data_iter, 
                   test_out, fields):

    print('start decoding ...')
    with open(test_out, 'w', encoding='utf8') as tgt_file:
        bar = progressbar.ProgressBar()

        for batch in bar(data_iter):
            ret = translator.inference_batch(batch)
            batch_sents = batch_indices_lookup(ret['predictions'][0], fields)
            for sent in batch_sents:
                tgt_file.write(sent+'\n')

def make_test_data_iter(data_path, fields, device, opt):
    if opt.vae_type in [0, 1, 2, 3, 4, 5]: #TODO it is better for 5 to use "tgt" from train dataset 
        test_dataset = kgdlg.IO.InferDataset(
            data_path=data_path,
            fields=[('src', fields["src"])])
    elif opt.vae_type in [6, 7, 8]:
        test_dataset = kgdlg.IO.TrainDataset( # Train Dataset
                data_path=data_path,
                fields=[('src', fields["src"]),
                ('tgt', fields["tgt"])])

    test_data_iter = kgdlg.IO.OrderedIterator(
                dataset=test_dataset, device=device,
                batch_size=1, train=False, sort=False,
                sort_within_batch=True, shuffle=False)
    return test_data_iter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_data", type=str)
    parser.add_argument("-test_out", type=str)
    parser.add_argument("-config", type=str)
    parser.add_argument("-config_with_loaded_model", type=str)
    parser.add_argument("-config_from_local_or_loaded_model", type=int)
    parser.add_argument("-model", type=str)
    parser.add_argument("-vocab", type=str)
    parser.add_argument('-gpuid', default=[], nargs='+', type=int)
    parser.add_argument("-beam_size", type=int)
    parser.add_argument("-decode_max_length", type=int)


    args = parser.parse_args()
    if 1 == args.config_from_local_or_loaded_model: # loaded model (from out_dir)
        opt = utils.load_hparams(args.config_with_loaded_model)
    elif 0 == args.config_from_local_or_loaded_model: # local (./config.yml)
        opt = utils.load_hparams(args.config)
    opt.out_dir = os.path.dirname(args.model)

    if args.gpuid:
        if -1 == int(args.gpuid[0]):
            device = None
            opt.use_cuda = False
            opt.cluster_param_in_cuda = 0
        else:
            cuda.set_device(args.gpuid[0])
            device = torch.device('cuda',args.gpuid[0])
            opt.gpuid = int(args.gpuid[0])
            opt.use_cuda = True
    #print("use_cuda:", use_cuda, "device:", device)

    fields = kgdlg.IO.load_fields_from_vocab(
                torch.load(args.vocab))
    test_data_iter = make_test_data_iter(args.test_data, fields, device, opt)
    model = kgdlg.ModelConstructor.create_base_model(opt,fields)

    print('Loading parameters ...')
    #print('args.model:', args.model)

    if 0 == opt.load_model_mode_for_inference:
        model.load_checkpoint(args.model)
    elif 1 == opt.load_model_mode_for_inference:
        model.load_checkpoint_by_layers(args.model)
    if opt.use_cuda:
        model.set_paramater_to_cuda()
        #model = model.cuda()  # Main Set GPU

    translator = kgdlg.Inferer(model=model, 
                                fields=fields,
                                beam_size=args.beam_size, 
                                opt=opt,
                                n_best=1,
                                max_length=args.decode_max_length,
                                global_scorer=None)
    inference_file(translator, test_data_iter, args.test_out, fields)

if __name__ == '__main__':
    main()
