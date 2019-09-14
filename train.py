import kgdlg.utils.misc_utils as misc_utils
import kgdlg.utils.print_utils as print_utils
import argparse
import codecs
import os
import shutil
import re
import torch
import torch.nn as nn
from torch import cuda
import kgdlg
import random
parser = argparse.ArgumentParser()
parser.add_argument("-config", type=str)
parser.add_argument("-config_with_loaded_model", type=str)
parser.add_argument("-config_from_local_or_loaded_model", type=int)
parser.add_argument("-vocab", type=str)
parser.add_argument('-train_data', type=str)
parser.add_argument('-gpuid', default=[], nargs='+', type=int)
parser.add_argument('-out_dir', type=str)

args = parser.parse_args()
if 1 == args.config_from_local_or_loaded_model: # loaded model (from out_dir)
    opt = misc_utils.load_hparams(args.config_with_loaded_model)
elif 0 == args.config_from_local_or_loaded_model: # local (./config.yml)
    opt = misc_utils.load_hparams(args.config)
opt.out_dir = args.out_dir

if args.gpuid:
    if -1 == int(args.gpuid[0]):
        device = None
        opt.use_cuda = False
        opt.cluster_param_in_cuda = 0
    else:
        cuda.set_device(int(args.gpuid[0]))
        device = torch.device('cuda',int(args.gpuid[0]))
        opt.gpuid = int(args.gpuid[0])
        opt.use_cuda = True
    
if opt.random_seed > 0:
    random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
def report_func(global_step, epoch, batch, num_batches,
                start_time, lr, report_stats):
    """
    This is the user-defined batch-level traing progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % opt.steps_per_stats == -1 % opt.steps_per_stats:
        report_stats.print_out(epoch, batch+1, num_batches, start_time)
        report_stats = kgdlg.Statistics()

    return report_stats




def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    return kgdlg.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.train_batch_size,
                device=device,                
                repeat=False,
                sort=False)


def load_fields_from_vocab():
    fields = kgdlg.IO.load_fields_from_vocab(
                torch.load(args.vocab))
    fields = dict([(k, f) for (k, f) in fields.items()])


    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def build_or_load_model(model_opt, fields):
    # model = build_model(model_opt, fields)
    model = kgdlg.ModelConstructor.create_base_model(model_opt, fields) # Main Step
    latest_ckpt = misc_utils.latest_checkpoint(model_opt.out_dir)
    start_epoch_at = 0 # Start from sepcific checkpoint or last checkpoint
    if model_opt.start_epoch_at is not None:
        ckpt = 'checkpoint_epoch%d.pkl'%(model_opt.start_epoch_at)
        ckpt = os.path.join(model_opt.out_dir,ckpt)
    else:
        ckpt = latest_ckpt
    # latest_ckpt = misc_utils.latest_checkpoint(model_dir)
    if ckpt: # If there are models in out_dir
        if 1 == model_opt.load_model_mode:
            start_epoch_at = model.load_checkpoint_by_layers(ckpt)
        elif 0 == model_opt.load_model_mode:
            start_epoch_at = model.load_checkpoint(ckpt)
    else:
        print('Start from scratch to build model...') # If nothing in out_dir

    print(model)
    print_utils.print_nn_module_model(model)

    if "" != model_opt.freeze_modules_list:
        model.freeze_modules(model_opt.freeze_modules_list)
        #if model_opt.debug_mode >= 3:
        #    print("After freeze_modules, model becomes:", model)
        
    return model, start_epoch_at


def build_optim(model, optim_opt):
    optim = kgdlg.Optim(optim_opt.optim_method, 
                  optim_opt.learning_rate,
                  optim_opt.max_grad_norm,
                  optim_opt.learning_rate_decay,
                  optim_opt.weight_decay,
                  optim_opt.start_decay_at)
     
    optim.set_parameters(model.parameters())
    return optim

def build_lr_scheduler(optimizer):

    lr_lambda = lambda epoch: opt.learning_rate_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, 
                                                  lr_lambda=[lr_lambda])
    return scheduler    

def check_save_model_path(opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
        print('saving config file to %s ...'%(opt.out_dir))
        # save config.yml
        shutil.copy(args.config, os.path.join(opt.out_dir,'config.yml'))    



def train_model(model, train_data, fields, optim, lr_scheduler, start_epoch_at):

    train_iter = make_train_data_iter(train_data, opt)

    train_loss = kgdlg.NMTLossCompute(model.generator,fields['tgt'].vocab, opt)
    valid_loss = kgdlg.NMTLossCompute(model.generator,fields['tgt'].vocab, opt) 

    if opt.use_cuda:
        train_loss = train_loss.cuda()
        valid_loss = valid_loss.cuda()    

    shard_size = opt.train_shard_size
    trainer = kgdlg.Trainer(opt, model,
                        train_iter,
                        train_loss,
                        optim,
                        lr_scheduler)

    num_train_epochs = opt.num_train_epochs
    print('start training...')
    for step_epoch in  range(start_epoch_at+1, num_train_epochs):

        if step_epoch >= opt.start_decay_at:
            trainer.lr_scheduler.step()
        # 1. Train for one epoch on the training set.
        train_stats = trainer.train(step_epoch, report_func) # Main Step
        print('Train perplexity: %g' % train_stats.ppl())

        
        #trainer.epoch_step(step_epoch, out_dir=opt.out_dir)


        
def main():

    # Load train and validate data.
    print("Loading fields from '%s'" % args.vocab)

    
    # Load fields generated from preprocess phase.
    fields = load_fields_from_vocab()


    train = kgdlg.IO.TrainDataset(
                    data_path=args.train_data,
                    fields=[('src', fields["src"]),
                            ('tgt', fields["tgt"])])

    # Build model.
    model, start_epoch_at = build_or_load_model(opt, fields)
    check_save_model_path(opt)

    # Build optimizer.
    optim = build_optim(model, opt)
    lr_scheduler = build_lr_scheduler(optim.optimizer)

    if opt.use_cuda:
        model.set_paramater_to_cuda()
    #    model = model.cuda() # Main Set GPU

    # Do training.
    
    train_model(model, train, fields, optim, lr_scheduler, start_epoch_at)

if __name__ == '__main__':
    main()
