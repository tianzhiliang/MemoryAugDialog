import kgdlg.IO
import argparse
import kgdlg.utils.misc_utils as utils
import torch
import json
import kgdlg
parser = argparse.ArgumentParser()
parser.add_argument('-train_data', type=str)
parser.add_argument('-save_data', type=str)
parser.add_argument('-config', type=str)
args = parser.parse_args()

opt = utils.load_hparams(args.config)

if opt.random_seed > 0:
    torch.manual_seed(opt.random_seed)

fields = kgdlg.IO.get_fields()
print("Building Training...")
train = kgdlg.IO.TrainDataset(
    data_path=args.train_data,
    fields=[('src', fields["src"]),
            ('tgt', fields["tgt"])])    
print("Building Vocab...")   
kgdlg.IO.build_vocab(train, opt)

print("Saving fields")
torch.save(kgdlg.IO.save_fields_to_vocab(fields),open(args.save_data+'.vocab.pt', 'wb'))
