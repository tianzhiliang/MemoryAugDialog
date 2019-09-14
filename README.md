# MemoryAugDialog
This is the accompany python package for the ACL 2019 paper：Learning to Abstract for Memory-augmented Conversational Response Generation (https://www.aclweb.org/anthology/P19-1371)


### Requirements
- python
- pytorch
- torchtext
- argparse
- codecs
- shutil
- re
- progressbar
- collections
- sklearn
- gensim

### Configuration
`./config.yml` is a configuration file, which contains configurations of model, training, and tesing.

#### 1.Preprocessing

Edit the script file `build_vocab.sh`.

```
sh scripts/build_vocab.sh

```

| parameter     | description |
|---            |--- |
| -train_data FILE |  Training Set (Source + Target)|
| -save_data STR  |  the Prefix of Output File Name |
| -config FILE    |  Configuration File |


#### 2.Training

```
sh run.sh

```

| parameter     | description |
|---            |---          |
| -gpuid INT    |  Choose Which GPU A Program Uses |
| -config FILE  |  Configuration File |
| -config_with_loaded_model PATH |  The config file when loading a warmstart model with its config |
| -config_from_local_or_loaded_model BOOL |   load warmstart model from local path or path mentioned in condig file  |
| -train_data FILE |   input of Training |
| -out_dir FILE |   Output model dir  |
| -vocab FILE     |   Vocabulary  (output from Preprocessing)     |


#### 3.Testing
```
sh generation.sh

```

| parameter     | description |
|---            |--- |
| -gpuid INT    |  Choose Which GPU A Program Uses |
| -test_data FILE  |  test file |
| -test_out FILE |  output file    |
| -config FILE  |  Configuration File |
| -config_with_loaded_model PATH |  The config file when loading a warmstart model with its config |
| -config_from_local_or_loaded_model BOOL |   load warmstart model from local path or path mentioned in condig file  |
| -model FILE   |  load existing model |
| -vocab FILE     |  Vocabulary |
| -beam_size INT |  size for beam search |
| -decode_max_length INT|   |
