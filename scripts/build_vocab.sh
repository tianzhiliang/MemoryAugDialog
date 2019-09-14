mark=v1
train_data=data/train.txt

mkdir -p log
log=log/log_build_vocab_$mark
err=log/err_build_vocab_$mark

alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'
pythont build_vocab.py \
    -train_data $train_data \
    -save_data dict_v1 \
    -config ./config.yml > ${log} 2> ${err}
