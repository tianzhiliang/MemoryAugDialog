mark=$1
GPUID=1

data_path=data/
train_data=${data_path}/train.txt
vocab_file=${data_path}/dict_v1.vocab.pt

config_from_local_or_loaded_model=1 # 0:local(from ./config.yml) 1:loaded model(from output_model/${modelMark}/config.yml)

mkdir -p log
log=log/log_$mark
err=log/err_$mark
model_output_dir=output_models/${mark}

python train.py \
    -gpuid ${GPUID} \
    -config ./config.yml \
    -config_with_loaded_model ./config.yml \
    -config_from_local_or_loaded_model ${config_from_local_or_loaded_model} \
    -train_data $train_data \
    -out_dir $model_output_dir \
    -vocab ${vocab_file} > ${log} 2> ${err}
