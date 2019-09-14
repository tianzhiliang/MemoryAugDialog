mark=$1
modelID=$2
generate_mark=_$3
if [ $# -eq 2 ]; then
generate_mark=""
fi

GPUID=1
data_path=data/
vocab_file=${data_path}/dict_v1.vocab.pt 

test_src_data=$data_path/test_src.txt
test_tgt_data=$data_path/test_tgt.txt
test_src_tgt_data=$data_path/test_src_tgt.txt
test_src_data_uniq=$data_path/test_src_tgt_uniq.txt

decode_max_length=50
beam_size=10
config_from_local_or_loaded_model=1 # 0:local(from ./config.yml) 1:loaded model(from output_model/${modelMark}/config.yml)

mkdir -p log
mkdir -p result
output_model_dir=output_models/$mark/
model=${output_model_dir}/checkpoint_epoch${modelID}.pkl
config=${output_model_dir}/config.yml
log=log/log_gene_${mark}_tdv22_epoch${modelID}${generate_mark}
err=log/err_gene_${mark}_tdv22_epoch${modelID}${generate_mark}
res=result/res_gene_${mark}_tdv22_epoch${modelID}${generate_mark}

PYTHON=python

function inference_on_uniq_data() {
${PYTHON} inference.py \
    -gpuid ${GPUID} \
    -test_data $test_src_data_uniq \
    -test_out $res \
    -config ./config.yml \
    -config_with_loaded_model ${config} \
    -config_from_local_or_loaded_model ${config_from_local_or_loaded_model} \
    -model $model \
    -beam_size ${beam_size} \
    -decode_max_length ${decode_max_length} \
    -vocab $vocab_file > ${log} 2> ${err}
}

inference_on_uniq_data
