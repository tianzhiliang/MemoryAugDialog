input=log/err_dv22v44k_save_z_from_pretrained_s2s_ed32_hd32_ld32_lm1_bs64_h1w
cluster_num=3000
type=condition
output=${input}.cluster_${cluster_num}

python scripts/cluster/cluster.py ${input} ${type} ${cluster_num} > ${output}


