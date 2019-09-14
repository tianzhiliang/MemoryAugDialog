mark=like_retrieval_find_top1_0303_r1

trainfile=log/err_s2sd620att_rerun_for_paper_baselines_0303.condition.uniq
testfile=log/log_gene_s2sd620att_rerun_for_paper_baselines_0303_tdv22_epoch6.condition
#trainfile=${testfile}
#trainfile=log/err_s2s8ln20d620c1k_cs0p01lmd0embt1att_mlmd10_kld0_bs56smalldict_simmax1dot_vmt5_for_casestudy.deal

#mark=dv22v44k_lm1bs64_s2sae_ln20_edhdld6h_for_save_z_tdv22_epoch20
#testfile=log/log_gene_dv22v44k_lm1bs64_s2sae_ln20_edhdld6h_for_save_z_tdv22_epoch20_for_save_z

#mark=dv22v44k_lm1bs64_s2sae_ln20_edhdld6h_generate_on_training_data_for_save_z_tdv22_epoch20
#testfile=log/err_dv22v44k_lm1bs64_s2sae_ln20_edhdld6h_for_save_z

#trainfile=log/err_dv22v44k_lm1bs64_s2sae_ln20_edhdld6h_for_save_z

test_size=919
#train_size=1000000
train_size=1183890
topK=1
#topK=50
search_by_source_or_target=source # source / target [default: source] 
#search_by_source_or_target=target # source / target [default: source] 
is_pairwise_data=0 # 1:pairwise_data 0:single_data
type_name=condition # src_tgt, condition, input
src_tgt_mapping_file=""
#src_tgt_mapping_file=/bigstore/hlcm2/tianzhiliang/paper/vae_seq2seq/data/weibo_huawei_dialog_Dec18/weibo_v2_40k_with_ch/weibo_src_tgt_vocab50k.train
#src_tgt_mapping_file=/bigstore/hlcm2/tianzhiliang/paper/vae_seq2seq/data/weibo_huawei_dialog_Dec18/weibo_v2_40k_with_ch/weibo_src_tgt_5ref.test

mark_new=_${mark}_${search_by_source_or_target}_train${train_size}_test${test_size}_top${topK}
outputfile=log/new_vectors_for_testing_${mark_new}.txt
log=log/log_get_new_vectors_for_testing_${mark_new}
err=log/err_get_new_vectors_for_testing_${mark_new}

alias pythont='/data/tianzhiliang/test/software/anaconda3_1812/bin/python'
#alias pythont='/bigstore/hlcm2/tianzhiliang/test/software/anaconda3_5_1_0_pytorch0_4_1/bin/python'
pythont get_near_neighbors.py ${trainfile} ${testfile} ${outputfile} ${train_size} ${test_size} ${topK} ${search_by_source_or_target} ${is_pairwise_data} ${type_name} ${src_tgt_mapping_file} > ${log} 2> ${err}
