queryvector=log/query_vectors
sample2cluster=log/new_vectors_for_testing__between_cluster_center_source_train1000_test1651243_top1.txt.clusterid
queryvectorsize=1000000

python get_cluster_all.py ${queryvector} ${sample2cluster} ${queryvectorsize} > ${queryvector}.get_cluster_all 2>  ${queryvector}.get_cluster_all_errlog
