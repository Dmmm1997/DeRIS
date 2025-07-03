# source_dir=configs/gres/sota
# file_names=(gref_sota_swinbase384_beitlarge224_ep6_nq20_mt50_rate0.15_mint0_simthr_0.6.py) 
# for file_name in "${file_names[@]}"
# do
#   related_filename=$source_dir/$file_name
#   # train
#   CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29511 bash tools/dist_train.sh $related_filename 4

#   # test -----
#   # basename without .py
#   file_name_without_suffix=$(basename "$related_filename" .py)
#   file_dir_suffix=$source_dir/$file_name_without_suffix
#   checkpoint_dir=$(echo "$file_dir_suffix" | sed 's/configs/work_dir/g')
#   latest_folder=$(ls -t "$checkpoint_dir" | head -n 1)
#   echo $latest_folder
#   checkpoint=$checkpoint_dir/$latest_folder/latest.pth
#   echo $checkpoint
#   # CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29522 bash tools/dist_test.sh $related_filename 4 --load-from $checkpoint --score-threshold 0.9
#   # CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29522 bash tools/dist_test.sh $related_filename 4 --load-from $checkpoint --score-threshold 0.8
#   # CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29522 bash tools/dist_test.sh $related_filename 4 --load-from $checkpoint --score-threshold 0.7
#   #test -----
# done

CUDA_VISIBLE_DEVICES=2,3 PORT=29402 bash tools/dist_test.sh configs/gres/sota/gref_sota_swinbase384_beitlarge224_ep6_nq20_mt50_rate0.15_mint0_simthr_0.6.py 2 --load-from work_dir/gres/sota/gref_sota_swinbase384_beitlarge224_ep3_nq20_mt50_rate0.15_mint0_simthr_0.6/20250227_114627/lightweight_model/compressed.pth --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0,1 PORT=29522 bash tools/dist_test.sh configs/gres/sota/gref_sota_swinsmall384_beitbase224_ep6_nq20_mt50_rate0.15_mint0_simthr_0.6.py 2 --load-from work_dir/gres/sota/gref_sota_swinsmall384_beitbase224_ep6_nq20_mt50_rate0.15_mint0_simthr_0.6/DERIS-B-grefcoco/compressed.pth --score-threshold 0.7
CUDA_VISIBLE_DEVICES=0 PORT=29520 bash tools/dist_test.sh configs/refcoco/sota/ref_sota_swinsmall384_beitbase224_ep20_nq10_mt20.py 1 --load-from work_dir/refcoco/sota/ref_sota_swinsmall384_beitbase224_ep20_nq10_mt20/DeRIS-B-refcoco-swin384-beit224/compressed.pth --score-threshold 0.7
CUDA_VISIBLE_DEVICES=1 PORT=29519 bash tools/dist_test.sh configs/refcoco/sota/ref_sota_swinbase384_beitlarge224_ep20_nq10_mt20.py 1 --load-from work_dir/refcoco/sota/ref_sota_swinbase384_beitlarge224_ep20_nq10_mt20/20250226_095740/lightweight_model/compressed.pth --score-threshold 0.7

