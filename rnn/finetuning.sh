net="LSTMPB"
log_path="./result/result200706_i12_epoch10000/finetuning_${net}/"
load_path="./result/result200706_i12_epoch10000/$net/model/"
data_path="./data/20200706_i12/"
save_path=${log_path}"model/"

mkdir -p $log_path
mkdir $save_path

cp finetuning.sh $log_path

python main.py \
--net $net \
--unit_nums 100 \
--batch_size 64 \
--pb_dims 7 \
--epoch 4000 \
--lr 0.001 \
--in_dims  12 \
--out_dims 12 \
--model_save_path $save_path \
--data_path $data_path \
--mode "train" \
--log_path $log_path \
--closed_rate 8 \
--closed_rate 0.8 \
--closed_step 0 \
--log_iter 200 \
--model_save_iter 200 \
--model_load_path $load_path \
--tensorboard true





