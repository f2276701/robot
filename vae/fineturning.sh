net="CVAE"
input_mode="goal"
#load_path="./result200522/$net/model/"
load_path="./result/result200706_z7b2/VAE/model/"
log_path="./result/result200706_z7b2/fine-tuning_${net}/"
save_path=${log_path}"model/"
data_path="./data/20200706/"

mkdir -p $log_path
mkdir $save_path

cp fineturning.sh $log_path

python main.py \
--net $net \
--z_dim 20 \
--prior_dist "G" \
--epoch 20 \
--beta 1 \
--lr 0.0005 \
--model_load_path $load_path \
--model_save_path $save_path \
--log_path $log_path \
--model_save_iter 1 \
--log_iter 1 \
--input_mode $input_mode \
--data_path $data_path \
--mode "train" \
--tensorboard false \
#--validate true





