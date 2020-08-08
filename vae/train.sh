time_stamp=$(date +%y%m%d)
net="VAE"
#net="CVAE"
input_mode="env"
#input_mode="goal"
#log_path="./result${time_stamp}/${net}_goal/"
log_path="./result/result${time_stamp}_z5b1/${net}/"
#data_path="./data/20200605/"
data_path="./data/20200706/"

if [ -e $log_path ]
then
	time_stamp=$(date +%y%m%d_%H)
	log_path="./result/result${time_stamp}/${net}/"
fi

tb_log_path=${log_path}"tensorboard_log/"
save_path=${log_path}"model/"

mkdir -p $log_path
mkdir $tb_log_path
mkdir $save_path

cp train.sh $log_path

python ./main.py \
--net $net \
--z_dim 5 \
--prior_dist "G" \
--beta 2 \
--batch_size 64 \
--epoch 200 \
--lr 0.001 \
--model_save_path $save_path \
--log_path $log_path \
--input_mode $input_mode \
--data_path $data_path \
--mode "train" \
--log_iter 5 \
--model_save_iter 5 \
#--validate true





