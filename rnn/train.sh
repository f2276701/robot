time_stamp=$(date +%y%m%d)
net="LSTMPB"
#net="GRUPB"
#net="LSTM"
log_path="./result/result${time_stamp}_epoch10000/${net}/"
#data_path="./data/old/20200527/"
data_path="./data/20200706/"
#data_path="./data/syn2/"
#data_path="./data/syn/"

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


python main.py \
--net $net \
--unit_nums 100 \
--batch_size 64 \
--pb_dims 7 \
--epoch 10000 \
--lr 0.001 \
--in_dims  12 \
--out_dims 12 \
--model_save_path $save_path \
--data_path $data_path \
--mode "train" \
--log_path $log_path \
--closed_index 5 \
--closed_rate 0.9 \
--closed_step 6000 \
--log_iter 200 \
--model_save_iter 200 \
--tensorboard true





