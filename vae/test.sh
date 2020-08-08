#net="VAE"
net="CVAE"
#input_mode="env"
input_mode="goal"
#log_path="./result200523/${net}/"
log_path="../result/result200523/fine-tuning_${net}/"
save_path=${log_path}"model/"
#data_path="./data_new/"
data_path="../data/"

cp fineturning.sh $log_path

python ../main.py \
--net $net \
--z_dim 20 \
--rec_num 32 \
--model_load_path $save_path \
--model_save_path $save_path \
--input_mode $input_mode \
--log_path $log_path \
--data_path $data_path \
--mode "test" \
--tensorboard false





