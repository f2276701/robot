net="LSTM"
#log_path="./result/result200611_05_adam/${net}/"
log_path="./result/result200616/${net}/"
load_path=${log_path}"model/"
data_path="./data/20200605/"

cp test.sh $log_path

python main.py \
--net $net \
--unit_nums 100 \
--pb_dims 3 \
--in_dims 11 \
--out_dims 11 \
--model_load_path $load_path \
--data_path $data_path \
--mode "predict" \
--log_path $log_path \





