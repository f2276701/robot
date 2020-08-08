data_path="../data/20200706_z7/orig/"
output_path="../data/20200706_z7_2/processed/"
z_dim=7
model_load_path="../model/result200706_vae_z7b2/"

mkdir -p $output_path
cp data_process.sh $output_path

python align_data.py \
--joint_path $data_path \
--img_path $data_path \
--output_path $output_path

python extract_img.py \
--z_dim $z_dim \
--data_path $output_path \
--model_load_path $model_load_path \

python combine.py \
--data_path $output_path \

python normalize.py \
--data_path $output_path \

python parallel_shift.py \
--data_path $output_path \

python plot.py \
--data_path $output_path \

