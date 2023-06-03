# #--------------------------------2S2F--------------------------------
# model=lstm
# system=2S2F
# trace_num=200
# total_t=5.1
# dt=0.01
# baseline_epoch=50
# seed_num=10
# tau_1=0.1
# tau_s=0.8
# device=cpu
# gpu_id=1
# cpu_num=1
# data_dir=Data/$system/data/
# baseline_log_dir=logs/$system/$model/
# result_dir=Results/$system/


#--------------------------------1S2F--------------------------------
model=lstm
system=1S2F
trace_num=100
total_t=15.1
dt=0.01
baseline_epoch=50
seed_num=10
tau_1=0.3
tau_s=3.0
device=cpu
gpu_id=0
cpu_num=1
data_dir=Data/$system/data/
baseline_log_dir=logs/$system/$model/
result_dir=Results/$system/


CUDA_VISIBLE_DEVICES=$gpu_id python run.py \
--model $model \
--system $system \
--trace_num $trace_num \
--total_t $total_t \
--dt $dt \
--baseline_epoch $baseline_epoch \
--seed_num $seed_num \
--tau_1 $tau_1 \
--tau_s $tau_s \
--device $device \
--cpu_num $cpu_num \
--data_dir $data_dir \
--baseline_log_dir $baseline_log_dir \
--result_dir $result_dir \
# --parallel