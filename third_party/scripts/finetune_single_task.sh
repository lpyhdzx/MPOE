# This scripts trains T5 in a single-task setting.

# We train the model on each single task from the GLUE benchmark by setting the `tasks` and `eval_tasks` 
# to one of GLUE_TASKS=["rte", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "cola"], and report the 
# average obtained test scores.
# export CUDA_VISIBLE_DEVICES=4
# python3 -m torch.distributed.launch --nproc_per_node=4  ./finetune_t5_trainer.py configs/finetune_single_task.json 
function get_gpu_count() {
  str=$1
  array=(${str//,/})
  echo ${#array}
}
function get_init_method() {
  ipaddr=`ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2}'|tr -d "addr:"`
  echo ${ipaddr}
}
function run_task_ddp() {
    export CUDA_VISIBLE_DEVICES=$1
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python -m torch.distributed.launch --nproc_per_node=`get_gpu_count ${1}` --master_addr=`get_init_method` --master_port=$2 finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
function run_task() {
    export CUDA_VISIBLE_DEVICES=$1
    export address=$2
    COMMON_ARGS="--output_sub_dir=$3 --mpo_layers=$4 --load_layer=$5 --emb_trunc=$6 --linear_trunc=$7 --attention_trunc=$8 $9"
    nohup python finetune_t5_trainer.py configs/finetune_single_task.json \
    ${COMMON_ARGS}> log/$3_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}
# run_task_ddp 4,5,6,7 12347 t5_sinlge word_embed,mlp,attention noload 30000 30000 30000
# run_task 4 12347 t5_sinlge_mrpc word_embed,mlp,attention noload 30000 30000 30000
# run_task 4 12347 t5_sinlge_sst2 word_embed,mlp,attention noload 30000 30000 30000
# run_task 4 12347 t5_sinlge_sst2_test word_embed,mlp,attention word_embed,mlp,attention 30000 30000 30000 --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer_exp/single_task/t5_sinlge_sst2/checkpoint-6000
# run_task 4 12347 t5_sinlge_sst2_test2 word_embed,mlp,attention word_embed,mlp,attention 30000 30000 30000 --load_experiment=/mnt/liupeiyu/checkpoint/hyperformer_exp/single_task/t5_sinlge_sst2
# run_task 5 12347 t5_sinlge_cola word_embed,mlp,attention noload 30000 30000 30000
# run_task 5 12347 t5_sinlge_cola word_embed,mlp,attention noload 30000 30000 30000
# run_task 5 12347 t5_sinlge_cola2 word_embed,mlp,attention noload 30000 30000 30000
run_task 6 12347 t5_large_sst2 word_embed,mlp,attention noload 30000 30000 30000