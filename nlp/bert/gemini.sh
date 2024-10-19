#!/usr/bin/zsh
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_CACHE_CAPACITY=0
echo "Container nvidia build = $NVIDIA_BUILD_ID"

compressor=${1:-"topk"}
comm_mode=${comm_mode:-"allgather"}
compressor_ratio=${2:-"0.01"}
init_checkpoint=${3:-"/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16/bert_large_pretrained_amp.pt"}
# init_checkpoint=${3:-"/data/dataset/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12/bert_base_wiki.pt"}
epochs=${4:-"1"}
batch_size=${5:-"32"}
learning_rate=${6:-"3e-5"}
warmup_proportion=${7:-"0.1"}
precision=${8:-"fp16"}
num_gpu=${9:-"8"}
seed=${10:-"1"}
squad_dir=${11:-"/data/dataset/nlp/bert/squad"}
vocab_file=${12:-"/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16/vocab.txt"}
# vocab_file=${12:-"/data/dataset/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt"}
OUT_DIR=${13:-"./log"}
mode=${14:-"train eval"}
CONFIG_FILE=${15:-"/data/dataset/nlp/bert/pre-model/bert-large-uncased/uncased_L-24_H-1024_A-16/bert_config.json"}
# CONFIG_FILE=${15:-"/data/dataset/nlp/bert/pre-model/bert-base-uncased/uncased_L-12_H-768_A-12/bert_config.json"}
max_steps=${16:-"-1"}
freq=${freq:-"1"}
diff=${diff:-"False"}
pack=${pack:-"1"}

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

use_fp16=""
if [[ "$precision" == "fp16" ]] ; then
  echo "fp16 activated!"
  use_fp16=" --fp16 "
fi

if [[ "$num_gpu" == "1" ]] ; then
  export CUDA_VISIBLE_DEVICES=0
  mpi_command=""
else
  unset CUDA_VISIBLE_DEVICES
  # mpi_command=" -m torch.distributed.launch --nproc_per_node=$num_gpu"
fi

CMD="horovodrun -np 4 -H localhost:4 python ./pytorch/gemini.py "
CMD+="--init_checkpoint=$init_checkpoint "
if [[ "$mode" == "train" ]] ; then
  CMD+="--do_train "
  CMD+="--train_file=$squad_dir/train-v1.1.json "
  CMD+="--train_batch_size=$batch_size "
elif [[ "$mode" == "eval" ]] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
elif [[ "$mode" == "prediction" ]] ; then
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
else
  CMD+=" --do_train "
  CMD+=" --train_file=$squad_dir/train-v1.1.json "
  CMD+=" --train_batch_size=$batch_size "
  CMD+="--do_predict "
  CMD+="--predict_file=$squad_dir/dev-v1.1.json "
  CMD+="--predict_batch_size=$batch_size "
  CMD+="--eval_script=$squad_dir/evaluate-v1.1.py "
  CMD+="--do_eval "
fi

CMD+=" --do_lower_case "
CMD+=" --bert_model=bert-base-uncased "
CMD+=" --learning_rate=$learning_rate "
CMD+=" --warmup_proportion=$warmup_proportion"
CMD+=" --seed=$seed "
CMD+=" --num_train_epochs=$epochs "
CMD+=" --max_seq_length=384 "
CMD+=" --doc_stride=128 "
CMD+=" --output_dir=$OUT_DIR "
CMD+=" --vocab_file=$vocab_file "
CMD+=" --config_file=$CONFIG_FILE "
CMD+=" --max_steps=$max_steps "
CMD+="--compressor=$compressor "
CMD+="--compressor_ratio=$compressor_ratio "
CMD+="--diff=$diff "
if [[ "$pack" != "" ]]; then
  CMD+=" --pack $pack"
fi
if [[ "$freq" != "" ]]; then
  CMD+=" --freq $freq"
fi

current_date=$(date +"%Y-%m-%d")
LOGFILE=$OUT_DIR/$current_date-$compressor-$compressor_ratio-$diff-logfile.txt

echo "$CMD |& tee $LOGFILE"
time eval $CMD |& tee $LOGFILE