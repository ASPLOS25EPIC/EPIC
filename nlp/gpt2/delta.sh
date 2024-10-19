#!/usr/bin/env zsh
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_CACHE_CAPACITY=0

OUT_DIR=${OUT_DIR:-"./log"}
DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/wikitext-2-raw-v1"}
# DATA_DIR=${DATA_DIR:-"/data/dataset/nlp/openai-community/wikitext-103-raw-v1"}

epochs="${epochs:-1}"
max_train_steps="${max_train_steps:-100}"
per_device_train_batch_size="${per_device_train_batch_size:-8}"
compressor_ratio="${compressor_ratio:-0.01}"
compressor="${compressor:-topk}"
memory="${memory:-residual}"
pack="${pack:-1}"
freq="${freq:-1}"

echo "out dir is $OUT_DIR"
mkdir -p $OUT_DIR
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi

CMD="horovodrun -np 4 -H localhost:4 python ./pytorch/delta.py  "
CMD+=" --dataset_name $DATA_DIR --dataset_config_name default  "
CMD+=" --model_name_or_path /data/dataset/nlp/openai-community/gpt2 "
CMD+=" --num_train_epochs=$epochs  --max_train_steps=$max_train_steps --per_device_train_batch_size=$per_device_train_batch_size "
CMD+=" --compressor_ratio=$compressor_ratio --compressor=$compressor --memory=$memory "
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
