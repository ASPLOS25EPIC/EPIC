#!/usr/bin/zsh
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_CACHE_CAPACITY=0

compressor="${compressor:-topk}"
comm_mode="${comm_mode:-allgather}"
compressor_ratio="${compressor_ratio:-0.01}"
diff="${diff:-False}"
b="${b:-256}"
model="${model:-vgg19}"
dataset="${dataset:-imagenet}"
resume="${resume:-}"
epochs="${epochs:-1}"
freq="${freq:-1}"
remote="${remote:-}"
pack="${pack:-}"

OUT_DIR=${OUT_DIR:-"./log"}
mkdir -p $OUT_DIR
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi
current_date=$(date +"%Y-%m-%d")
LOGFILE=$OUT_DIR/$current_date-$compressor-$compressor_ratio-$diff-logfile.txt

CMD="horovodrun -np 4 -H 192.168.3.23:4 python ./pytorch/gemini.py"
CMD+=" --noeval"
CMD+=" -b $b"
CMD+=" --model $model --dataset $dataset --epochs $epochs"
CMD+=" --compressor_ratio $compressor_ratio --compressor $compressor --comm_mode $comm_mode"

[[ "$diff" == "True" ]] && CMD+=" --diff True"
[[ "$remote" != "" ]] && CMD+=" --remote $remote"
[[ "$resume" != "" ]] && CMD+=" --resume $resume"
[[ "$pack" != "" ]] && CMD+=" --pack $pack"
[[ "$freq" != "" ]] && CMD+=" --freq $freq"

echo "$CMD |& tee $LOGFILE"
time eval $CMD |& tee $LOGFILE
