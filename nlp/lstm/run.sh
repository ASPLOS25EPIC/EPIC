#!/usr/bin/env zsh
zsh ../../trans.sh
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_CACHE_CAPACITY=0

compressor_ratio="${compressor_ratio:-0.1}"
compressor="${compressor:-topk}"
diff="${diff:-True}"

OUT_DIR=${OUT_DIR:-"./log"}
mkdir -p $OUT_DIR
if [[ ! -d "$OUT_DIR" ]]; then
  echo "ERROR: non existing $OUT_DIR"
  exit 1
fi
current_date=$(date +"%Y-%m-%d")
LOGFILE=$OUT_DIR/$current_date-$compressor-$compressor_ratio-$diff-logfile.txt

# CMD="horovodrun -np 2 -H n20:1,n19:1 python lstm.py  "
CMD="horovodrun -np 2 -H n15:1,n16:1 python lstm.py  "
CMD+=" --compressor_ratio $compressor_ratio --compressor $compressor "
if [[ "$diff" == "True" ]]; then
  CMD+=" --diff True"
fi

echo "$CMD |& tee $LOGFILE"
time eval $CMD |& tee $LOGFILE
