#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=ZF
ITERS=10 #70000

for fold in `seq 0 3`
do
  LOG="experiments/logs/`date +'%Y-%m-%d_%H-%M-%S'`_faster_rcnn_end2end_${NET}_fold${fold}.txt"
  TRAIN_IMDB=houseware_train${fold}
  TEST_IMDB=houseware_test${fold}

  {

    time ./tools/train_net.py --gpu ${GPU_ID} \
      --solver models/houseware/${NET}/faster_rcnn_end2end/solver.prototxt \
      --weights data/imagenet_models/${NET}.v2.caffemodel \
      --imdb ${TRAIN_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/faster_rcnn_end2end_houseware.yml \
      --set EXP_DIR end2end_houseware_${fold}

    set +x
    NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
    set -x

    time ./tools/test_net.py --gpu ${GPU_ID} \
      --def models/houseware/${NET}/faster_rcnn_end2end/test.prototxt \
      --net ${NET_FINAL} \
      --imdb ${TEST_IMDB} \
      --cfg experiments/cfgs/faster_rcnn_end2end_houseware.yml \
      --set EXP_DIR end2end_houseware_${fold}

  } >> >(tee -a "$LOG")

done
