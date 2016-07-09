#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=0
NET=ZF

for fold in `seq 0 3`
do
  LOG="experiments/logs/`date +'%Y-%m-%d_%H-%M-%S'`faster_rcnn_alt_opt_houseware_${NET}_fold${fold}.txt"
  TRAIN_IMDB=houseware_train${fold}
  TEST_IMDB=houseware_test${fold}

  {

    time ./tools/train_faster_rcnn_alt_opt.py --gpu ${GPU_ID} \
      --net_name ${NET} \
      --weights data/imagenet_models/${NET}.v2.caffemodel \
      --imdb ${TRAIN_IMDB} \
      --cfg experiments/cfgs/faster_rcnn_alt_opt_houseware.yml \
      --set EXP_DIR alt_opt_houseware_${fold}

    set +x
    NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
    set -x

    time ./tools/test_net.py --gpu ${GPU_ID} \
      --def models/houseware/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
      --net ${NET_FINAL} \
      --imdb ${TEST_IMDB} \
      --cfg experiments/cfgs/faster_rcnn_alt_opt_houseware.yml \
      --set EXP_DIR alt_opt_houseware_${fold}

  } >> >(tee -a "$LOG")

done
