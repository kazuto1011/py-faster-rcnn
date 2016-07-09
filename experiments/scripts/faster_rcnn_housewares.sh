#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=ZF

for fold in `seq 0 3`
do
  LOG="experiments/logs/`date +'%Y-%m-%d_%H-%M-%S'`_houseware_${NET}_fold${fold}.txt"
  exec &> >(tee -a "$LOG")
  echo Logging output to "$LOG"

  time ./tools/train_faster_rcnn_alt_opt.py --gpu 0 \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb houseware_train${fold} \
  --cfg experiments/cfgs/faster_rcnn_housewares.yml \
  --set EXP_DIR houseware_${fold}

  set +x
  NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
  set -x

  time ./tools/test_net.py --gpu 0 \
  --def models/houseware/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb houseware_test${fold} \
  --cfg experiments/cfgs/faster_rcnn_housewares.yml \
  --set EXP_DIR houseware_${fold}
done
