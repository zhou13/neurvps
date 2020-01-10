#!/bin/bash
echo "checkpoint_best"
./eval.py -o $1/result-best.npz -d $2 $1/config.yaml $1/checkpoint_best.pth.tar
echo "checkpoint_latest"
./eval.py -o $1/result-latest.npz -d $2 $1/config.yaml $1/checkpoint_latest.pth.tar
