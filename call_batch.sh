#!/bin/bash
sbatch --killable --requeue --mem=8g -c4 --gres=gpu:2,vmem:4g --time=72:0:0 remote.sh