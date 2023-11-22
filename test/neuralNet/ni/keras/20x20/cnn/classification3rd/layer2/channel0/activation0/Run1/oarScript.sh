#!/bin/bash

EXEC_DIR=/mnt/home/kkarimi/Project/git/crystalDefect/test/neuralNet/ni/keras/20x20/cnn/classification3rd/layer2/channel0/activation0/Run1
 source /mnt/opt/spack-0.17/share/spack/setup-env.sh

spack load python@3.8.12%gcc@8.3.0


time ipython3 py_script.py

