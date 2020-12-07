#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


conda activate vitis-ai-tensorflow

# folders
export BUILD=./build
export APP=./FPGA_demo/
export TARGET=${BUILD}/target
export LOG=${BUILD}/logs
export TB_LOG=${BUILD}/tb_logs
export KERAS=${BUILD}/keras_model
export FREEZE=${BUILD}/freeze
export COMPILE=${BUILD}/compile/
export QUANT=${BUILD}/quantize
export TFCKPT_DIR=${BUILD}/tf_chkpt

# make the necessary folders
mkdir -p ${KERAS}
mkdir -p ${LOG}

# logs & results files
export TRAIN_LOG=train.log
export KERAS_LOG=keras_2_tf.log
export FREEZE_LOG=freeze.log
export EVAL_FR_LOG=eval_frozen_graph.log
export QUANT_LOG=quant.log
export EVAL_Q_LOG=eval_quant_graph.log
export COMP_LOG=compile.log

# Keras file. Change to the model file you want to use
export K_MODEL=CustomCNN.h5

# TensorFlow files
export FROZEN_GRAPH=frozen_graph.pb
export TFCKPT=tf_float.ckpt

# calibration list file
export CALIB_LIST=calib_list.txt
export CALIB_IMAGES=400

# network parameters
export INPUT_HEIGHT=224
export INPUT_WIDTH=224
export INPUT_CHAN=3
export INPUT_SHAPE=?,${INPUT_HEIGHT},${INPUT_WIDTH},${INPUT_CHAN}

export NET_NAME=fpga_model	

export INPUT_NODE=conv2d_input                          #use for CustomCnn
export OUTPUT_NODE=dense_1/BiasAdd                      #use for CustomCnn

if [ $# -eq 1 ]; then
        if [ $1 = DenseNetX ]; then                     #use for DenseNetX
                export K_MODEL=DenseNetX.h5
                export INPUT_NODE=input_1
                export OUTPUT_NODE=activation_99/Softmax
        fi
fi

echo Using $K_MODEL

# training parameters
export EPOCHS=160
export BATCHSIZE=150
export LEARNRATE=0.001

# target board parameters
ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json


