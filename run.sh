#!/bin/bash

# 配置文件路径
CONFIG_FILE="/home/shang/evfly_ws/src/evfly/flightmare/flightpy/configs/vision/config.yaml"

configs=(
    # "0,static,OrigUnet_lstm,17,dvs,40 80"
    # "0,static,OrigUnet_lstm,21,dvs,40 80"
    # "1,dynamic,Vit,17,depth,80"
    # "0,static,OrigUnet_lstm,5,dvs,20 40 60 80 100"
    # "0,static,OrigUnet_lstm,9,dvs,20 40 60 80 100"
    # "0,static,OrigUnet_lstm,13,dvs,20 40 60 80 100"
    # "0,static,OrigUnet_lstm,17,dvs,20 40 60 80 100"
    # "0,static,OrigUnet_lstm,21,dvs,20 40 60 80 100"
    # "1,dynamic,OrigUnet_lstm,5,dvs,20 40 60 80 100"
    # "1,dynamic,OrigUnet_lstm,9,dvs,20 40 60 80 100"
    # "1,dynamic,OrigUnet_lstm,13,dvs,20 40 60 80 100"
    # "1,dynamic,OrigUnet_lstm,17,dvs,20 40 60 80 100"
    # "1,dynamic,OrigUnet_lstm,21,dvs,20 40 60 80 100"
    # "0,static,Vit,5,depth,20 40 60 80 100"
    # "0,static,Vit,9,depth,20 40 60 80 100"
    # "0,static,Vit,13,depth,20 40 60 80 100"
    # "0,static,Vit,17,depth,20 40 60 80 100"
    # "0,static,Vit,21,depth,20 40 60 80 100"
    # "1,dynamic,Vit,5,depth,20 40 60 80 100"
    # "1,dynamic,Vit,9,depth,20 40 60 80 100"
    # "1,dynamic,Vit,13,depth,20 40 60 80 100"
    # "1,dynamic,Vit,17,depth,20 40 60 80 100"
    # "1,dynamic,Vit,21,depth,20 40 60 80 100"
    # "0,static,RVT_Stack,5,depth,20 40 60 80 100"
    # "0,static,RVT_Stack,9,depth,20 40 60 80 100"
    # "0,static,RVT_Stack,13,depth,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,5,depth,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,9,depth,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,13,depth,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,17,depth,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,21,depth,20 40 60 80 100"
    # "0,static,RVT_Stack,5,dvs,20 40 60 80 100"
    # "0,static,RVT_Stack,9,dvs,20 40 60 80 100"
    # "0,static,RVT_Stack,13,dvs,20 40 60 80 100"
    # "0,static,RVT_Stack,17,dvs,20 40 60 80 100"
    # "0,static,RVT_Stack,21,dvs,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,5,dvs,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,9,dvs,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,13,dvs,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,17,dvs,20 40 60 80 100"
    # "1,dynamic,RVT_Stack,21,dvs,20 40 60 80 100"
    # "0,static,FusionNet,5,fusion,20 40 60 80 100"
    # "0,static,FusionNet,9,fusion,20 40 60 80 100"
    # "0,static,FusionNet,13,fusion,20 40 60 80 100"
    # "0,static,FusionNet,17,fusion,20 40 60 80 100"
    # "0,static,FusionNet,21,fusion,20 40 60 80 100"
    # "1,dynamic,FusionNet,5,fusion,20 40 60 80 100"
    # "1,dynamic,FusionNet,9,fusion,20 40 60 80 100"
    # "1,dynamic,FusionNet,13,fusion,20 40 60 80 100"
    # "1,dynamic,FusionNet,17,fusion,20 40 60 80 100"
    # "1,dynamic,FusionNet,21,fusion,20 40 60 80 100"
    #"0,static,FusionCross2,9,fusion,20 40 60 80 100"
    "0,static,FusionCross,21,fusion,60"
    # "0,static,FusionCross,9,fusion,20 40 60 80 100"
    # "0,static,FusionCross,13,fusion,20 40 60 80 100"
    # "0,static,FusionCross,17,fusion,20 40 60 80 100"
    # "1,dynamic,FusionCross,5,fusion,20 40 60 80 100"
    # "1,dynamic,FusionCross,9,fusion,20 40 60 80 100"
    # "1,dynamic,FusionCross,13,fusion,20 40 60 80 100"
    # "1,dynamic,FusionCross,17,fusion,20 40 60 80 100"
    # "0,static,ResCross,9,fusion,20 40 60 80 100"
    # "0,static,ResCross,13,fusion,20 40 60 80 100"
    # "0,static,ResCross,17,fusion,20 40 60 80 100"
    # "0,static,ResCross,21,fusion,20 40 60 80 100"
    # "1,dynamic,ResCross,5,fusion,20 40 60 80 100"
    # "1,dynamic,ResCross,9,fusion,80 100"
    # "1,dynamic,ResCross,13,fusion,20 40 60 80 100"
    # "1,dynamic,ResCross,17,fusion,20 40 60 80 100"
    # "1,dynamic,ResCross,21,fusion,20 40 60 80 100"
    
)


for config in "${configs[@]}"; do
    IFS=',' read -r dynamic_value ob_type model vel input_mode epochs_str <<< "$config"
    yq e ".dynamic=$dynamic_value" -i $CONFIG_FILE

    # 将字符串转换为数组
    IFS=' ' read -r -a checkpoint_epoch_list <<< "$epochs_str"

    for checkpoint_epoch in "${checkpoint_epoch_list[@]}"; do
        echo "当前 checkpoint_epoch 设置为: $checkpoint_epoch"
        # 遍历 environment_100 到 environment_91
        for i in {100..99..-1}; do
            env_folder="environment_$i"
            # 使用 yq 修改 env_folder 字段的值
            yq e ".environment.env_folder=\"$env_folder\"" -i $CONFIG_FILE
            echo "当前 env_folder 设置为: $env_folder"
            # 执行 launch_evaluation.bash 脚本
            bash launch_evaluation.bash N=2 model=$model desiredVel=$vel ob_type=$ob_type checkpoint_epoch=$checkpoint_epoch  input_mode=$input_mode 
            sleep 20
        done
    done
done