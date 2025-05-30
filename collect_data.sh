#!/bin/bash

# 配置文件路径
CONFIG_FILE="/home/shang/evfly_ws/src/evfly/flightmare/flightpy/configs/vision/config.yaml"
N=10             #表示运行的轮次
desiredVel=17    #表示desiredVel
dynamic_value=1  #0表示static，1表示dynamic

if [ "$dynamic_value" -eq 0 ]; then
    ob_type="static"
elif [ "$dynamic_value" -eq 1 ]; then
    ob_type="dynamic"
else
    echo "dynamic_value 取值无效，应为 0 或 1"
    exit 1
fi

yq e ".dynamic=$dynamic_value" -i $CONFIG_FILE
# 遍历 environment_100 到 environment_91
for i in {0..9..1}; do
    env_folder="environment_$i"
    # 使用 yq 修改 env_folder 字段的值
    yq e ".environment.env_folder=\"$env_folder\"" -i $CONFIG_FILE
    echo "当前 env_folder 设置为: $env_folder"
    # 执行 launch_evaluation.bash 脚本
    bash launch_evaluation.bash N=$N desiredVel=$desiredVel ob_type=$ob_type
    sleep 20
done