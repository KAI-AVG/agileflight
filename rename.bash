#!/bin/bash
if [ -z "$1" ]; then
    echo "请提供文件夹目录作为参数。"
    exit 1
fi
# 遍历当前目录及其子目录下的所有文件夹
find "$1" -type d | while read -r folder; do
    # 提取文件夹名
    folderName=$(basename "$folder")
    # 检查文件夹名是否以 _数字_数字_数字_数字_数字_数字 格式结尾
    if [[ $folderName =~ _[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}_[0-9]{2}$ ]]; then
        # 使用 sed 去除时间戳
        newName=$(echo "$folderName" | sed 's/_[0-9]\{4\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}_[0-9]\{2\}$//')
        # 提取文件夹所在目录
        dir=$(dirname "$folder")
        # 重命名文件夹
        mv "$folder" "$dir/$newName"
    fi
done
