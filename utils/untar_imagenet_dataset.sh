#!/bin/bash

# # use the following dataset
# /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar
# # untar some data into /root/autodl-tmp/imagenet/
# tar xvf ILSVRC2012_img_train.tar -C /root/autodl-tmp/imagenet/
# run this script in /root/autodl-tmp/imagenet/
# # mkdir val train, and copy all untar files into these two directory  

# 遍历当前目录下的所有 xxx.tar 文件
for file in *.tar; do
    # 检查文件是否存在
    if [[ -f $file ]]; then
        # 获取文件名（不包含扩展名）
        dir_name=$(basename "$file")
        dir_name=$(echo $dir_name | sed 's/.tar$//')
        # 创建以文件名命名的目录
        mkdir -p "$dir_name"
        # 解压文件到该目录
        tar -xvf "$file" -C "$dir_name"
    fi
done