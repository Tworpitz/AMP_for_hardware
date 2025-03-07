#!/bin/bash

# 定义URDF文件夹路径
URDF_DIR="./urdf"

# 遍历./urdf/文件夹下的所有.urdf文件
for URDF_FILE in "$URDF_DIR"/*.urdf; do
    # 获取文件名（不含路径）
    FILENAME=$(basename "$URDF_FILE" .urdf)
    
    # 定义输出文件路径
    OUTPUT_FILE="./${FILENAME}_output.txt"
    
    # 使用check_urdf命令检查URDF文件，并将输出重定向到文本文件
    echo "正在检查文件: $URDF_FILE"
    check_urdf "$URDF_FILE" > "$OUTPUT_FILE"
    
    # 打印完成信息
    echo "检查完成，输出已保存到 $OUTPUT_FILE"
    echo ""
done

echo "所有URDF文件检查完成！"

