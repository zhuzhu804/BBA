# 定义文件路径
input_file = '../list_file/Emotion6_FI/FI_train_8.txt'  # 替换为您的txt文件路径
output_file = '../list_file/Emotion6_FI/FI_train.txt'  # 输出文件路径

# 定义类别映射规则
def map_label(label):
    if label in {0, 2, 3, 5}:
        return 0
    elif label in {1, 4, 6, 7}:
        return 1
    else:
        raise ValueError(f"未知的类别标签: {label}")

# 读取文件并处理
with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        # 分割路径和标签
        path, label = line.rsplit(' ', 1)
        label = int(label.strip())  # 将标签转为整数
        # 进行类别映射
        new_label = map_label(label)
        # 写入新的行到输出文件
        f_out.write(f"{path} {new_label}\n")

print("类别标签转换完成！新的标签文件已保存。")
