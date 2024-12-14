import os

# 替换为你的顶级文件夹路径
top_folder = '/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/office-home/Real_World'

# 创建或覆盖一个名为images_list.txt的文件，以保存结果
with open('/nfs/ofs-902-1/object-detection/zhujiankun/EDA/data/office-home/Real_World/images_list.txt', 'w') as file:
    # 遍历顶级文件夹中的每个子文件夹
    for label, folder_name in enumerate(sorted(os.listdir(top_folder))):
        folder_path = os.path.join(top_folder, folder_name)

        # 确保是一个文件夹
        if os.path.isdir(folder_path):
            # 遍历文件夹中的每个文件
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # 确保是一个文件而不是子文件夹
                if os.path.isfile(image_path):
                    # 写入图像的路径和标签（数字）
                    file.write(f'{image_path} {label}\n')

print('列表文件已生成。')

