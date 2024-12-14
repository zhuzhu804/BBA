import os
from PIL import Image
import pytesseract


def remove_corrupted_images(image_folder):
    # 遍历图片文件夹中的所有文件
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        try:
            # 打开图片文件
            img = Image.open(image_path)

            # 使用pytesseract进行OCR识别
            text = pytesseract.image_to_string(img)

            # 检查图片中是否包含特定的文本
            if "This photo is no longer available" in text:
                # 如果包含，则删除该图片
                os.remove(image_path)
                print(f"Removed corrupted image: {image_name}")

        except IOError:
            print(f"Cannot open or read file: {image_name}")
        except Exception as e:
            print(f"An error occurred: {e}")


# 使用示例
# 假设您的图片文件夹路径为 'path_to_your_dataset'
remove_corrupted_images('/Users/didi/Desktop/EmotionDA/FI/emotion_dataset_remove_blank/sadness')
