from PIL import Image
import os

def convert_gif_to_jpg(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否以 .gif 结尾
        if filename.lower().endswith('.png'):
            # 构建完整的文件路径
            old_file = os.path.join(directory, filename)
            # 读取 .gif 文件
            with Image.open(old_file) as img:
                # 移除 .gif 后缀并添加 .jpg 后缀
                new_filename = filename[:-4] + '.jpg'
                new_file = os.path.join(directory, new_filename)
                # 将图片转换为 RGB 模式并保存为 .jpg 文件
                rgb_img = img.convert('RGB')
                rgb_img.save(new_file, 'JPEG')
                print(f'Converted: {old_file} to {new_file}')


def look_for_not_jpg(directory):
    # 遍历指定目录下的所有文件
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.jpg'):
            print(filename)


# 指定你的目录路径
directory_path = '/zhaowei/data/LLaVA-Instruct-150K/ocr_vqa/images'
look_for_not_jpg(directory_path)
