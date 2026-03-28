import os
import shutil
import random
from sklearn.model_selection import train_test_split


def split_yolo_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                       random_seed=42):
    """
    分割YOLO数据集为训练集、验证集和测试集

    参数:
    images_dir: 源图片文件夹路径
    labels_dir: 源标签文件夹路径
    output_dir: 输出数据集文件夹路径
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    random_seed: 随机种子
    """

    # 验证比例总和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "比例总和必须为1"

    # 创建输出目录结构
    dirs = [
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'labels'),
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'labels'),
        os.path.join(output_dir, 'test', 'images'),
        os.path.join(output_dir, 'test', 'labels')
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有图片文件名（不带扩展名）
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # 检查对应的标签文件是否存在
    valid_base_names = []
    for base_name in base_names:
        label_file = os.path.join(labels_dir, base_name + '.txt')
        if os.path.exists(label_file):
            valid_base_names.append(base_name)
        else:
            print(f"警告: 找不到标签文件 {base_name}.txt")

    # 随机打乱数据
    random.seed(random_seed)
    random.shuffle(valid_base_names)

    # 计算各集合大小
    total_size = len(valid_base_names)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    # 分割数据集
    train_names, temp_names = train_test_split(valid_base_names, train_size=train_size, random_state=random_seed)
    val_names, test_names = train_test_split(temp_names, train_size=val_size, test_size=test_size,
                                             random_state=random_seed)

    print(f"数据集统计:")
    print(f"总样本数: {total_size}")
    print(f"训练集: {len(train_names)} 个样本")
    print(f"验证集: {len(val_names)} 个样本")
    print(f"测试集: {len(test_names)} 个样本")

    # 复制文件到对应目录
    def copy_files(names, image_dest_dir, label_dest_dir):
        for name in names:
            # 查找原始图片文件（支持多种格式）
            src_image = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                possible_path = os.path.join(images_dir, name + ext)
                if os.path.exists(possible_path):
                    src_image = possible_path
                    break

            if src_image:
                # 复制图片
                dest_image = os.path.join(image_dest_dir, os.path.basename(src_image))
                shutil.copy2(src_image, dest_image)

                # 复制标签
                src_label = os.path.join(labels_dir, name + '.txt')
                dest_label = os.path.join(label_dest_dir, name + '.txt')
                shutil.copy2(src_label, dest_label)
            else:
                print(f"警告: 找不到图片文件 {name}")

    # 复制各集合文件
    copy_files(train_names,
               os.path.join(output_dir, 'train', 'images'),
               os.path.join(output_dir, 'train', 'labels'))

    copy_files(val_names,
               os.path.join(output_dir, 'val', 'images'),
               os.path.join(output_dir, 'val', 'labels'))

    copy_files(test_names,
               os.path.join(output_dir, 'test', 'images'),
               os.path.join(output_dir, 'test', 'labels'))

    print("数据集分割完成！")


if __name__ == "__main__":
    # 设置路径参数
    images_directory = "Z:/wxy/RT-DETR/数据/VOC2007/JPEGImages"  # 替换为你的图片文件夹路径
    labels_directory = "Z:/wxy/RT-DETR/output_txt"  # 替换为你的标签文件夹路径
    output_directory = "Z:/wxy/RT-DETR/dataset"  # 替换为输出路径

    # 设置分割比例
    train_ratio = 0.7  # 训练集比例
    val_ratio = 0.2  # 验证集比例
    test_ratio = 0.1  # 测试集比例

    # 执行分割
    split_yolo_dataset(images_directory,
                       labels_directory,
                       output_directory,
                       train_ratio,
                       val_ratio,
                       test_ratio)