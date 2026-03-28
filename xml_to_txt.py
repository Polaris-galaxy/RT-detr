import os
import xml.etree.ElementTree as ET

# 定义类别列表
classes = ['pipe', 'ground', 'leak', 'ignored regions']  # 替换成你的类别名称

# 输入和输出文件夹路径
xml_folder = './input_xml'  # XML文件夹路径
output_folder = './output_txt'  # TXT输出文件夹路径

os.makedirs(output_folder, exist_ok=True)

failed_files = []  # 存储转换失败的文件名

# 遍历XML文件夹中的所有文件
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        xml_path = os.path.join(xml_folder, xml_file)
        
        try:
            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 获取图像宽度和高度
            width = int(float(root.find('size').find('width').text))
            height = int(float(root.find('size').find('height').text))
            
            # 创建YOLOv5格式的TXT文件
            txt_file_path = os.path.join(output_folder, xml_file.replace('.xml', '.txt'))
            with open(txt_file_path, 'w') as txt_file:
                # 遍历所有对象
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name == 'buning':
                        class_name = 'burning'
                    if class_name == 'short-circut':
                        class_name = 'short-circuit'
                    class_id = classes.index(class_name)
                    
                    # 获取边界框坐标
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)
                    
                    # 计算边界框中心点和宽高
                    x_center = (xmin + xmax) / 2 / width
                    y_center = (ymin + ymax) / 2 / height
                    bbox_width = (xmax - xmin) / width
                    bbox_height = (ymax - ymin) / height
                    
                    # 写入YOLOv5格式的行
                    txt_file.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
            
            print(f'{xml_file}转换完成')
        
        except Exception as e:
            print(f'转换 {xml_file} 时出现错误：{e}')
            failed_files.append(xml_file)

if failed_files:
    print("以下文件转换失败:")
    for file in failed_files:
        print(file)
else:
    print("所有文件转换完成！")
