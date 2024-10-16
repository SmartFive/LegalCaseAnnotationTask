# # -*- coding = utf-8 -*-
# # @Time : 2024/10/7 1:50
# # @Author : 小五
# # @File : encode_num.py
# # @Sofeware : PyCharm

import os
import json
import pandas as pd

# 指定要处理的键
keys_to_process = {
    "劳动者何时达到法定退休年龄": "A",
    "有无书面合同": "B",
    "有无享受养老保险待遇": "C",
    "是否认定为基本养老保险待遇": "D",
    "劳动者性别": "E",
    "养老保险待遇类型": "F",
    "审理法院的对应裁判结果": "G"
}
# 对各列进行数字编码的字典
encoding_dict = {
    "劳动者何时达到法定退休年龄": {"劳动前达到法定退休年龄": 0, "劳动期间达到法定退休年龄": 1},
    "有无书面合同": {"有": 1, "无": 0},
    "有无享受养老保险待遇": {"无": 0, "有": 1},
    "是否认定为基本养老保险待遇": {"是": 1, "否": 0},
    "劳动者性别": {"男": 0, "女": 1},
    "养老保险待遇类型": {"城乡居民养老保险": 0, "城镇居民社会养老保险": 3, "新型农村社会养老保险": 1, "城镇职工基本养老保险待遇": 2},
    "审理法院的对应裁判结果": {"劳动关系": 1, "劳务关系": 0}
}


# 处理单个文件夹中的JSON文件
def process_json_files_in_folder(folder_path):
    data = []  # 用于存储每个文件处理后的数据

    # 遍历文件夹中的所有JSON文件
    for json_file in os.listdir(folder_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            row = {}
            # 针对每个指定的键进行处理
            for key, column in keys_to_process.items():
                value = json_data.get(key, "0")

                # 处理缺失值，记录指示列
                if value == "0":
                    row[column] = -1
                    row[column + "0"] = 0  # 指示值缺失
                else:
                    # 根据编码字典对值进行编码
                    row[column] = encoding_dict.get(key, {}).get(value, -1)  # 若无匹配，编码为 -1
                    row[column + "0"] = 1  # 指示值不缺失

            data.append(row)

    return data


# 处理所有子文件夹并保存到一个Excel文件
def process_folders_and_save(folders_path, output_file):
    all_data = []  # 用于存储所有文件夹处理后的数据

    # 遍历主文件夹中的四个子文件夹
    for folder in os.listdir(folders_path):
        folder_path = os.path.join(folders_path, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")
            folder_data = process_json_files_in_folder(folder_path)
            all_data.extend(folder_data)  # 将每个子文件夹的数据追加到总数据中

    # 将所有数据保存为一个Excel文件
    df = pd.DataFrame(all_data)
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")


# 示例调用
folders_path = '/home/wangyuting/Datasets/LegalCases/'  # 包含四个子文件夹的主文件夹路径
output_file = '/home/wangyuting/Datasets/LegalCases/statics.xlsx'  # 输出的Excel文件路径

process_folders_and_save(folders_path, output_file)
