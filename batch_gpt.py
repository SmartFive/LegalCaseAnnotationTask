# -*- coding = utf-8 -*-
# @Time : 2024/10/10 19:09
# @Author : 小五
# @File : batch_gpt.py
# @Sofeware : PyCharm
import json
import os
from openai import OpenAI
import time
import argparse

# 初始化OpenAI客户端
OPENAI_API_KEY = ""
OPENAI_API_BASE = ""
client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)


# 第1步：批量上传文件夹中的.jsonl文件
def upload_files_from_folder(folder_path):
    file_ids = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):  # 确保只处理.jsonl文件
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                response = client.files.create(file=f, purpose="batch")
                file_ids[filename] = response.id
                print(f"Uploaded {filename}, file ID: {response.id}")
    return file_ids


# 第2步：为每个文件创建批量请求
def create_batches_for_files(file_ids):
    batch_ids = {}
    for filename, file_id in file_ids.items():
        response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"batch job for {filename}"}
        )
        batch_ids[filename] = response.id
        print(f"Batch created for {filename}, batch ID: {response.id}")
    return batch_ids


# 第3步：检查所有批量请求的状态，直到完成
def check_batch_statuses(batch_ids):
    completed_batches = {}
    for filename, batch_id in batch_ids.items():
        while True:
            status = client.batches.retrieve(batch_id)  # 获取Batch对象
            if status.status == 'completed':  # 使用对象属性访问方式
                completed_batches[filename] = status.output_file_id
                print(f"Batch {batch_id} for {filename} completed!")
                break
            elif status.status == 'failed':  # 使用对象属性访问方式
                print(f"Batch {batch_id} for {filename} failed!")
                break
            elif status.status == 'expired':  # 使用对象属性访问方式
                print(f"Batch {batch_id} for {filename} expired!")
                break
            else:
                time.sleep(10)
    return completed_batches



# 第4步：下载并保存所有结果到json文件
def download_and_save_results(completed_batches, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历所有已完成的批处理任务
    for filename, output_file_id in completed_batches.items():
        file_response = client.files.content(output_file_id)
        result_jsonl = file_response.text

        file_data = {}  # 用来存储每个文件的键值对

        # 处理每一行输出结果
        for line in result_jsonl.splitlines():
            result = json.loads(line)
            custom_id = result["custom_id"]

            # 从 custom_id 中提取文件名和键
            extracted_filename, key = custom_id.split('-')

            # 模型返回的回答
            response_content = result['response']['body']['choices'][0]['message']['content']

            # 检查是否已有该文件的内容，若已有则继续添加键值对
            if extracted_filename in file_data:
                file_data[extracted_filename][key] = response_content
            else:
                file_data[extracted_filename] = {key: response_content}

        # 保存到指定的json文件
        for extracted_filename, content in file_data.items():
            output_filename = os.path.join(output_folder, f"{extracted_filename}.json")

            # 如果该文件已存在，加载现有数据并更新
            if os.path.exists(output_filename):
                with open(output_filename, 'r') as f:
                    existing_data = json.load(f)
                existing_data.update(content)
            else:
                existing_data = content

            # 保存更新后的内容到json文件
            with open(output_filename, 'w') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
                print(f"Results for {extracted_filename} saved to {output_filename}")



def main(args):
    # 主流程
    input_folder = f"/home/wangyuting/PycharmWorkplace/LegalCase/batch_gpt4/{args.input}/"  # 存放已有的jsonl文件的文件夹
    output_folder = f"/home/wangyuting/PycharmWorkplace/LegalCase/{args.output}/"  # 存放模型结果的文件夹

    # 上传所有jsonl文件
    file_ids = upload_files_from_folder(input_folder)

    # 为每个jsonl文件创建批量任务
    batch_ids = create_batches_for_files(file_ids)

    # 检查批量任务状态直到所有任务完成
    completed_batches = check_batch_statuses(batch_ids)

    # 下载并保存所有批处理结果
    download_and_save_results(completed_batches, output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gpt-batch model.")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Name of the model to use (default: glm-4-flash)')
    parser.add_argument('--api_key', type=str, default=OPENAI_API_KEY, help='API key for authentication (default: glm4f_APIKEY)')
    parser.add_argument('--input', type=str, default='1shot-883', help='Input file name')
    parser.add_argument('--output', type=str, default='gpt4-1shot-883', help='Output file name')
    args = parser.parse_args()
    main(args)