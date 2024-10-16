# -*- coding = utf-8 -*-
# @Time : 2024/10/7 16:12
# @Author : 小五
# @File : gpt4.py
# @Sofeware : PyCharm


from extract import extract_number, elements_prompt
import argparse
import os
from glob import glob
import json
# 关于老年劳工的
system_prompt = """你是一名专门在中国老年劳工法律诉讼判决书中从事法律要素标注的人工智能助手，你的职责是抽取给定法律判决书中与指定法律要素相关的原文，并对有属性值的法律要素推理出它的属性值。"""
eg1_raw_path = "/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/del_xml-再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).txt.labeled.txt"
eg1_ans_path = "/home/wangyuting/Datasets/LegalCases/再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).json"
eg2_raw_path = "/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/del_xml-二审修订-883/155.重庆市巴南区环境卫生管理处与邹俊确认劳动关...(FBM-CLI.C.16147796).txt.labeled.txt.labeled.txt"
eg2_ans_path = "/home/wangyuting/Datasets/LegalCases/二审修订-883/155.重庆市巴南区环境卫生管理处与邹俊确认劳动关...(FBM-CLI.C.16147796).json"

# eg_raw_path = r"/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/del_xml-再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).txt.labeled.txt"
# eg_ans_path = r"/home/wangyuting/Datasets/LegalCases/再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).json"
egs = []


def cut_jsonl(input_file):
    output_file_dir = "/home/wangyuting/PycharmWorkplace/LegalCase/batch_gpt4/"
    output_file_prefix = output_file_dir + os.path.basename(input_file).split('.')[0] + '/' + os.path.basename(input_file).split('.')[0]
    chunk_size = 10  # 每个文件包含的行数

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for i in range(0, len(lines), chunk_size):
        part_filename = f"{output_file_prefix}_{i // chunk_size}.jsonl"
        with open(part_filename, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines[i:i + chunk_size])

    print("文件分割完成！")

def create_batch_input_file(input_dir, output_jsonl_file):
    batch_requests = []
    for input_file in glob(f"{input_dir}/*.txt"):
        with open(input_file, 'r', encoding='utf-8') as f_in:
            text = f_in.read()
        file_num = os.path.basename(input_file).split('.')[0]
        # 循环遍历每个法律要素
        for ele, prompt in elements_prompt.items():
            ans = []
            for eg in egs:
                ans.append(eg[1].get(ele, "0"))

            base_prompt = prompt + '\n判决书内容如下:\n"""\n{}\n"""'
            request = {
                "custom_id": f"{file_num}-{ele}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},

                        {"role": "user", "content": base_prompt.format(egs[0][0])},
                        {"role": "assistant", "content": ans[0]},
                        {"role": "user", "content": base_prompt.format(egs[1][0])},
                        {"role": "assistant", "content": ans[1]},

                        {"role": "user", "content": base_prompt.format(text)}
                    ],
                    "max_tokens": 4095,
                    "temperature": 0
                }
            }
            batch_requests.append(request)

    with open(output_jsonl_file, 'w', encoding='utf-8') as f_out:
        for request in batch_requests:
            f_out.write(json.dumps(request, ensure_ascii=False) + "\n")


def main(args):
    input_dir = fr"/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/{args.input}/"  # 输入的文件夹
    output_jsonl = fr"/home/wangyuting/PycharmWorkplace/LegalCase/batch_gpt4/2shot-883.jsonl"  # 输出文件夹

# few-shot
    # 全局变量样本列表，eg[0][0]表示第一个文本的原文，eg[0][1]表示第一个文本的问答字典
    global egs
    eg1 = []
    with open(eg1_raw_path, 'r', encoding='utf-8') as f:
        eg1.append(f.read())
    with open(eg1_ans_path, 'r', encoding='utf-8') as f:
        eg1.append(json.load(f))
    egs.append(eg1)
    eg2 = []
    with open(eg2_raw_path, 'r', encoding='utf-8') as f:
        eg2.append(f.read())
    with open(eg2_ans_path, 'r', encoding='utf-8') as f:
        eg2.append(json.load(f))
    egs.append(eg2)

    create_batch_input_file(input_dir,output_jsonl)
    cut_jsonl(output_jsonl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Zhipu model.")
    parser.add_argument('--model', type=str, default="gpt-4o-mini", help='Name of the model to use (default: glm-4-flash)')
    parser.add_argument('--api_key', type=str, default=OPENAI_API_KEY, help='API key for authentication (default: glm4f_APIKEY)')
    parser.add_argument('--input', type=str, default='del_xml-二审修订-883', help='Input file name')
    # parser.add_argument('--output', type=str, default='gpt4-0shot-883', help='Output file name')
    args = parser.parse_args()
    main(args)
