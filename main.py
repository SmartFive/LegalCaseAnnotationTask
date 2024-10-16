# -*- coding = utf-8 -*-
# @Time : 2024/9/27 13:20
# @Author : 小五
# @File : main.py
# @Sofeware : PyCharm

from zhipuai import ZhipuAI
from openai import OpenAI
from extract import extract_number, elements_prompt
import argparse
import os
from glob import glob
import json

OPENAI_API_BASE = "http://localhost:{}/v1"  # 如果是本地推理就写localhost，如果是调用openai 就写给的openai接口名字
DEEPSEEK_URL = "https://api.deepseek.com"


# 关于老年劳工的
system_prompt = """你是一名专门在中国老年劳工法律诉讼判决书中从事法律要素标注的人工智能助手，你的职责是抽取给定法律判决书中与指定法律要素相关的原文，并对有属性值的法律要素推理出它的属性值。"""

eg1_raw_path = "/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/del_xml-再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).txt.labeled.txt"
eg1_ans_path = "/home/wangyuting/Datasets/LegalCases/再审-16/4.纳雍县烤烟综合服务农民专业合作社确认劳动关...(FBM-CLI.C.318017818).json"
eg2_raw_path = "/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/del_xml-二审修订-883/155.重庆市巴南区环境卫生管理处与邹俊确认劳动关...(FBM-CLI.C.16147796).txt.labeled.txt.labeled.txt"
eg2_ans_path = "/home/wangyuting/Datasets/LegalCases/二审修订-883/155.重庆市巴南区环境卫生管理处与邹俊确认劳动关...(FBM-CLI.C.16147796).json"

egs = []


def call_LLM(text):
    """
    处理单个文件的法律要素问答，并返回结果字典。
    :param text: 判决书内容
    :param eg: 案例列表，每个元素的第一个元素是raw，第二个元素是问答json
    :return: 文件的问答结果字典
    """
    model_name = ''
    if args.model == 'glm-4-flash' or args.model == 'glm-4-plus':
        client = ZhipuAI(api_key=args.api_key)
        model_name = args.model
    elif args.model == "deepseek-chat":
        client = OpenAI(api_key=args.api_key, base_url=DEEPSEEK_URL)
        model_name = "deepseek-chat"
    else:  # 本地模型推理
        url = OPENAI_API_BASE.format(args.port)
        print(url)
        client = OpenAI(base_url=url, api_key=args.api_key)
        model_name = "/home/public/" + args.model
    global egs
    file_result = {}
    # 循环遍历每个法律要素
    for ele, prompt in elements_prompt.items():
        ans = []
        for eg in egs:
            ans.append(eg[1].get(ele, "0"))

        base_prompt = prompt + '\n判决书内容如下:\n"""\n{}\n"""'

        # 发送请求
        if args.shot == "0":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_prompt.format(text)}
                ],
                max_tokens=4095,
                temperature=0,
                stream=False
            )
        elif args.shot == "1":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},

                    {"role": "user", "content": base_prompt.format(egs[0][0])},
                    {"role": "assistant", "content": ans[0]},

                    {"role": "user", "content": base_prompt.format(text)}
                ],
                max_tokens=4095,
                temperature=0,
                stream=False
            )
        elif args.shot == "2":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},

                    {"role": "user", "content": base_prompt.format(egs[0][0])},
                    {"role": "assistant", "content": ans[0]},
                    {"role": "user", "content": base_prompt.format(egs[1][0])},
                    {"role": "assistant", "content": ans[1]},

                    {"role": "user", "content": base_prompt.format(text)}
                ],
                max_tokens=4095,
                temperature=0,
                stream=False
            )
        else:
            raise ValueError("输入案例数量错误！请确保输入为 '0', '1' 或 '2'。")

        # 获取响应内容
        answer = response.choices[0].message.content

        # 将法律要素的问答结果保存到字典中
        file_result[ele] = answer

    return file_result


def main(args):
    input_dir = fr"/home/wangyuting/PycharmWorkplace/LagalLabelExtractor/DocumentLabeling/{args.input}/"  # 输入的文件夹
    output_dir = fr"/home/wangyuting/PycharmWorkplace/LegalCase/{args.output}/"  # 输出文件夹

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in glob(f"{input_dir}/*.txt"):
        print(f"Processing {input_file}")
        input_file_name = os.path.basename(input_file)
        output_file_name = input_file_name + '.json'
        output_file = os.path.join(output_dir, output_file_name)
        # 确认输出文件是否已经存在，如果存在则跳过
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping.")
            continue
        try:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                file = f_in.read()
                # 处理文件并获取问答结果
                file_result = call_LLM(file)
                if file_result:
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        json.dump(file_result, f_out, ensure_ascii=False, indent=4)
                        print(f"{input_file} finished!")

        except Exception as e:
            num = extract_number(input_file)
            print(f"Error processing {num} file: {e}")

            file_path = os.path.join(output_dir, 'error_file.txt')
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"{num}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Large Language Model.")
    parser.add_argument('--model', type=str, default="Qwen2.5-32b-AWQ", help='Name of the model to use (default: Qwen2.5-32b-AWQ)')
    parser.add_argument('--port', type=str, default="8098", help='Port of the model to use (default: 8098)')
    parser.add_argument('--api_key', type=str, default='', help='API key for authentication')
    parser.add_argument('--shot', type=str, default='0', help='The number of samples to use (default: 0)')
    parser.add_argument('--input', type=str, default='del_xml-二审修订-883', help='Input file name (default: del_xml-二审修订-883)')
    parser.add_argument('--output', type=str, default='qwen-2shot-883', help='Output file name (default: qwen-2shot-883)')
    args = parser.parse_args()
    main(args)