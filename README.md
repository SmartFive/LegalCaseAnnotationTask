# README

## 项目简介

本项目用于从中国老年劳工法律诉讼判决书中，利用大型语言模型 (LLM) 进行法律要素的标注。通过提供判决书文本内容，系统会自动识别并提取与指定法律要素相关的原文，并对带有属性值的法律要素进行推理，生成相应的输出文件。

## 文件结构

- `Dataset`: OLDoc 部分数据。


- `main.py`: 主程序，负责调用LLM模型，处理输入文件，并将结果输出为JSON文件。
- `batch_gpt.py`: 对于chatGPT，使用batch api完成标注。`generate_jsonl.py`: 将输入文本生成jsonl文件。
- `extract.py`: 用于提取文件中的信息和构建法律要素的提示信息。
- `cause_inference.py`: 对标注的部分法律要素进行因果推理。`encode_num.py`: 进行数据编码用于因果推理。
- `metric.py`: 对标注结果进行评估，计算抽取任务的BLEU、ROUGE和BertScore，分类任务的soft F1-Score。
- `text.py`, `allDict.json`: 保存代码中会用到的关于数据集的一些信息。
- `rouge_score`: rouge_score库针对英文，对rouge_scorer等文件进行修改，使其能够评估中文文本。

## 使用方法

### 1. 运行命令

你可以通过命令行执行以下命令运行程序：

```bash
python main.py --model <模型名> --port <端口号> --api_key <API密钥> --shot <样本数量> --input <输入文件夹> --output <输出文件夹>
```

- `--model`：使用的模型名称（默认为 `Qwen2.5-32b-AWQ`）。
- `--port`：本地模型推理时使用的端口号（默认为 `8098`）。
- `--api_key`：访问API时的认证密钥。
- `--shot`：使用的样本数量，0表示不使用样本，1或2表示使用few-shot学习的样本数量（默认为 `0`）。
- `--input`：输入文件所在的文件夹（默认为 `del_xml-二审修订-883`）。
- `--output`：输出文件将保存到的文件夹（默认为 `qwen-2shot-883`）。

### 2. 输入和输出文件

- **输入文件**：判决书的 `.txt` 文本文件。
- **输出文件**：生成的法律要素的问答结果将保存为 `.json` 文件。

### 示例命令

以下命令会调用本地模型 `Qwen2.5-32b-AWQ`，处理输入目录中的 `.txt` 文件，并将输出保存到指定的目录中：

```bash
python main.py --model "Qwen2.5-32b-AWQ" --port "8098" --api_key "<你的API密钥>" --shot "2" --input "del_xml-二审修订-883" --output "qwen-2shot-883"
```

## 主要功能

- **法律要素抽取**：从判决书文本中识别并提取特定法律要素相关的文本。
- **属性值推理**：对于具有属性值的要素，模型可以通过推理获取对应的值。
- **多种模型支持**：支持智谱AI、deepseek、和 OpenAI 模型（正常api和batch api调用），以及本地模型推理。
- **Few-shot学习**：支持0-shot、1-shot或2-shot的样本方式进行模型推理。

## 注意事项

- 确保输入文件夹和输出文件夹的路径正确设置。
- 输出文件如果已经存在，程序将跳过该文件的处理。
- 若出现错误，会记录在 `error_file.txt` 中。
