from bert_score import score
import os
from tqdm import tqdm
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba
import re

from text import attribute_keys, text_keys
def DrawBertScoreSimilarityMatrix(pred, ref):
    from bert_score import plot_example

    # 使用本地的 bert-base-chinese 模型
    plot_example(pred, ref, lang="zh", fname="bert_score.png",
                 model_type="/home/public/bert-base-chinese/", num_layers=12)  # 指定本地模型路径


# 定义文件夹路径
# reference_folder_template = "/home/wangyuting/Datasets/LegalCases/{}/"
# prediction_folder = "/home/wangyuting/PycharmWorkplace/LegalCase/GPT4-0shot/gpt4-0shot-16/"

def BertScore(predictions, references):
    # 使用本地的 bert-base-chinese 模型
    P, R, F1 = score(predictions, references, lang="zh", # verbose=True,
                     model_type="/home/public/bert-base-chinese", num_layers=12)  # 指定本地模型路径

    return P, R, F1


punctuation_pattern = r'[^\w\s]'
def RougeScore(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)  # 中文不需要词干化
    rouge_1_scores, rouge_2_scores, rouge_L_scores = [], [], []

    for pred, ref in zip(predictions, references):
        # 替换标点符号为空字符串
        pred = re.sub(punctuation_pattern, '', " ".join(jieba.cut(pred)))
        ref = re.sub(punctuation_pattern, '', " ".join(jieba.cut(ref)))
        scores = scorer.score(ref, pred)

        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_L_scores.append(scores['rougeL'].fmeasure)

    return sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0,\
        sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0, \
        sum(rouge_L_scores) / len(rouge_L_scores) if rouge_L_scores else 0

def BleuScore(predictions, references):
    bleu_1_scores, bleu_2_scores, bleu_n_scores = [], [], []
    smoothing_function = SmoothingFunction()
    for pred, ref in zip(predictions, references):
        # 使用 jieba 进行中文分词
        pred = re.sub(punctuation_pattern, '', " ".join(jieba.cut(pred)))
        ref = re.sub(punctuation_pattern, '', " ".join(jieba.cut(ref)))
        ref_list = [ref.split()]  # BLEU expects a list of reference translations
        pred_list = pred.split()
        bleu_1_scores.append(sentence_bleu(ref_list, pred_list, weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method1))
        bleu_2_scores.append(sentence_bleu(ref_list, pred_list, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function.method1))
        bleu_n_scores.append(sentence_bleu(ref_list, pred_list, smoothing_function=smoothing_function.method1))

    return sum(bleu_1_scores) / len(bleu_1_scores) if bleu_1_scores else 0, \
        sum(bleu_2_scores) / len(bleu_2_scores) if bleu_2_scores else 0, \
        sum(bleu_n_scores) / len(bleu_n_scores) if bleu_n_scores else 0

def calculate_accuracy(predicted_dict, reference_dict):
    correct = 0
    total = 0
    for key in reference_dict:
        if key in predicted_dict:
            total += 1
            if predicted_dict[key] == reference_dict[key]:
                correct += 1
    return correct / total if total > 0 else 0


def Soft_f1_score(predicted_dict, reference_dict):
    # 初始化 true positives, false positives, 和 false negatives
    tp = 0  # true positives
    fp = 0  # false positives
    fn = 0  # false negatives

    for key in reference_dict:
        if predicted_dict[key] != "0":
            # 参考答案是否包含在预测值中
            if reference_dict[key] in predicted_dict[key]:
                tp += 1  # 正确预测
            else:
                fp += 1  # 预测了但不匹配
        else:
            fn += 1  # 参考中有，但预测中没有

    # 计算 precision, recall 和 F1 分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1



def main():
    # 定义父文件夹路径，包含多个文件夹
    parent_folder = "/home/wangyuting/Datasets/LegalCases/"
    prediction_folder = "/home/wangyuting/PycharmWorkplace/LegalCase/GLM4f-2shot/"

    total_bertP, total_bertR, total_bertF1 = 0, 0, 0
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    total_bleu1, total_bleu2, total_bleun = 0, 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    total_count = 0

    # 获取父文件夹下所有子文件夹
    sub_folders = [f for f in os.listdir(prediction_folder) if os.path.isdir(os.path.join(prediction_folder, f))]

    for sub_folder in sub_folders:
        print(f"Processing folder: {sub_folder}")

        folder_path = os.path.join(prediction_folder, sub_folder)
        folder_name = os.path.basename(os.path.normpath(folder_path))
        parts = folder_name.split('-')
        numeric_part = None
        for part in parts:
            if part.isdigit():
                numeric_part = part
                break
        num_dict = {
            "883": "二审修订-883",
            "679": "一审简易修订-679",
            "289": "一审普通修订-289",
            "16": "再审-16"
        }

        num_dir = num_dict.get(numeric_part)
        reference_folder = os.path.join(parent_folder, num_dir)

        pred_files = os.listdir(folder_path)
        folder_count = 0

        for pred_file in tqdm(pred_files, desc="Processing files", unit="file"):
            if pred_file.endswith(".json"):
                pred_file_path = os.path.join(folder_path, pred_file)

                pred_file_number = pred_file.split('.')[0]
                pred_file_number += '.'

                reference_file = None
                for ref_file in os.listdir(reference_folder):
                    if ref_file.startswith(pred_file_number):
                        reference_file = ref_file
                        break

                if reference_file is None:
                    print(f"No matching reference file for {pred_file}")
                    continue

                reference_file_path = os.path.join(reference_folder, reference_file)
                try:
                    with open(pred_file_path, 'r') as f:
                        pred_data = json.load(f)
                    with open(reference_file_path, 'r') as f:
                        ref_data = json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    print(f"Error processing file {pred_file_path}: {e}")
                    continue

                # 对 reference_texts 进行排序
                reference_texts = {k: v for k, v in ref_data.items() if k in text_keys}
                reference_texts = {k: reference_texts[k] for k in sorted(reference_texts.keys())}
                # 对 prediction_texts 进行排序，并确保键是与 reference_texts 对应的
                prediction_texts = {k: v for k, v in pred_data.items() if k in reference_texts.keys()}
                prediction_texts = {k: prediction_texts[k] for k in sorted(prediction_texts.keys())}
                # 对 reference_attrs 进行排序
                reference_attrs = {k: v for k, v in ref_data.items() if k in attribute_keys}
                reference_attrs = {k: reference_attrs[k] for k in sorted(reference_attrs.keys())}
                # 对 prediction_attrs 进行排序，并确保键是与 reference_attrs 对应的
                prediction_attrs = {k: v for k, v in pred_data.items() if k in reference_attrs.keys()}
                prediction_attrs = {k: prediction_attrs[k] for k in sorted(prediction_attrs.keys())}

                bertP, bertR, bertF1 = BertScore(list(prediction_texts.values()), list(reference_texts.values()))
                rouge1, rouge2, rougeL = RougeScore(list(prediction_texts.values()), list(reference_texts.values()))
                bleu1, bleu2, bleu_n = BleuScore(list(prediction_texts.values()), list(reference_texts.values()))
                precision, recall, f1 = Soft_f1_score(prediction_attrs, reference_attrs)

                total_bertP += bertP.mean().item()
                total_bertR += bertR.mean().item()
                total_bertF1 += bertF1.mean().item()
                total_rouge1 += rouge1
                total_rouge2 += rouge2
                total_rougeL += rougeL
                total_bleu1 += bleu1
                total_bleu2 += bleu2
                total_bleun += bleu_n
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_count += 1
                folder_count += 1

        print(f"Folder {sub_folder} processed with {folder_count} files.")

    if total_count > 0:
        print(f"BertScore 平均 P: {total_bertP / total_count:.4f}, 平均 R: {total_bertR / total_count:.4f}, 平均 F1: {total_bertF1 / total_count:.4f}")
        print(f"平均 ROUGE-1: {total_rouge1 / total_count:.4f}, 平均 ROUGE-2: {total_rouge2 / total_count:.4f}, 平均 ROUGE-L: {total_rougeL / total_count:.4f}")
        print(f"平均 BLEU-1: {total_bleu1 / total_count:.4f}, 平均 BLEU-2: {total_bleu2 / total_count:.4f}, 平均 BLEU-n: {total_bleun / total_count:.4f}")
        print(f"分类任务 平均 P: {total_precision / total_count:.4f}, 平均 R: {total_recall / total_count:.4f}, 平均 F1: {total_f1 / total_count:.4f}")
    else:
        print("No valid files processed.")

if __name__ == "__main__":
    main()
