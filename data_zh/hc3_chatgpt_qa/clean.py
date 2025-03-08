import json
import re

def is_english(text):
    non_english_chars = sum(1 for char in text if ord(char) > 127)
    total_chars = len(text)
    # 允许少量非 ASCII 字符（如 5%）
    return non_english_chars / total_chars < 0.05

# 定义需要过滤的关键字
keywords = ['抱歉', '语言模型', '不知道', '不确定', '计算机程序', '无法确定', '不能回答']

def clean_data(input_file_path, output_file_path):
    # 读取文件
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 过滤数据
    filtered_lines = []
    for line in lines:
        data = json.loads(line)
        article = data.get('article', '')
        label = data.get('label', '')
        
        # 检查条目长度
        if len(article) < 100:
            continue
        if is_english(article):
            continue
        if any(keyword in article for keyword in keywords) and \
           label == 'machine':
            continue
        
        # 预处理文章：去除 \\n
        article = article.replace('\\n', '')
        
        # 更新数据中的 article
        data['article'] = article
        filtered_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

    # 保存过滤后的数据
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)

    print(f"清洗完成，结果保存在 {output_file_path} 文件中。")

# 使用函数进行数据清洗
input_file_path = '/data/home/donglz/codespace/AIGT/AIGTDet/data_zh/hc3_chatgpt_qa/train.jsonl'
output_file_path = '/data/home/donglz/codespace/AIGT/AIGTDet/data_zh/hc3_chatgpt_qa/train_cleaned.jsonl'
clean_data(input_file_path, output_file_path)