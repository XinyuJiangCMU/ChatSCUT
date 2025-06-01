import os
import docx
import re
from docx import Document
from docx.table import Table


def process_docx_text(input_text):

    paragraphs = input_text.split('\n')
    texts = []

    for paragraph in paragraphs:
        if paragraph.startswith("第"):

            texts.append(paragraph.rstrip('\n'))
        else:
            texts.append(paragraph)

    modified_text = "".join(texts)
    return modified_text

def remove_spaces(input_text):
    paragraphs = input_text.split('\n')
    texts = []

    for paragraph in paragraphs:
        if paragraph.strip():
            para_text = paragraph.replace('\n', '')
            cleaned_text = re.sub(r'\s+', '', para_text)
            texts.append(cleaned_text)

    modified_text = "\n".join(texts)
    return modified_text


def read_and_format_table(table):
    formatted_data = ""

    try:
        # 检查表格是否为空
        if not table.rows:
            print("Warning: Empty table found")
            return formatted_data

        # 处理表头
        if table.rows[0].cells:
            header_row = table.rows[0]
            col_count = len(header_row.cells)

            head_data = []
            for cell in header_row.cells:
                formatted_data = f"{cell.text.strip()}:"
                head_data.append(formatted_data)
            formatted_data = ""

            # 处理数据行
            for row in table.rows[1:]:
                i = 0
                for cell in row.cells:
                    # 确保每一行的列数与表头相同
                    if i < col_count:
                        formatted_data += head_data[i]
                        formatted_data += f"{cell.text.strip()}\n"
                        i += 1
                formatted_data += "。"    
                formatted_data += "\n"
            return formatted_data
        else:
            print("Warning: Empty header row found")
            return formatted_data
    except Exception as e:
        print(f"Error while processing table: {e}")
        return formatted_data


def process_text(input_text):
    paragraphs = input_text.split('\n')
    texts = []
    text = ""
    has_punctuation = any(char in ",?!;:" for char in input_text)

    if not has_punctuation:
        return input_text  # 如果整个文档没有标点符号，直接返回原文本

    for idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            runs = [run for run in paragraph]
            text += "".join(runs).strip()

            for en, zh in zip([",", "?", "!", ";", ":"], ["，", "？", "！", "；", "："]):
                text = text.replace(en, zh)

            if text[-1] not in [".", "。", "！", "？"]:
                # 如果当前行末尾没有表示结束的标点，检查下一行是否存在，并合并
                if idx < len(paragraphs) - 1:
                    next_paragraph = paragraphs[idx + 1].strip()
                    if next_paragraph and next_paragraph[0] not in [".", "。", "！", "？"]:
                        text += " "
                    else:
                        text += "\n"
                else:
                    text += " "
            elif text[-1] in [".", "。", "！", "？"]:
                texts.append(text)
                text = ""

    # 重新设置段落内容
    modified_text = "\n".join(texts)
    return modified_text

def process_table(docx_path):
    try:
        document = Document(docx_path)
    except KeyError as e:
        print(f"Error processing document: {e}. Skipping file: {docx_path}")
        return ""

    result = ""
    
    skip_next_table = False  # 用于标记是否需要跳过下一个表格

    for element in document.element.body:
        if element.__class__.__name__ == "CT_Tbl":  # 检查是否为表格
            if skip_next_table:
                skip_next_table = False
                continue  # 跳过下一个表格

            table = Table(element, document)  # 直接使用 Table 创建表格对象
            result += read_and_format_table(table)
        elif element.__class__.__name__ == "CT_P":  # 检查是否为段落
            if "教学计划总体安排表" in element.text:
                skip_next_table = True
                continue
            result += element.text + "\n"

    return result
    return result
def main(input_folder, output_folder):
    # 遍历文件夹中的所有docx文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".docx"):
            file_path = os.path.join(input_folder, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as input_file:
                input_content = input_file.read().strip()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as input_file:
                    input_content = input_file.read().strip()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin1') as input_file:
                        input_content = input_file.read().strip()
                except UnicodeDecodeError as e:
                    print(f"无法解码文件 '{file_path}': {e}")
                    continue


            if not input_content:
                print(f"跳过空文档: {file_path}")
                continue

            # 检查输出目录中是否已存在同名文件
            output_file_path = os.path.join(output_folder, file_name.replace(".docx", ".txt"))
            if os.path.exists(output_file_path):
                continue

            print(f"处理文件: {file_path}")
            input_text = process_table(file_path)

            # cleaned_text = remove_spaces(input_text)
            cleaned_text = process_text(input_text)

            cleaned_text = '\n'.join(line for line in cleaned_text.split('\n') if not line.strip() == "。")

    
            try:
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cleaned_text)
            except Exception as e:
                print(f"写入文件时发生异常：{e}")
if __name__ == "__main__":
    input_folder = "docs/docx"  
    output_folder = "docs/cleaned_txt"  

    main(input_folder, output_folder)
