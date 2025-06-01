from docx.document import Document
from docx import Document as document
import os
import json
import docx

def docx_to_json(docx_path: str, json_path: str):
    doc: Document = document(docx_path)

    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    runs_ls = [p.runs for p in paragraphs]
    texts = []
    text = ""

    for runs in runs_ls:
        for run in runs:
            # 过滤掉上下标
            if not (run.font.subscript or run.font.superscript):
                text += run.text
        text = text.strip()
        for en, zh in zip([",", "?", "!", ";", ":"], ["，", "？", "！", "；", "："]):
            text.replace(en, zh)
        if len(text)!=0:
            if text[-1] in [".", "。", "！", "？"]:
                texts.append(text)
                text = ""
                # 标题融合
            else:
                    text += "\n"
    with open(json_path, "w") as f:
        json.dump(
            {"knowledges": {i: k for i, k in enumerate(texts)}},
            fp=f,
            indent=4,
            ensure_ascii=False,
        )

def build_jsons(docx_dir: str, json_dir: str):
    docx_paths = [os.path.join(docx_dir, name) for name in os.listdir(docx_dir)]
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    json_paths = [
        os.path.join(json_dir, os.path.basename(path).split(".")[0] + ".json")
        for path in docx_paths
    ]
    for docx_path, json_path in zip(docx_paths, json_paths):
        try:
            docx_to_json(docx_path=docx_path, json_path=json_path)
        except (KeyError,docx.opc.exceptions.PackageNotFoundError):
            # 处理数值转换异常
            print(f"{docx_path} 无法导入")
            # 继续下一次循环迭代
            continue
if __name__ == "__main__":
    from config import PROJ_TOP_DIR
    docx_dir = os.path.join(PROJ_TOP_DIR, "docs", "docx")
    json_dir = os.path.join(PROJ_TOP_DIR, "docs", "json")
    build_jsons(docx_dir=docx_dir, json_dir=json_dir)