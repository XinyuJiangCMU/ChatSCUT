import os
import json

def txt_to_json(txt_path: str, json_path: str):
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        lines = [line.strip() for line in txt_file.readlines() if line.strip()] 
    # 构建json结构
    json_data = {"knowledges": {i: line for i, line in enumerate(lines)}}

    # 保存为json文件
    with open(json_path, "w", encoding='utf-8') as json_file:
        json.dump(json_data, fp=json_file, indent=4, ensure_ascii=False)

def build_jsons(txt_dir: str, json_dir: str):
    txt_paths = [os.path.join(txt_dir, name) for name in os.listdir(txt_dir) if name.endswith(".txt")]
    if not os.path.exists(json_dir):
        os.mkdir(json_dir)
    
    total_knowledges_count = 0
    empty_knowledges_count = 0

    json_paths = [os.path.join(json_dir, os.path.basename(path).split(".")[0] + ".json") for path in txt_paths]
    #print(json_paths)
    for txt_path, json_path in zip(txt_paths, json_paths):
        try:
#             with open(txt_path, 'r', encoding='utf-8') as txt_file:
#                 lines = txt_file.readlines()
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                lines = [line.strip() for line in txt_file.readlines() if line.strip()] 
            total_knowledges_count += len(lines)
            empty_knowledges_count += sum(1 for line in lines if not line.strip())

            txt_to_json(txt_path=txt_path, json_path=json_path)
        except Exception as e:
            # 处理异常
            print(f"{txt_path} 处理时发生异常: {e}")
            continue

    print(f"总的knowledges个数: {total_knowledges_count}")
    print(f"空值的knowledges个数: {empty_knowledges_count}")
if __name__ == "__main__":
    from config import PROJ_TOP_DIR

    txt_dir = os.path.join(PROJ_TOP_DIR, "docs", "cleaned_txt")
    json_dir = os.path.join(PROJ_TOP_DIR, "docs", "cleaned_json")
    build_jsons(txt_dir=txt_dir, json_dir=json_dir)
