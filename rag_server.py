import argparse
import os
import subprocess
import sys
import time

import jieba
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

sys.path.append(r".//common")

# 本地模块导入
from common.func_MM import FOLDERS_TO_CLEAR, clear_folders
from common.config import AGENT_PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE
from common.func_agent import SimpleAgent
from common.func_llm import llmresponse, load_llm, load_retriever
from common.func_MM import *  # noqa: F403

# ------------------------ 命令行参数 ------------------------
parser = argparse.ArgumentParser(description='运行 ChatSCUT 服务端脚本')
parser.add_argument('--add_selectllm', type=str, default='chatglm2-6b', help='选择的LLM模型')
parser.add_argument('--add_selectemb', type=str, default='m3e-base', help='选择的嵌入模型')
parser.add_argument('--add_selectret', type=str, default='1-stage', help='选择的检索阶段')
parser.add_argument('--agent_RAG', action='store_true', help='是否使用代理判断是否需要RAG')
parser.add_argument('--use_history', action='store_true', help='是否使用历史对话进行摘要')

args = parser.parse_args()

# ------------------------ 参数赋值 ------------------------
add_selectllm = args.add_selectllm
add_selectemb = args.add_selectemb
add_selectret = args.add_selectret
agent_RAG = args.agent_RAG
use_history = args.use_history

print('\tLLM模型:', add_selectllm, '\t使用代理RAG:', agent_RAG, '\t使用历史摘要:', use_history)

# ------------------------ Flask 配置 ------------------------
UPLOAD_FOLDER = 'docs/docx'
topk_absname = 1
topk_knowledge = 5
knowledge_threshold = 0.80

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------ 初始化 LLM 和检索器 ------------------------
llm = load_llm(llm_used=add_selectllm, device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(embedding_used=add_selectemb, device=0)

# ------------------------ 模板 ------------------------
agent_prompt = AGENT_PROMPT_TEMPLATE
summary_prompt_template = SUMMARY_PROMPT_TEMPLATE

# ------------------------ 响应生成函数 ------------------------
def generate_response(data):
    history_list = data["message"]
    query = history_list[-1]["message"]
    if query.endswith('undefined'):
        query = query[:-len('undefined')]

    # 如果启用代理，根据用户问题判断是否需要RAG
    if agent_RAG:
        print('\t' * 5, '使用代理判断是否需要RAG')
        is_ret_respond = llmresponse(
            agent_prompt.format(query=query),
            topk_absname, topk_knowledge, topk_absname, llm,
            knowledge_threshold, 'No Retrieval',
            one_stage_retriever, two_stage_retriever, 0
        )["result"]
        agent = SimpleAgent()
        add_selectret = agent.preprocess_message(is_ret_respond)
        print(add_selectret * 10)
    else:
        add_selectret = '1-stage'

    # 如果启用历史摘要，则生成新的查询
    if use_history and len(history_list) > 1:
        history = "\n".join([f"{msg['role']}：{msg['message']}" for msg in history_list[:-1]])
        if history:
            print('\t' * 5, '摘要历史对话并修改查询')
            summary_prompt = summary_prompt_template.format(history=history, query=query)
            modified_query = llmresponse(
                summary_prompt, topk_absname, topk_knowledge, topk_absname,
                llm, knowledge_threshold,
                add_selectret, one_stage_retriever, two_stage_retriever, 0
            )["result"]
            print("修改后的查询:", modified_query)
            query = modified_query
    else:
        print("未使用历史摘要")

    # 调用模型生成最终回答
    print('\t' * 5, '调用LLM生成最终回答')
    start_time = time.time()
    results = llmresponse(
        query, topk_absname, topk_knowledge, topk_absname,
        llm, knowledge_threshold,
        add_selectret, one_stage_retriever, two_stage_retriever, 1
    )
    print("生成时间:", time.time() - start_time)

    result = results["result"]
    print("最终回答:", result)

    # 分词流式返回
    for token in jieba.lcut(result):
        time.sleep(0.06)
        yield token

# ------------------------ 主接口：问答与上传 ------------------------
@app.route('/', methods=['POST'])
def handle_request():
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.docx'):
            return "请上传 .docx 文件", 400

        clear_folders(FOLDERS_TO_CLEAR)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        scripts_to_run = [
            'common/clean_data.py',
            'common/txt2json.py',
            'common/write_abstract.py',
            'common/vector_store.py',
        ]

        for script in scripts_to_run:
            try:
                result = subprocess.run(['python', script], check=True, capture_output=True, text=True)
                print(f"{script} 输出: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"{script} 运行失败: {e.stderr}")
                return jsonify({"error": f"{script} 运行失败: {e.stderr}"}), 500

        return jsonify({"success": "文件上传并处理成功"})

    elif request.is_json:
        data = request.get_json()
        response = generate_response(data)
        return Response(response, headers=headers)
    else:
        return jsonify({'error': '请求必须是JSON或.docx文件'}), 400

# ------------------------ 接口：生成标题摘要 ------------------------
@app.route('/TitleSummary', methods=['POST'])
def process_message():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "缺少message字段"}), 400

    new_message = message + "\n请将以上文字总结为十个字以内的标题。你的回答只用告诉我标题内容是什么，并且一定不能超过十个字"
    new_data = {
        "message": [{
            "id": "7a740b84-06a0-42de-b030-b3dd284abfde",
            "chatid": "401addd4-78a7-4801-9292-0f4b29870102",
            "message": new_message,
            "role": "user",
            "time": "2024-06-10T17:49:05"
        }]
    }

    try:
        response = requests.post('http://localhost:1203', json=new_data, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    received_text = ''
    for chunk in response.iter_content(chunk_size=512):
        received_text += chunk.decode('utf-8')

    return jsonify({"result": received_text})

# ------------------------ 启动服务 ------------------------
if __name__ == '__main__':
    app.run("0.0.0.0", port=1203, debug=True, use_reloader=False)