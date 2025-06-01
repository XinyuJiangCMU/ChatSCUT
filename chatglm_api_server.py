import os
import time
import sys
import jieba
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import requests
import argparse
import fitz  # PyMuPDF  
import docx
import io
import hashlib
import hmac
import base64
from datetime import datetime
from werkzeug.utils import secure_filename

# Add common module path
sys.path.append(r".//common")

# Custom modules
from func_llm import load_llm, load_retriever, llmresponse
from func_agent import SimpleAgent
from func_MM import *
from func_RAG import *
from config import SUMMARY_PROMPT_TEMPLATE, AGENT_PROMPT_TEMPLATE

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run chatserver_rag with configurable options')
parser.add_argument('--add_selectllm', type=str, default='chatglm2-6b', help='LLM model to use')
parser.add_argument('--add_selectemb', type=str, default='m3e-base', help='Embedding model to use')
parser.add_argument('--add_selectret', type=str, default='1-stage', help='Retriever type: 1-stage or No Retrieval')
parser.add_argument('--agent_RAG', action='store_true', help='Use agent for RAG decision making')
parser.add_argument('--use_history', action='store_true', help='Use historical conversation context')
args = parser.parse_args()

# Configuration
add_selectllm = args.add_selectllm
add_selectemb = args.add_selectemb
add_selectret = args.add_selectret
agent_RAG = args.agent_RAG
use_history = args.use_history

print('\t'*2, 'add_selectret:', add_selectret)
print('\t', 'add_selectllm:', add_selectllm, '\t', 'agent_RAG:', agent_RAG, '\t', 'use_history:', use_history)

UPLOAD_FOLDER = 'docs/docx'
topk_absname = 1
topk_knowledge = 5
knowledge_threshold = 0.80

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load LLM and retrievers
llm = load_llm(llm_used=add_selectllm, device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(embedding_used=add_selectemb, device=0)

agent_prompt = AGENT_PROMPT_TEMPLATE
summary_prompt_template = SUMMARY_PROMPT_TEMPLATE

# Response generation with optional RAG and history
def generate_response(data):
    global add_selectret
    history_list = data["message"]
    query = history_list[-1]["message"]
    
    if query.endswith('undefined'):
        query = query[:-len('undefined')]

    if agent_RAG:
        print('\t'*5, 'Using agent to determine if RAG is needed')
        is_ret_respond = llmresponse(agent_prompt.format(query=query), topk_absname, topk_knowledge, topk_absname, llm,
                                     knowledge_threshold, 'No Retrieval', one_stage_retriever, two_stage_retriever, 0)["result"]
        agent = SimpleAgent()
        add_selectret = agent.preprocess_message(is_ret_respond)
        print(add_selectret * 10)

    if use_history and len(history_list) > 1:
        print('\t'*5, 'Summarizing conversation history to update query')
        history = "\n".join([f"{msg['role']}：{msg['message']}" for msg in history_list[:-1]])
        summary_prompt = summary_prompt_template.format(history=history, query=query)
        modified_query = llmresponse(summary_prompt, topk_absname, topk_knowledge, topk_absname, llm,
                                     knowledge_threshold, add_selectret, one_stage_retriever, two_stage_retriever, 0)["result"]
        print("Modified Query:", modified_query)
        query = modified_query
    else:
        print("No history available or history not enabled")

    print('\t'*5, 'Generating final response using LLM')
    results = llmresponse(query, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                          add_selectret, one_stage_retriever, two_stage_retriever, 1)
    result = results["result"]
    print("LLM Final Response:\n", result)

    result = jieba.lcut(result)
    for i in result:
        time.sleep(0.06)
        yield i

# Main route for handling file uploads and JSON requests
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
            return "Unsupported or missing file", 400

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
                print(f"Output of {script}: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e.stderr}")
                return jsonify({"error": f"Script failed: {e.stderr}"}), 500

        return jsonify({"success": "File uploaded and processed successfully"})

    elif request.is_json:
        data = request.get_json()
        response = generate_response(data)
        return Response(response, headers=headers)

    else:
        return jsonify({'error': 'Unsupported request format'}), 400

# Route for generating a short title summary
@app.route('/TitleSummary', methods=['POST'])
def process_message():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400

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

if __name__ == '__main__':
    app.run("0.0.0.0", port=1201, debug=True, use_reloader=False)