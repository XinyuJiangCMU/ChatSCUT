import os
import time
import sys
import jieba
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import subprocess
import requests
import argparse
sys.path.append(r".//common")
from func_llm import load_llm, load_retriever, llmresponse   # 确保这部分代码和路径是正确的 , generate_response
from func_agent import SimpleAgent
from func_MM import *
from func_RAG import *
from config import SUMMARY_PROMPT_TEMPLATE, AGENT_PROMPT_TEMPLATE

parser = argparse.ArgumentParser(description='运行chatscut脚本并传递参数')
parser.add_argument('--add_selectllm', type=str, default='chatglm2-6b', help='选择的LLM模型，默认为chatglm2-6b')
parser.add_argument('--add_selectemb', type=str, default='m3e-base', help='选择的嵌入模型，默认为m3e-base')
parser.add_argument('--add_selectret', type=str, default='1-stage', help='选择的检索阶段，默认为1-stage')
parser.add_argument('--agent_RAG', action='store_true', help='是否使用RAG代理')
parser.add_argument('--use_history', action='store_true', help='是否调用历史记录')
args = parser.parse_args()
# input examples: 
# python chatscut.py --add_selectllm chatglm2-6b --add_selectemb m3e-base --add_selectret 1-stage --agent_RAG
# python chatscut.py --add_selectllm chatglm2-6b --add_selectemb m3e-base --add_selectret 1-stage
# python chatscut.py --add_selectllm gpt-4-1106-preview_azure --add_selectemb m3e-base --add_selectret 1-stage --agent_RAG --use_history
# python chatscut.py --add_selectllm ChatGLM4-9B --add_selectemb m3e-base --add_selectret 1-stage
# python chatscut_web_glm.py --add_selectllm ChatGLM4-9B --add_selectret 'No Retrieval'
# add_selectret 用No Retrieval输入会有bug 没办法判断负

add_selectllm = args.add_selectllm # gpt-4-1106-preview_azure chatglm2-6b gpt-3.5-turbo-1106_azure ChatGLM4-9B
add_selectemb = args.add_selectemb # m3e-base
add_selectret = args.add_selectret # No Retrieval 1-stage 
agent_RAG = args.agent_RAG  # True or False
use_history = args.use_history # True or False
print('\t'*2, 'add_selectret:',add_selectret)
print('\t', 'add_selectllm:',add_selectllm,'\t','agent_RAG:',agent_RAG,'\t','use_history:',use_history)

# 配置参数

UPLOAD_FOLDER = 'docs/docx'
topk_absname = 1
topk_knowledge = 5
knowledge_threshold = 0.80

app = Flask(__name__)

if sys.version_info[0] <= 2:
    pass
else:
    pass

# 初始化 llm，retriever

llm = load_llm(llm_used=add_selectllm, device_map="auto")
llm_gpt = load_llm(llm_used='gpt-4-1106-preview_azure', device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(embedding_used=add_selectemb, device=0)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

agent_prompt = AGENT_PROMPT_TEMPLATE
summary_prompt_template = SUMMARY_PROMPT_TEMPLATE



def generate_response_glm(data):
    add_selectret = args.add_selectret
    # 提取对话历史和最新一轮的问题
    history_list = data["message"]
    query = history_list[-1]["message"]  # 最新一轮的问题
    
    llmresponse_start_time = time.time()
    
    if query.endswith('undefined'):
        query = query[:-len('undefined')]
    
    if agent_RAG:
        print('\t'*5,'调用agent，判断是否需要RAG')
        is_ret_respond = llmresponse(agent_prompt.format(query = query), topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                                    'No Retrieval', one_stage_retriever, two_stage_retriever,0)["result"]
        agent = SimpleAgent()
        add_selectret = agent.preprocess_message(is_ret_respond)
        print(add_selectret*10)
    # else:
    #     add_selectret = '1-stage'  # 根据需要修改默认值 1-stage
    
    if use_history:
        # 生成summary prompt
        
        his_sum_start_time = time.time()
        if len(history_list) > 1:
            history = "\n".join([f"{msg['role']}：{msg['message']}" for msg in history_list[:-1]])  # 对话历史
        else:
            history = ""
        if len(history)!=0:
            print('\t'*5,'调用agent，总结历史，修改用户问题')
            
            summary_prompt = summary_prompt_template.format(history=history, query=query)
            
            modified_query = llmresponse(summary_prompt, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                                        add_selectret, one_stage_retriever, two_stage_retriever,0)["result"]
            his_sum_end_time = time.time()
            print("modified_query",modified_query)
            query = modified_query
            print("his_sum_time",his_sum_end_time-his_sum_start_time)
        else:
            print("No history available.")
            
    # 调用模型生成最终回答
    print('\t'*5,'调用glm模型生成最终回答')
    results = llmresponse(query, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                          add_selectret, one_stage_retriever, two_stage_retriever,1)
      
    
    llmresponse_end_time = time.time()
    print("llmresponse_time",llmresponse_end_time-llmresponse_start_time)
    result = results["result"]
    print('\t'*3,"glm回答",'\n', result)
    
    return result



def generate_response(data,response_glm):
    add_selectret = '1-stage'
    # 提取对话历史和最新一轮的问题
    history_list = data["message"]
    query = history_list[-1]["message"]  # 最新一轮的问题
    if query.endswith('undefined'):
        query = query[:-len('undefined')]
    
    glm_response_prompt_templete = '''
    用户输入的问题: {query}

    根据华南理工大学数据微调后的大语言模型给出的回答:{response_glm}

    '''
    
    prompt_glm_respond = glm_response_prompt_templete.format(query=query, response_glm=response_glm)
    print(prompt_glm_respond)
    exit()
    if agent_RAG:
        print('\t'*5,'调用agent，判断是否需要RAG')
        is_ret_respond = llmresponse(agent_prompt.format(query = query), topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                                    'No Retrieval', one_stage_retriever, two_stage_retriever,0)["result"]
        agent = SimpleAgent()
        add_selectret = agent.preprocess_message(is_ret_respond)
        print(add_selectret*10)
    # else:
    #     add_selectret = '1-stage'  # 根据需要修改默认值 1-stage
    
    if use_history:
        # 生成summary prompt
        
        his_sum_start_time = time.time()
        if len(history_list) > 1:
            history = "\n".join([f"{msg['role']}：{msg['message']}" for msg in history_list[:-1]])  # 对话历史
        else:
            history = ""
        if len(history)!=0:
            print('\t'*5,'调用agent，总结历史，修改用户问题')
            
            summary_prompt = summary_prompt_template.format(history=history, query=query)
            
            modified_query = llmresponse(summary_prompt, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                                        add_selectret, one_stage_retriever, two_stage_retriever,0)["result"]
            his_sum_end_time = time.time()
            print("modified_query",modified_query)
            query = modified_query
            print("his_sum_time",his_sum_end_time-his_sum_start_time)
        else:
            print("No history available.")
            
    # 调用模型生成最终回答
    print('\t'*5,'调用模型生成最终回答')
    
    llmresponse_start_time = time.time()
    #add_selectret = 'No Retrieval'
    # print('\t'*2, 'add_selectret2:',add_selectret)
     
    
    results = llmresponse(query, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                          add_selectret, one_stage_retriever, two_stage_retriever,1)
    
    
    llmresponse_end_time = time.time()
    print("llmresponse_time",llmresponse_end_time-llmresponse_start_time)
    result = results["result"]
    print("ChatSCUT回答",'\n', result)
    
    result = jieba.lcut(result)
    for i in result:
        time.sleep(0.06)
        yield i

@app.route('/', methods=['POST'])
def handle_request():
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    if 'file' in request.files:
        # 处理文件上传
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and file.filename.endswith('.docx'):
            clear_folders(FOLDERS_TO_CLEAR)
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # 这里你可以添加处理 .docx 文件的代码
            # 按顺序运行指定的 Python 脚本
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
                    return jsonify({"error": f"Error running {script}: {e.stderr}"}), 500
                #不对就报错
            return jsonify({"success": "File uploaded successfully"})
        else:
            return "Unsupported file type", 400
    elif request.is_json:
        # 处理 JSON 请求
        data = request.get_json()
        response_glm = generate_response_glm(data)  # 最后改这里就行 先改上面 exit（）
        response = generate_response(data, response_glm)  # 最后改这里就行 先改上面 exit（）
        return Response(response, headers=headers)
    else:
        return jsonify({'error': 'Request must be JSON or .docx file'}), 400
    
@app.route('/TitleSummary', methods=['POST'])
def process_message():
    # 接收请求中的JSON数据
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    # 修改message字符串
    new_message = message + "\n请将以上文字总结为十个字以内的标题。你的回答只用告诉我标题内容是什么，并且一定不能超过十个字"
    # 构造新的JSON数据
    new_data = {"message": [{"id":"7a740b84-06a0-42de-b030-b3dd284abfde","chatid":"401addd4-78a7-4801-9292-0f4b29870102","message":new_message,"role":"user","time":"2024-06-10T17:49:05"}]}
    # 发送POST请求到指定URL
    try:
        response = requests.post('http://localhost:1203', json=new_data, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    # 接收返回的流式传输字符串并将其转换为完整的字符串
    received_text = ''
    for chunk in response.iter_content(chunk_size=512):
        received_text += chunk.decode('utf-8')
    # 返回接收到的字符串
    return jsonify({"result": received_text}) 

if __name__ == '__main__':
    app.run("0.0.0.0", port=1201, debug=True, use_reloader=False)
