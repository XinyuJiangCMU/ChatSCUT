import os
import time
import sys
import json
import jieba
import base64
import hashlib
import hmac
import io
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from http.client import HTTPSConnection
import fitz  # PyMuPDF for PDF text extraction
import docx  # For DOCX file text extraction
import requests
import subprocess

# Append common module path
sys.path.append(r".//common")
from func_gpt import load_llm, load_retriever, llmresponse  # Make sure this path and imports are valid

# Generate HMAC-SHA256 signature
def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

# Check if uploaded file is an image
def is_image(file):
    image_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    return file.filename.split('.')[-1].lower() in image_extensions

# Extract text content from PDF file using PyMuPDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

# Extract text content from DOCX file using python-docx
def extract_text_from_docx(file):
    text = ""
    doc = docx.Document(io.BytesIO(file.read()))
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# Perform OCR using Tencent Cloud's GeneralFastOCR API
def perform_ocr(image_file_path, secret_id, secret_key, region='ap-guangzhou', token=""):
    service = "ocr"
    host = "ocr.tencentcloudapi.com"
    version = "2018-11-19"
    action = "GeneralFastOCR"

    # Read image and convert to base64
    with open(image_file_path, 'rb') as image_file:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    # JSON payload
    payload = json.dumps({
        "ImageBase64": image_base64,
    })

    # Construct canonical request
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    content_type = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{content_type}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (
        f"{http_request_method}\n{canonical_uri}\n{canonical_querystring}\n"
        f"{canonical_headers}\n{signed_headers}\n{hashed_request_payload}"
    )

    # Timestamp and credential scope
    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    credential_scope = f"{date}/{service}/tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    algorithm = "TC3-HMAC-SHA256"
    string_to_sign = (
        f"{algorithm}\n{timestamp}\n{credential_scope}\n{hashed_canonical_request}"
    )

    # Signature generation
    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # Set HTTP headers
    authorization = (
        f"{algorithm} "
        f"Credential={secret_id}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    headers = {
        "Authorization": authorization,
        "Content-Type": content_type,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": timestamp,
        "X-TC-Version": version,
        "X-TC-Region": region
    }
    if token:
        headers["X-TC-Token"] = token

    # Send request and parse response
    try:
        req = HTTPSConnection(host)
        req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        resp = req.getresponse()
        response_json = resp.read().decode('utf-8')
        data = json.loads(response_json)

        # Extract detected text
        detected_texts = [detection['DetectedText'] for detection in data['Response']['TextDetections']]
        all_text = ''.join(detected_texts)
        return all_text

    except Exception as err:
        print("OCR Error:", err)
        return None

def perform_asr(audio_file_path, secret_id, secret_key, region='', token=""):
    service = "asr"
    host = "asr.tencentcloudapi.com"
    version = "2019-06-14"
    action = "SentenceRecognition"

    with open(audio_file_path, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    payload = json.dumps({
        "SubServiceType": 2,
        "EngSerViceType": "16k_zh-PY",
        "SourceType": 1,
        "VoiceFormat": "wav",
        "UsrAudioKey": "unique_audio_key",
        "Data": audio_base64,
        "DataLen": len(audio_data),
    })

    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    canonical_headers = f"content-type:{ct}\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    timestamp = int(time.time())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    algorithm = "TC3-HMAC-SHA256"
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    headers = {
        "Authorization": authorization,
        "Content-Type": ct,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": timestamp,
        "X-TC-Version": version
    }
    if region:
        headers["X-TC-Region"] = region
    if token:
        headers["X-TC-Token"] = token

    try:
        req = HTTPSConnection(host)
        req.request("POST", "/", headers=headers, body=payload.encode("utf-8"))
        resp = req.getresponse()
        response_json = resp.read().decode('utf-8')
        data = json.loads(response_json)
        result_content = data['Response']['Result']
        return result_content
    except Exception as err:
        print(err)
        return None

def clear_folders(folders):
    for folder in folders:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    # 如果没有子目录，可以省略对目录的处理
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

SUMMARY_PROMPT_TEMPLATE = """\
以下是【多轮对话历史】和最新一轮的问题，请把最新一轮问题补充为带有对话历史背景的问题，方便之后对问题进行知识库检索。

【多轮对话历史】
{history}

请结合【多轮对话历史】，分析最新一轮用户的问题“{query}”,改为一段带有对话历史信息叙述的问题，但是不能引入对话历史中不存在的信息。

返回格式：
用户：{{问题}}\
"""

# Configuration
# gpt-4-1106-preview_azure  Qwen-7B-Chat chatglm2-6b gpt-3.5-turbo-1106_azure
UPLOAD_FOLDER = 'docs/docx'
SELECT_LLM = "gpt-4-1106-preview_azure"
SELECT_EMBEDDING = "m3e-base"
SELECT_RETRIEVER = "1-stage"
TOPK_ABSNAME = 1
TOPK_KNOWLEDGE = 5
KNOWLEDGE_THRESHOLD = 0.80

app = Flask(__name__)

if sys.version_info[0] <= 2:
    from http.client import HTTPSConnection
else:
    from http.client import HTTPSConnection

# 清空文件
FOLDERS_TO_CLEAR = [
    'docs/cleaned_json',
    'docs/cleaned_txt',
    'docs/docx',
    'docs/json',
]


# 初始化 llm，retriever
llm = load_llm(llm_used=add_selectllm, device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(embedding_used=add_selectemb, device=0)


# from langchain_community.tools.tavily_search import TavilySearchResults
# search = TavilySearchResults(max_results=2)
# search.invoke("华南理工大学有校长吗")
# tools = [search]

# model_with_tools = llm.bind_tools(tools)



app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER123

summary_prompt_template = SUMMARY_PROMPT_TEMPLATE



def generate_response(data):
    # 提取对话历史和最新一轮的问题
    history_list = data["message"]
    query = history_list[-1]["message"]  # 最新一轮的问题
    if len(history_list) > 1:
        history = "\n".join([f"{msg['role']}：{msg['message']}" for msg in history_list[:-1]])  # 对话历史
    else:
        history = ""
    # 生成summary prompt
    summary_prompt = summary_prompt_template.format(history=history, query=query)

    # 调用模型生成补充后的问题
    modified_query = llmresponse(summary_prompt, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                                 add_selectret, one_stage_retriever, two_stage_retriever,0)["result"]

    print("modified_query",modified_query)
    # 调用模型生成最终回答
    results = llmresponse(modified_query, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                          add_selectret, one_stage_retriever, two_stage_retriever,1)
    
    result = results["result"]
    print("result",result)
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
        response = generate_response(data)
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
    new_message = message + "请将以上文字总结为十个字以内的标题。你的回答只用告诉我标题内容是什么，并且一定不能超过十个字"

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
    app.run("0.0.0.0", port=1203, debug=True, use_reloader=False)
