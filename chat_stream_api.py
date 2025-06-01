from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time
import sys
import jieba

sys.path.append(r".//common")
from func import *

# 预设参数
add_selectllm = "chatglm2-6b"
add_selectemb = "m3e-base"
add_selectret = "1-stage"
topk_absname = 1
topk_knowledge = 5
knowledge_threshold = 0.80
his_k = 5

# 初始化 llm，retriever（全局范围，只加载一次）
llm = load_llm(llm_used=add_selectllm, device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(
    embedding_used=add_selectemb, device=0
)

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def abc():
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    print("hello")
    
    def generateResponse(data):
        chat_history = []
        history = ""
        for i in chat_history:
            history += "用户：" + i["query"] + "\n" + "回答：" + i["result"] + "\n"
        summary_prompt = data["message"]
        print("summary_prompt", summary_prompt)
        results = llmresponse(summary_prompt, topk_absname, topk_knowledge, topk_absname, llm, knowledge_threshold,
                              add_selectret, one_stage_retriever, two_stage_retriever)
        result = results["result"]
        result = jieba.lcut(result)
        print("result", result)
        for i in result:
            time.sleep(0.1)
            yield i

    if request.is_json:
        print("Request received")
        data = request.get_json()
        print("data", data)
        re = generateResponse(data)
        return Response(re, headers=headers)
    else:
        return jsonify({'error': 'Request must be JSON'}), 400

if __name__ == '__main__':
    app.run("0.0.0.0", port=1203, debug=False, use_reloader=False)