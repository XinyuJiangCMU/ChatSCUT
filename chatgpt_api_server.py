from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import time
import sys
import jieba

# Add custom module path
sys.path.append(r"./common")

# Import custom functions
from func_gpt import *

# === Configuration Parameters ===
SELECTED_LLM = "gpt-4-1106-preview_azure"
SELECTED_EMBEDDING = "m3e-base"
SELECTED_RETRIEVER = "1-stage"
TOPK_ABSNAME = 1
TOPK_KNOWLEDGE = 5
KNOWLEDGE_THRESHOLD = 0.80
HISTORY_TOPK = 5

# === Initialize LLM and Retriever ===
llm = load_llm(llm_used=SELECTED_LLM, device_map="auto")
one_stage_retriever, two_stage_retriever = load_retriever(
    embedding_used=SELECTED_EMBEDDING, device=0
)

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def handle_request():
    # SSE (Server-Sent Events) headers
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }

    print("Received request")

    def generate_response(data):
        # Example: could be filled with actual chat history if needed
        chat_history = []
        history = ""

        for item in chat_history:
            history += "用户：" + item["query"] + "\n" + "回答：" + item["result"] + "\n"

        summary_prompt = data["message"]
        print("Summary prompt:", summary_prompt)

        # Generate response using the LLM pipeline
        results = llmresponse(
            summary_prompt,
            TOPK_ABSNAME,
            TOPK_KNOWLEDGE,
            TOPK_ABSNAME,
            llm,
            KNOWLEDGE_THRESHOLD,
            SELECTED_RETRIEVER,
            one_stage_retriever,
            two_stage_retriever
        )

        result_text = results["result"]
        print("Raw result:", result_text)

        # Tokenize response for streaming
        for token in jieba.lcut(result_text):
            time.sleep(0.1)
            yield token

    # Handle incoming POST request
    if request.is_json:
        data = request.get_json()
        print("Request data:", data)
        return Response(generate_response(data), headers=headers)
    else:
        return jsonify({'error': 'Request must be JSON'}), 400


# === Main Entry Point ===
if __name__ == '__main__':
    app.run("0.0.0.0", port=1203, debug=False, use_reloader=False)