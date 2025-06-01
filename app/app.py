from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# 渲染HTML页面
@app.route('/')
def index():
    return render_template('index.html')

# 处理POST请求
@app.route('/ask', methods=['POST'])
def ask():
    # 获取POST请求的数据
    data = request.get_json()
    text = data['message']
    
    # 替换为你的接口地址
    url = "http://10.48.8.76:1202"
    # 调用接口
    response = requests.post(url=url, json={"message": text})
    answer = response.json()['answer']  # 假设接口返回的JSON中有一个名为'answer'的字段
    
    # 返回JSON格式的回答
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
