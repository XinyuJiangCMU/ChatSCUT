class SimpleAgent:
    def __init__(self):
        self.history = []
        self.retrieval_mode = "No Retrieval"  # Default mode

    def preprocess_message(self, message):
        print("agent关于是否需要调用RAG的回答：",message)
        # 判断是否需要检索数据库
        if "YES" in message:
            return "1-stage"
        return "No Retrieval"