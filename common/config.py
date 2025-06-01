import os
import sys
sys.path.append(r"..//common")
PROJ_TOP_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

huggingface_embeddings_config = {
    "m3e-base": {
        "model_name": "moka-ai/m3e-base",
        "model_path": os.path.join(PROJ_TOP_DIR,"models", "m3e-base"),
        "encode_kwargs": {"normalize_embeddings": True},
        "max_len": 512,
    },
    # "m3e-large": {
    #     "model_name": "moka-ai/m3e-large",
    #     "model_path": os.path.join(PROJ_TOP_DIR, "models", "m3e-large"),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 512,
    # },
    # "text2vec-base-chinese": {
    #     "model_name": "shibing624/text2vec-base-chinese",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model", "text2vec-base-chinese"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 512,
    # },
    # "text2vec-large-chinese": {
    #     "model_name": "GanymedeNil/text2vec-large-chinese",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model", "text2vec-large-chinese"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 512,
    # },
    # "bge-base-zh-v1.5": {
    #     "model_name": "BAAI/bge-base-zh-v1.5",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model", "bge-base-zh-v1.5"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 512,
    # },
    # "bge-large-zh-v1.5": {
    #     "model_name": "BAAI/bge-large-zh-v1.5",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model",  "bge-large-zh-v1.5"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 512,
    # },
    # "stella-base-zh-v2": {
    #     "model_name": "infgrad/stella-base-zh",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model",  "stella-base-zh-v2"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 1024,
    # },
    # "stella-large-zh-v2": {
    #     "model_name": "infgrad/stella-large-zh",
    #     "model_path": os.path.join(
    #         PROJ_TOP_DIR, "model",  "stella-large-zh-v2"
    #     ),
    #     "encode_kwargs": {"normalize_embeddings": True},
    #     "max_len": 1024,
    # },
}

llms_config = {
    "Baichuan2-7B-Chat": {
        "model_name": "baichuan-inc/Baichuan2-7B-Chat",
        "model_path": os.path.join(
            "../models/Baichuan2-7B-Chat"
        ),
    },
    "Baichuan2-13B-Chat": {
        "model_name": "baichuan-inc/Baichuan2-13B-Chat",
        "model_path": os.path.join(
            # "/home/ShareFiles/PretrainedModel/models.huggingface.co/Baichuan2-13B-Chat"
            # "/home/ShareFiles/PretrainedModel/models.huggingface.co/baichuan2-13b-KBNutritionHealthLLM-v1-sft-epoch-2-lr-constant_with_warmup-2e-5"
            #"/home/ShareFiles/PretrainedModel/models.huggingface.co/Baichuan2-13B-Chat"
            "../models/Baichuan2-13B-Chat"
        ),
    },
    "chatglm2-6b": {
        "model_name": "THUDM/chatglm2-6b",
        "model_path": os.path.join(
            #"/home/ShareFiles/PretrainedModel/models.huggingface.co/chatglm2-6b"
            "models/chatglm2-6b"
        )
    },
    "internlm-chat-7b-v1_1": {
        "model_name": "internlm/internlm-chat-7b-v1_1",
        "model_path": os.path.join(
            "../models/internlm-chat-7b-v1_1"
        )
    },
    "internlm-chat-20b": {
        "model_name": "internlm/internlm-chat-20b",
        "model_path": os.path.join(
            "../models/internlm-chat-20b"
        )
    },
    "Qwen-7B-Chat": {
        "model_name": "Qwen/Qwen-7B-Chat",
        "model_path": os.path.join(
            "models/Qwen-7B-Chat"
        )
    },
    "Qwen-14B-Chat": {
        "model_name": "Qwen/Qwen-14B-Chat",
        "model_path": os.path.join(
            "../models/Qwen-14B-Chat"
        )
    },
}


SUMMARY_PROMPT_TEMPLATE = """\
你是一位专业的中文语言学专家，给定如下【多轮对话历史】和最新一轮的问题，请把最新一轮问题补充为语义完整的句子，方便摆脱对话历史进行知识库检索。

【多轮对话历史】
{history}

请结合【多轮对话历史】，分析最新一轮用户的问题“{query}”是否存在信息缺失，如果存在信息缺失，将这句话**改写为语义完整、可以单独理解的句子**，但是不能引入对话历史中不存在的信息；如果这句话信息完整，则直接返回该句话，方便摆脱对话历史进行知识库检索。

返回格式：
用户：{{问题}}\
"""


RAG_PROMPT_TEMPLATE = """\
请参考以下知识，结合你已有的知识回答问题。


{knowledge}

问题:{query}

请针对以上问题，结合参考知识给出详细的、自然的、对用户有帮助的回答。\
"""


CHAT_PROMPT_TEMPLATE = """\
请针对以下问题，给出详细的、自然的、对用户有帮助的回答。

问题:{query}\
"""

CHAT_PROMPT_TEMPLATE_GLM = """\
请针对以下问题，给出详细的、自然的、对用户有帮助的回答。注意，你的回答不能太长，不能超过150字。

问题:{query}\
"""

SUMMARY_PROMPT_TEMPLATE = """\
以下是【多轮对话历史】和最新一轮的问题，请把最新一轮问题补充为带有对话历史背景的问题，方便之后对问题进行知识库检索。

【多轮对话历史】
{history}

请结合【多轮对话历史】，分析最新一轮用户的问题“{query}”,改为一段带有对话历史信息叙述的问题，但是不能引入对话历史中不存在的信息。

返回格式：
用户：{{问题}}\
"""


AGENT_PROMPT_TEMPLATE = """\
你是一个智能助手，负责根据用户的输入判断是否需要检索数据库。数据库包含有关学生的学习和生活的相关资料。

当用户提出问题时，你需要分析问题的内容，并判断是否需要从数据库中获取相关信息。如果问题涉及到学生的学习和生活，请返回"YES"。否则，请返回"NO"。

以下是一些示例输入和相应的判断：

输入："请问华南理工大学的校长是谁？"
判断："YES"

输入："今天的天气怎么样？"
判断："NO"

输入："计算机学院有哪些专业？"
判断："YES"

输入："给我推荐一些学习Python的书籍。"
判断："YES"

输入："你能给我讲个笑话吗？"
判断："NO"

输入："你好？"
判断："NO"
请根据以上规则，分析用户的输入并进行判断。

注意，你只能回答YES或者NO，不能回答任何其他有关的问题，以下是用户的问题：
{query}

\
"""