"""
Usage:
    CUDA_VISIBLE_DEVICES=0 streamlit run app.py --server.port 9030 
"""

import sys
sys.path.append(r"./common")

import streamlit as st
from streamlit_chat import message
from func import *

# Set page title
st.title("ChatSCUT")

# Sidebar configuration
with st.sidebar:
    st.subheader("模型的参数选则")

    add_selectllm = st.sidebar.selectbox(
        "采用的基础对话模型：",
        (
            "gpt-4-1106-preview_azure", "gpt-3.5-turbo-1106_azure",
            "Baichuan2-13B-Chat","chatglm2-6b","Baichuan2-7B-Chat",
            "internlm-chat-7b-v1_1","internlm-chat-20b",
            "Qwen-7B-Chat","Qwen-14B-Chat"
        )
    )

    selected_embedding = st.selectbox(
        "采用的embedding模型：",
        (
            "m3e-base", "m3e-large",
            "text2vec-base-chinese", "text2vec-large-chinese",
            "bge-base-zh-v1.5", "bge-large-zh-v1.5",
            "stella-base-zh-v2", "stella-large-zh-v2"
        )
    )

    selected_retrieval = st.selectbox(
        "您选则的检索方式：",
        ("No Retriaval", "1-stage", "2-stage By Abstract", "2-stage By Name")
    )

    topk_absname = st.slider('检测的文档数：', min_value=1, max_value=10, value=1)
    topk_knowledge = st.slider('检测的知识点数：', min_value=1, max_value=10, value=5)
    
    knowledge_threshold = st.number_input(
        "检测相似度选则",
        value=0.80,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
    )

    his_k = st.slider('记忆的历史轮数：', min_value=1, max_value=10, value=5)
# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load prompt and settings
summary_prompt_template = SUMMARY_PROMPT_TEMPLATE
retrieval_mode = selected_retrieval

# Load LLM and retrievers
llm = load_llm(llm_used=selected_llm, device_map="auto")
ons_stage_retriever, two_stage_retriever = load_retriever(
    embedding_used=selected_embedding,
    device=0
)

# Introduction message
st.markdown("#### 我是华南理工大学聊天机器人,我可以回答您问题!")

# Reset button
if st.button("重新开启对话"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state.chat_history = []

# User input and chat interaction
user_input = st.chat_input()
if user_input:
    # Display chat history
    for i in range(len(st.session_state['generated'])):
        with st.chat_message(name="user"):
            st.write(st.session_state['past'][i])
        with st.chat_message(name="assistant"):
            st.write(st.session_state['generated'][i])

    # Display current user input
    with st.chat_message(name="user"):
        st.write(user_input)

    # Build history string for summarization
    history_text = ""
    for entry in st.session_state.chat_history:
        history_text += f"用户：{entry['query']}\n回答：{entry['result']}\n"

    # Prepare summarization prompt
    summary_prompt = summary_prompt_template.format(
        history=history_text,
        query=user_input
    )

    # Call LLM to summarize if needed
    if llm.model_name in ['gpt-3.5-turbo-1106_azure', 'gpt-4-1106-preview_azure']:
        summary_prompt = [{"role": "system", "content": summary_prompt}]
        summary_prompt = llm._call(summary_prompt).choices[0].message.content
    else:
        summary_prompt = llm._call(summary_prompt)

    # If no history, use raw query
    if history_text == "":
        summary_prompt = user_input

    # Get response from llmresponse
    result = llmresponse(
        summary_prompt,
        topk_abstract=topk_absname,
        topk_knowledge=topk_knowledge,
        topk_name=topk_absname,
        llm=llm,
        knowledge_threshold=knowledge_threshold,
        retrieval_mode=retrieval_mode,
        ons_stage_retriever=ons_stage_retriever,
        two_stage_retriever=two_stage_retriever
    )

    # Store chat history
    st.session_state.chat_history.append({
        "query": user_input,
        "result": result["result"]
    })
    output = result["result"]

    # Keep only last `his_k` messages
    if len(st.session_state.chat_history) > (his_k - 1):
        st.session_state.chat_history = st.session_state.chat_history[-(his_k - 1):]

    # Append to session state
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

    # Show sidebar debug info
    st.sidebar.text_area(label="**检索到的知识点**", value=result["knowledge"])
    st.sidebar.text_area(label="**总结历史的prompt**", value=summary_prompt)

    # Show final answer
    with st.chat_message(name="assistant"):
        st.write(output)