import sys

sys.path.append(r"..//common")
import time

from config import (
    CHAT_PROMPT_TEMPLATE,
    CHAT_PROMPT_TEMPLATE_GLM,
    RAG_PROMPT_TEMPLATE,
)
from custom_llm import (
    Baichuan2_7B,
    Baichuan2_13B,
    ChatGLM2_6B,
    ChatGLM4_9B,
    Internlm_7B,
    Internlm_20B,
    OpenAI_LLM,
    Qwen_7B,
    Qwen_14B,
)
from embedding import load_huggingface_embedding
from retriever import OneStageRetriever, SearchResult, TwoStageRetriever


def load_llm(llm_used, device_map):
    if llm_used == "chatglm2-6b":
        llm = ChatGLM2_6B(device_map=device_map)
    elif llm_used == "ChatGLM4-9B":
        llm = ChatGLM4_9B(device_map=device_map)
    elif llm_used == "Baichuan2-7B-Chat":
        llm = Baichuan2_7B(device_map=device_map)
    elif llm_used == "Baichuan2-13B-Chat":
        llm = Baichuan2_13B()
    elif llm_used == "internlm-chat-7b-v1_1":
        llm = Internlm_7B(device_map=device_map)
    elif llm_used == "internlm-chat-20b":
        llm = Internlm_20B(device_map=device_map)
    elif llm_used == "Qwen-7B-Chat":
        llm = Qwen_7B(device_map=device_map)
    elif llm_used == "Qwen-14B-Chat":
        llm = Qwen_14B(device_map=device_map)
    elif llm_used == "gpt-3.5-turbo-1106_azure":
        llm=OpenAI_LLM('gpt-3.5-turbo-1106_azure')
    elif llm_used == "gpt-4-1106-preview_azure":
        llm=OpenAI_LLM('gpt-4-1106-preview_azure')
    return llm

def load_retriever(embedding_used, device):
    embedding_model = load_huggingface_embedding(name=embedding_used, device=device)
    one_stage_retriever = OneStageRetriever(embedding_model=embedding_model)
    two_stage_retriever = TwoStageRetriever(embedding_model=embedding_model)
    return one_stage_retriever, two_stage_retriever

def llmresponse(query,topk_abstract,topk_knowledge,topk_name,llm,knowledge_threshold,retrieval_mode,ons_stage_retriever:OneStageRetriever,two_stage_retriever:TwoStageRetriever,his):
    print('\t'*5,"retrieval_mode:\t", retrieval_mode)
    if his==1:
        rag_prompt_template = RAG_PROMPT_TEMPLATE
        chat_prompt_template = CHAT_PROMPT_TEMPLATE
        chat_prompt_template_glm = CHAT_PROMPT_TEMPLATE_GLM
        #进行知识库的查找
        if retrieval_mode == "No Retrieval":
            search_result = SearchResult()
            if llm.model_name == 'glm4':
                reply_prompt = chat_prompt_template_glm.format(query=query)
            else:
                reply_prompt = chat_prompt_template.format(query=query)
        else:
            if retrieval_mode == "1-stage":
                retrieval_start_time = time.time()
                #ons_stage_retriever.updatestore()
                search_result = ons_stage_retriever.search(
                    query=query,
                    topk_knowledge=topk_knowledge,
                    knowledge_threshold=knowledge_threshold,
                )
                retrieval_end_time = time.time()
                print("retrieval_time", '\t'*3, retrieval_end_time-retrieval_start_time)
            elif retrieval_mode == "2-stage By Name":
                search_result = two_stage_retriever.search_by_name(
                    query=query,
                    topk_name=topk_name,
                    topk_knowledge=topk_knowledge,
                    knowledge_threshold=knowledge_threshold,
                )
            elif retrieval_mode == "2-stage By Abstract":
                search_result = two_stage_retriever.search_by_abstract(
                    query=query,
                    topk_abstract=topk_abstract,
                    topk_knowledge=topk_knowledge,
                    knowledge_threshold=knowledge_threshold,
                )

            if not search_result.knowledges:
                reply_prompt = chat_prompt_template.format(query=query)
            else:
                search_result.time = round(search_result.time, 3)
                reply_prompt = rag_prompt_template.format(
                        query=query, knowledge=search_result.knowledges_for_llm
                    )
    else:
        reply_prompt= query  
    if llm.model_name == "nihao":
        output=llm._call(reply_prompt)

    elif llm.model_name=='gpt-3.5-turbo-1106_azure' or llm.model_name=='gpt-4-1106-preview_azure':
        reply_prompt=[{"role":"system","content":reply_prompt}]
        output=llm._call(reply_prompt).choices[0].message.content
    else:
        output=llm._call(reply_prompt)
    # output=llm._call(reply_prompt)
    
    print("prompt",reply_prompt)
    #print("LLM_output",output)
    if his==1:
        return {"query":query,
                "knowledge":search_result.knowledges_for_llm,
                "prompt":reply_prompt,
                "result":output,
                "retrival_time": search_result.time,
                "knowledges": search_result.knowledges,
                "knowledge_scores": search_result.knowledge_scores,
                "knowledge_names": search_result.knowledge_names,
                "knowledge_ids": search_result.knowledge_ids,
                "names_or_abstracts": search_result.names_or_abstracts,
                "names_or_abstracts_scores": search_result.names_or_abstracts_scores,
            }
    else:
        return{"query":query,
                "prompt":reply_prompt,
                "result":output,
            }




