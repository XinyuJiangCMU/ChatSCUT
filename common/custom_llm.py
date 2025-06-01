# coding=gbk
from typing import Any, Optional
from langchain.llms.base import LLM
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from config import llms_config
import torch
import requests
import os
import time
from typing import Literal
from openai import OpenAI, AzureOpenAI
class ChatGLM2_6B(LLM):
    tokenizer: Any
    model: Any
    messages: list = []
    model_name : str = "glm4"

    def __init__(self, device_map):
        super().__init__()

        model_name: str = llms_config["chatglm2-6b"]["model_name"]
        model_path: str | None = llms_config["chatglm2-6b"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path, trust_remote_code=True, device_map=device_map
        )

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {"model_name": "ChatGLM2-6B"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        if history_turns:
            if len(self.messages) > history_turns:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages[-history_turns:-1],
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages,
                )
            self.messages.append((prompt, response))
        else:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response
    
from pathlib import Path
from typing import Union
import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, encode_special_tokens=True, use_fast=False
    )
    return model, tokenizer

class ChatGLM4_9B(LLM):
    tokenizer: Any
    model: Any
    messages: list = []
    model_name : str = "glm4"

    def __init__(self, device_map):
        super().__init__()

         # model_path: str | None = llms_config["chatglm2-6b"]["model_path"]
        # model_name_or_path = model_path if model_path else model_name
        self.model,self.tokenizer = load_model_and_tokenizer("/home/pci/work/zbw/model/lora")

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {"model_name": "ChatGLM4-9B"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        if history_turns:
            if len(self.messages) > history_turns:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages[-history_turns:-1],
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages,
                )
            self.messages.append((prompt, response))
        else:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class Baichuan2_7B(LLM):
    tokenizer: Any = None
    model: Any = None

    def __init__(self, device_map):
        super().__init__()

        model_name = llms_config["Baichuan2-7B-Chat"]["model_name"]
        model_path = llms_config["Baichuan2-7B-Chat"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True,
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "Baichuan2-7B-Chat"}

    def _call(self, prompt: str, stop: Optional = None):
        messages = [{"role": "user", "content": prompt}]
        response = self.model.chat(self.tokenizer, messages)
        return response

    def stream_chat(self, prompt: str):
        messages = [{"role": "user", "content": prompt}]
        response = self.model.chat(self.tokenizer, messages, stream=True)
        return response

    def batch_generate(self, prompts: list) :
        prompts = [f"<reserved_106>{p}<reserved_107>" for p in prompts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        generation_config = self.model.generation_config
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [r.split("<reserved_107>")[-1] for r in responses]

        return responses


class Baichuan2_13B():
    def __init__(self):
        super().__init__()
    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "Baichuan2-13B-Chat"}
    def _call(self,prompt, stop: Optional = None):
        url="http://116.57.86.190:8081/chat"
        data = {
    	       "model_name" : "Baichuan2-13B-Chat",
    	       "stream" : False,
    	       "messages" : [
        	{"role": "user", "content":prompt}
    			],
    	    "generation_config" : None
		}
        s = requests.session()
        s.keep_alive = False
        response = s.post(url=url, json=data)
        if response.status_code == 200:
            text = response.text
        else:
            print("Error:", response.status_code, response.text)
        return text
    def stream_chat(self,prompt):
        url="http://116.57.86.190:8081/chat"
        data = {
    	       "model_name" : "Baichuan2-13B-Chat",
    	       "stream" : False,
    	       "messages" : [
        	{"role": "user", "content":prompt}
    			],
    	    "generation_config" : None
		}
        response = requests.post(url=url, json=data)
        if response.status_code == 200:
            text = ""
            for new_token in response.iter_content(decode_unicode=True, chunk_size=1024):
                text += new_token
        else:
            print("Error:", response.status_code, response.text)
        return text

class Internlm_7B(LLM):
    tokenizer: Any = None
    model: Any = None
    messages: list = []

    def __init__(self, device_map):
        super().__init__()

        model_name = llms_config["internlm-chat-7b-v1_1"]["model_name"]
        model_path = llms_config["internlm-chat-7b-v1_1"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "internlm-chat-7b-v1_1"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        if history_turns:
            if len(self.messages) > history_turns:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages[-history_turns:-1],
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages,
                )
            self.messages.append((prompt, response))
        else:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class Internlm_20B(LLM):
    tokenizer: Any = None
    model: Any = None
    messages: list = []

    def __init__(self, device_map):
        super().__init__()

        model_name = llms_config["internlm-chat-20b"]["model_name"]
        model_path = llms_config["internlm-chat-20b"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "internlm-chat-20b"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        if history_turns:
            if len(self.messages) > history_turns:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages[-history_turns:-1],
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages,
                )
            self.messages.append((prompt, response))
        else:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response


class Qwen_7B(LLM):
    tokenizer: Any = None
    model: Any = None
    messages: list = []

    def __init__(self, device_map):
        super().__init__()

        model_name = llms_config["Qwen-7B-Chat"]["model_name"]
        model_path = llms_config["Qwen-7B-Chat"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=False, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path
        )

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "Qwen-7B-Chat"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        if history_turns:
            if len(self.messages) > history_turns:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages[-history_turns:-1],
                )
            else:
                response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=self.messages,
                )
            self.messages.append((prompt, response))
        else:
            
            def chat(model, tok, ques, history=[]):
                iids = tok.apply_chat_template(
                    history + [{'role': 'user', 'content': ques}], 
                    add_generation_prompt=1,
                )
                oids = model.generate(
                    inputs=torch.tensor([iids]).to(model.device),
                    **(model.generation_config.to_dict()),
                    max_length=100000,
                    temperature=0.8,
                )
                oids = oids[0][len(iids):].tolist()
                if oids[-1] == tok.eos_token_id:
                    oids = oids[:-1]
                ans = tok.decode(oids)
                
                return ans
            
            response, _ = chat(self.model,self.tokenizer, prompt, history=[])

        return response


class Qwen_14B(LLM):
    tokenizer: Any = None
    model: Any = None
    messages: list = []

    def __init__(self, device_map):
        super().__init__()

        model_name = llms_config["Qwen-14B-Chat"]["model_name"]
        model_path = llms_config["Qwen-14B-Chat"]["model_path"]
        model_name_or_path = model_path if model_path else model_name

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        return {"name": "Qwen-14B-Chat"}

    def _call(self, prompt: str, stop: Optional = None, history_turns=0):
        url="http://116.57.86.190:8081/chat"
        data = {
    	       "model_name" : "Qwen-14B-Chat",
    	       "stream" : False,
    	       "messages" : [
        	{"role": "user", "content":prompt}
    			],
    	    "generation_config" : None
		}
        response = requests.post(url=url, json=data)
        if response.status_code == 200:
            text = response.text
        else:
            print("Error:", response.status_code, response.text)
        return text

class OpenAI_LLM:
    '''
        Azure(https://portal.azure.com):
            'gpt-3.5-turbo-1106_azure': 'GPT-35',
            'gpt-4-1106-preview_azure': 'GPT4',
        API2D(https://api2d.com/wiki/doc):
            'gpt-3.5-turbo-1106_api2d': 'gpt-3.5-turbo-1106',
            'gpt-3.5-turbo-16k-0613_api2d': 'gpt-3.5-turbo-16k-0613',
            'gpt-3.5-turbo-16k_api2d': 'gpt-3.5-turbo-16k',
            'gpt-3.5-turbo-0613_api2d': 'gpt-3.5-turbo-0613',
            'gpt-3.5-turbo_api2d': 'gpt-3.5-turbo',
            'gpt-3.5-turbo-0301_api2d': 'gpt-3.5-turbo-0301',
            'gpt-4-1106-preview_api2d': 'gpt-4-1106-preview',
            'gpt-4-0613_api2d': 'gpt-4-0613',
            'gpt-4_api2d': 'gpt-4',
            'qwen-72b-chat_hefei': 'qwen-72b-chat',
            'qwen-14b-chat_hefei': 'qwen-14b-chat',
    '''

    def __init__(self, model_name):
        self.model_name = model_name
        if model_name.endswith("_hefei"):
            self.client = OpenAI(
                base_url="https://dev.iai007.cloud/ai/api/v1",
                api_key="Hh8VazfByLx3fONHdrW_6muIBvfhZhfh",
            )
            self.model = model_name.lower().split("_")[0]

        elif model_name.endswith("_api2d"):
            self.client = OpenAI(
                base_url="https://openai.api2d.net/v1",
                api_key="fk204884-QxqDEvomnE6PRVe6WBxEnVcC8v88TxSL",
            )
            self.model = model_name.split("_")[0]
            if "gpt-4" in model_name:
                now = time.localtime()
                current_date = time.strftime("%Y-%m", now)
                self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-04\nCurrent date: {current_date}'
            elif "gpt-3.5" in model_name:
                now = time.localtime()
                current_date = time.strftime("%Y-%m", now)
                self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'


        elif model_name.endswith("_azure"):
            if "gpt-4" in model_name:
                self.client = AzureOpenAI(
                    azure_endpoint="https://zhishenggpt40.openai.azure.com/",
                    api_key='',
                    api_version="2024-02-15-preview",
                )
                self.model = "GPT4"
                now = time.localtime()
                current_date = time.strftime("%Y-%m", now)
                self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2023-04\nCurrent date: {current_date}'
            elif "gpt-3.5" in model_name:
                self.client = AzureOpenAI(
                    azure_endpoint="https://zhishenggpt.openai.azure.com/",
                    api_key='',
                    #api_key="",
                    api_version="2024-02-15-preview",
                )
                self.model = "GPT-35"
                now = time.localtime()
                current_date = time.strftime("%Y-%m", now)
                self.system_prompt = f'You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'
            else:
                raise ValueError(f"Unsupported model name: {model_name}")

        elif model_name.endswith("_kimi"):
            self.system_prompt = ''
            self.client = OpenAI(
                base_url="https://api.moonshot.cn/v1",
                api_key="",
            )
            self.model = model_name.split("_")[0]


        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def _call(
            self,
            messages,
            generation_config=None,
            temperature=0.7,
            max_tokens=4096,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False,
            add_system_prompt=False,
    ):
        if add_system_prompt:
            if self.model_name.endswith("_api2d"):
                if messages[0]["role"] != "system":
                    messages = [{"role": "system", "content": self.system_prompt}] + messages 

            elif self.model_name.endswith("_azure"):
                if messages[0]["role"] != "system":
                    messages = [{"role": "system", "content": self.system_prompt}] + messages 

            else:
                if messages[0]["role"] != "system":
                    messages = [{"role": "system", "content": self.system_prompt}] + messages 

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=stream  
        )

        return completion