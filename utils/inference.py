from __future__ import annotations

import re 
import re
import argparse
from typing import List, Dict
from termcolor import colored

from vllm.outputs import RequestOutput
from timeout_decorator import timeout

from transformers import AutoTokenizer

PROMPT_PREFIX = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request in {language}. Please answer in {language}.\n\n### Instruction:\n"
    "{question}\n\n### Response:\n"
)

SFT_PROMPT = {
    "llama2": (
        "<s>[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, "
        "racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, "
        "explain why instead of answering something not correct. "
        "If you don't know the answer to a question, please don't share false information."
        "\n<</SYS>>\n\n{instruction} [/INST]"
    ),
    "llama3": (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "qwen": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "aya": (
        "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{instruction}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
    ),
    "gemma2": (
        "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "mistral": (
        "<s>[INST] {instruction} [/INST]"
    ),
    "bloomz": (
        "{instruction}</s>"
    ),
    "intern2": (
        "<s><|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "polylm": (
        "{instruction}\n\n"
    ),
}

STOP = {
    "llama2": ["\n</s>", "</s>"],
    "llama3": ["<|end_of_text|>", "<|eot_id|>"],
    "qwen": ["<|im_end|>"],
    "aya": ["<|END_OF_TURN_TOKEN|>"],
    "gemma2": ["<|im_end|>"],
    "mistral": ["</s>"],
    "bloomz": ["</s>"],
    "intern2": ["</s>", "<|im_end|>"],
    "polylm": ["</s>"],
}

LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "it": "Italian",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "ja": "Japanese",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "th": "Thai",
}

class LlamaInferce(object):
    def __init__(self, args, question: str):
        self.args = args
        self.question = question

        self.response = ""
        self.reponses = []
    
    def get_llm_request(self) -> str:
        prompt = SFT_PROMPT[self.args.template].format(instruction=self.question)
        if self.args.verbose:
            print(colored(prompt, "red"))
        return prompt
    
    def get_llm_response(self, output: RequestOutput) -> None:
        response = output.outputs[0].text.strip()
        if self.args.verbose:
            print(colored(response, "green"))
        self.response = response
        self.reponses.append(response)

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('-c', '--checkpoint_dir', type=str, default="/pretrained_models/transformers/Meta-Llama-3-8B-Instruct", help="folder of model checkpoint.")

    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--temperature', type=float, default=0.8, help="for sampling")

    args.add_argument('-q', '--question', type=str, default=None, help="question")

    args.add_argument('--template', type=str, default="llama3", help="template")

    args = args.parse_args()
    return args
    
if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    llm = LLM(
        model = args.checkpoint_dir,
        tensor_parallel_size=1,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=-1,
        top_p=0.95,
        use_beam_search=False,
        best_of=1,
        max_tokens=2048, 
        n=1, 
        stop=STOP[args.template]
    )

    # define question and solver
    if args.question:
        question = args.question
    else:
        # an example question
        question = "When did Virgin Australia start operating?"
    if args.verbose:
        print(colored(f"Question: {question}\n", "yellow"))



    responser = LlamaInferce(args, question)

    # run reponser
    prompt = responser.get_llm_request()
    prompts = [prompt]
    outputs = llm.generate(prompts, sampling_params)
    responser.get_llm_response(outputs[0])


    