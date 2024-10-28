from __future__ import annotations

import re 
import re
import argparse
from typing import List, Dict
from termcolor import colored

from vllm.outputs import RequestOutput
from timeout_decorator import timeout

from transformers import AutoTokenizer

from trans_prompts import TRANS_INSTRUCTION

# # sft llama2 prompt
# SFT_PROMPT = (
#     "<sys>You are a helpful assistant.</sys>\n\n"
#     "<user>{instruction}</user>\n\n"
#     "<assistant>"
# )

# llama-3-chat prompt

# PREFIX = (
#     "Please answer the following question in {language}.\n\n"
# )
PREFIX = {
    "zh": "请用中文回答下面的问题。\n\n",
    "en": "Please answer the following question in English.\n\n",
    "es": "Por favor responda la siguiente pregunta en español.\n\n",
    "it": "Si prega di rispondere alla seguente domanda in italiano.\n\n",
    "ko": "다음 질문에 대한 답변을 한국어로 작성해주세요.\n\n",
}


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
    )
}


STOP = {
    "llama2": ["\n</s>", "</s>"],
    "llama3": ["<|end_of_text|>", "<|eot_id|>"],
    "qwen": ["<|im_end|>"],
    "aya": ["<|END_OF_TURN_TOKEN|>"],
}

LANGUAGES= {
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

PRIMER = "<translated>\n"

class LlamaInferce(object):
    def __init__(self, args, question: str):
        self.args = args
        self.question = question

        self.response = ""
        self.reponses = []
    
    def get_llm_request(self) -> str:
        input_prompt = TRANS_INSTRUCTION.format(lang=LANGUAGES[self.args.tgt_lang], text=self.question)
        prompt = SFT_PROMPT[self.args.template].format(instruction=input_prompt)
        prompt = prompt + PRIMER
        if self.args.verbose:
            print(colored(prompt, "red"))
        return prompt
    
    def get_llm_response(self, output: RequestOutput) -> None:
        response = output.outputs[0].text.strip()
        response = PRIMER + response
        if self.args.verbose:
            print(colored(response, "green"))
        self.response = response
        self.reponses.append(response)

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument('-c', '--checkpoint_dir', type=str, default="/mnt/workspace/workgroup/huaike.wc/pretrained_models/transformers/Meta-Llama-3-8B-Instruct", help="folder of model checkpoint.")

    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")
    args.add_argument('--temperature', type=float, default=0.8, help="for sampling")

    args.add_argument('-q', '--question', type=str, default=None, help="question")

    args.add_argument('--template', type=str, default="llama3", help="template")
    args.add_argument('--lang', type=str, default="en", help="language of the question")

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


    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    # messages = [
    #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    #     {"role": "user", "content": "When did Virgin Australia start operating?\n\nVirgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It is the largest airline by fleet size to use the Virgin brand. It commenced services on 31 August 2000 as Virgin Blue, with two aircraft on a single route.[3] It suddenly found itself as a major airline in Australia's domestic market after the collapse of Ansett Australia in September 2001. The airline has since grown to directly serve 32 cities in Australia, from hubs in Brisbane, Melbourne and Sydney.[4]"},
    # ]

    # question = tokenizer.apply_chat_template(messages,
    #     tokenize=False, 
    #     add_generation_prompt=True)

    # import pdb; pdb.set_trace()

    responser = LlamaInferce(args, question)

    # run reponser
    prompt = responser.get_llm_request()
    prompts = [prompt]
    outputs = llm.generate(prompts, sampling_params)
    responser.get_llm_response(outputs[0])


    
    # # save responses
    # full_reponses = "\n\n".join(reponser.reponses)
    # with open("log.txt", "w") as f:
    #     f.write(full_reponses)


