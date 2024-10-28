from __future__ import annotations

import os
import time
import json
import argparse
from termcolor import colored
from tqdm import tqdm
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from translate import LlamaInferce, STOP, LANGUAGES, PREFIX
from transformers import AutoTokenizer

from utils import load_jsonl, load_json

TIMEOUT_SECONDS = 40

def batch(iterable, n=-1):
    l = len(iterable)
    if n < 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generator(solver, output):
    solver.get_llm_response(output)
    return solver

def batch_main(args, llm, sampling_params, questions):
        
    try:
        solvers = [LlamaInferce(args, question) for question in questions]

        prompts = []
        stop_words = []
        for solver in solvers:
            prompt_text = solver.get_llm_request()
            prompts.append(prompt_text)
            stop_words.extend(STOP[args.template])
        
        if len(solvers) < 1:
            raise ValueError("No solver found.")
            
        sampling_params.stop = list(set(stop_words))
        outputs = llm.generate(prompts, sampling_params)

        with ProcessPool(max_workers=min(len(solvers), os.cpu_count())) as pool:
            future = pool.map(generator, solvers, outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()

        if len(solvers) > 100:  
            progress_bar = tqdm(total=len(solvers), desc="Execute")  
        else:  
            progress_bar = None 


        next_solvers = []
        while True:
            try:
                result = next(iterator)
                next_solvers.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                next_solvers.append(None)
                print(error)
            except Exception as error:
                print(error)
                next_solvers.append(None)
            if progress_bar is not None:
                progress_bar.update(1) 

        if progress_bar is not None:
            progress_bar.close() 

        solvers = next_solvers
        

    except Exception as e:
        print(colored(f"Exception: {e}", "red"))
        return [""] * len(questions)
    
    jsonlines = {}
    for solver in solvers:            
        try:
            response = solver.response
            jsonlines[solver.question] = response
        except:
            raise ValueError("Error in response generation")
    return jsonlines

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-q', '--question_file', type=str, default=None, help="file path of question file, should be jsonl format.")
    args.add_argument('--question_key', type=str, default="response", help="questioin key in json")
    args.add_argument('--save_dir', type=str, default=None, help="folder to save prediction file.")
    args.add_argument('--save_file', type=str, default=None, help="file to save prediction file.")
    args.add_argument('--num_per_inference', type=int, default=-1, help="number of questions per inference")

    args.add_argument('--xalpaca_eval', action="store_true", help="for x-alpacaeval")
    args.add_argument('--src_lang', type=str, default="en", help="source language")
    args.add_argument('--tgt_lang', type=str, default="fr", help="target language")
    args.add_argument('--with_prefix', type=lambda x: (str(x).lower() == 'true'), default=False)
    
    # model
    args.add_argument('-c', '--checkpoint_dir', type=str, default=None, help="folder of model checkpoint.")
    args.add_argument('--verbose', action="store_true", help="print intermediate result on screen")

    # llm 
    args.add_argument('--max_tokens', type=int, default=1024, help="decoding tokens")
    args.add_argument('--temperature', type=float, default=0, help="for sampling")
    args.add_argument('--top_k', type=int, default=-1, help="for sampling")
    args.add_argument('--top_p', type=float, default=1, help="for sampling")
    args.add_argument('--use_beam_search', action="store_true", help="use beam search")
    args.add_argument('--best_of', type=int, default=1, help="for beam search")
    args.add_argument('--n_generate_sample', type=int, default=1, help="number of generated samples")
    args.add_argument('--seed', type=int, default=1234, help="random seed.")
    args.add_argument('--repetition_penalty', type=float, default=1.0, help="repetition penalty")

    args.add_argument('--template', type=str, default="llama3", help="template")

    args = args.parse_args()
    return args

def input_format(args, d):
    if d.get('input', "") == "":
        if args.tgt_lang == "en":
            question = d[args.question_key]
        else:
            if args.with_prefix:
                question = PREFIX[args.tgt_lang] + d[args.question_key]
            else:
                question = d[args.question_key]
    else:
        if args.tgt_lang == "en":
            question = d[args.question_key] + "\n" + d['input']
        else:
            if args.with_prefix:
                question = PREFIX[args.tgt_lang] + d[args.question_key] + "\n" + d['input']
            else:
                question = d[args.question_key] + "\n" + d['input']
    # if args.verbose:
    #     print(colored(question, "blue"))
    return question

def main(args):
    # init llm
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', "0").split(',')
    llm = LLM(
        model=args.checkpoint_dir, 
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=False, 
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        top_k=args.top_k,
        best_of=args.best_of,
        use_beam_search=args.use_beam_search,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_generate_sample,
        repetition_penalty=args.repetition_penalty,
    )

    # load question file
    data = []

    if args.question_file.endswith(".jsonl"):
        data = load_jsonl(args.question_file)

    elif args.question_file.endswith(".json"):
        data = load_json(args.question_file)
    else:
        raise ValueError("Question file should be json or jsonl format.")           


    # data = data[:100]


    # write results
    if getattr(args, "save_dir", None) is None and getattr(args, "save_file", None) is None:
        raise ValueError("save_dir or save_file should be provided.")
    elif getattr(args, "save_dir", None) is not None and getattr(args, "save_file", None) is not None:
        raise ValueError("save_dir and save_file cannot be provided at the same time.")
    elif getattr(args, "save_dir", None) is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)
        save_file = os.path.join(args.save_dir, os.path.basename(args.question_file) + f".to_{args.tgt_lang}.jsonl")
    elif getattr(args, "save_file", None) is not None:
        if not os.path.exists(os.path.dirname(args.save_file)):
            os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
        save_file = args.save_file
    



    with open(save_file, "w") as writer:
        for cur_data in tqdm(batch(data, args.num_per_inference), desc="Main Processing"):
            # load question
            questions = []
            for d in cur_data:
                question = d[args.question_key]
                questions.append(question)
            
            # inference
            jsonlines = batch_main(args, llm, sampling_params, questions)
            
            # write response
            for d in cur_data:
                question = d[args.question_key]
                text = d

                import re
                pattern = re.compile(r"<translated>(.*?)</translated>", re.DOTALL)
                matches = pattern.search(jsonlines[question])

                if matches:
                    text[f"{args.src_lang}2{args.tgt_lang}_reponse"] = matches.group(1).strip()
                else:
                    print(colored(f"Translation Error: {jsonlines[question]}", "red"))
                    text[f"{args.src_lang}2{args.tgt_lang}_reponse"] = "Translation Error"


                writer.write(json.dumps(text, ensure_ascii=False) + '\n')       
                writer.flush()

if __name__ == '__main__':
    args = parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    main(args)