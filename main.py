import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import save_results, load_json, setup_seeds, clean_str, clean_str_to_words
from src.attack import Attacker
from src.prompts import wrap_prompt, response_check_prompt
from src.llm_api import chat_completion
import torch
import requests
import time  # 添加到文件最上方
 
def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="contriever")
    parser.add_argument('--eval_dataset', type=str, default="nq", help='BEIR dataset to evaluate')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--orig_beir_results", type=str, default=None, help='Eval results of eval_model on the original beir eval_dataset')
    parser.add_argument("--query_results_dir", type=str, default='main')

    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--response_check_model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument('--response_check_model_name', type=str, default='gpt4')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_truth', type=str, default='False')
    parser.add_argument('--gpu_id', type=int, default=0)

    # attack
    parser.add_argument('--attack_method', type=str, default='LM_targeted')
    parser.add_argument('--adv_per_query', type=int, default=5, help='The number of adv texts for each target query.')
    parser.add_argument('--score_function', type=str, default='dot', choices=['dot', 'cos_sim'])
    parser.add_argument('--repeat_times', type=int, default=10, help='repeat several times to compute average')
    parser.add_argument('--M', type=int, default=10, help='one of our parameters, the number of target queries')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument("--name", type=str, default='debug', help="Name of log and result.")
    
    parser.add_argument('--poison_ratio', type=float, default=0.2,
                    help='The ratio of adversarial texts to inject into the top-k context (0.0 to 1.0).')
    
    parser.add_argument('--diverse_n', type=int, default=3, help='Number of contexts to select in diverse selection')
    args = parser.parse_args()
    print(args)
    return args

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = "cuda"
    setup_seeds(args.seed)

    if args.model_config_path is None:
        args.model_config_path = f"model_configs/{args.model_name}_config.json"
    args.response_check_model_config_path = f"model_configs/{args.response_check_model_name}_config.json"

    # Load corpus/queries/qrels and target queries
    corpus, queries, qrels = load_beir_datasets(
        "msmarco" if args.eval_dataset == "msmarco" else args.eval_dataset,
        "train" if args.eval_dataset == "msmarco" else args.split
    )
    incorrect_answers = load_json(f"results/target_queries/{args.eval_dataset}.json")

    # Load BEIR results (auto-fill path if missing)
    if args.orig_beir_results is None:
        suffix = "" if args.split == "test" else "-dev"
        cos_suffix = "-cos" if args.score_function == "cos_sim" else ""
        args.orig_beir_results = f"results/beir_results/{args.eval_dataset}-{args.eval_model_code}{cos_suffix}{suffix}.json"
    assert os.path.exists(args.orig_beir_results), f"Missing BEIR results at {args.orig_beir_results}"
    with open(args.orig_beir_results) as f:
        results = json.load(f)
    print("Total samples:", len(results))

    if args.use_truth == "True":
        args.attack_method = None

    attacker = None
    if args.attack_method not in [None, "None"]:
        model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
        model.eval().to(device)
        c_model.eval().to(device)
        attacker = Attacker(args, model=model, c_model=c_model, tokenizer=tokenizer, get_emb=get_emb)

    llm = create_model(args.model_config_path)

    # Statistics accumulators
    asr_list, tru_list, ret_list, retain_poison_list, all_time = [], [], [], [], []
    total_poisoned_samples = 0
    poison_retained_cnt = 0
    d_list = []

    for iter_idx in range(args.repeat_times):
        print(f"Iter {iter_idx+1}/{args.repeat_times}")
        base = iter_idx * args.M
        target_indices = range(base, base + args.M)

        # Prepare target queries with top1 score if attack is enabled
        target_queries = []
        if args.attack_method not in [None, "None"]:
            for i in target_indices:
                qid = incorrect_answers[i]["id"]
                top1_idx = next(iter(results[qid].keys()))
                top1_score = results[qid][top1_idx]
                target_queries.append({
                    "query": incorrect_answers[i]["question"],
                    "top1_score": top1_score,
                    "id": qid
                })
            adv_text_groups = attacker.get_attack(target_queries)
            adv_text_list = sum(adv_text_groups, [])
            adv_input = tokenizer(adv_text_list, padding=True, truncation=True, return_tensors="pt")
            adv_input = {k: v.cuda() for k, v in adv_input.items()}
            with torch.no_grad():
                adv_embs = get_emb(c_model, adv_input)
        else:
            adv_text_groups, adv_text_list, adv_embs = [], [], None

        iter_asr, iter_tru, iter_retain = 0, 0, 0
        iter_d_list, iter_time_list = [], []

        for offset, i in enumerate(target_indices):
            qid = incorrect_answers[i]["id"]
            question = incorrect_answers[i]["question"]
            incorrect_ans = incorrect_answers[i]["incorrect answer"]
            correct_ans = incorrect_answers[i]["correct answer"]
            gt_ids = list(qrels[qid].keys())
            ground_truth = [corpus[id]["text"] for id in gt_ids]

            # Truth evaluation path
            if args.use_truth == "True":
                prompt = wrap_prompt(question, ground_truth, 4)
                response = llm.query(prompt)
                is_true = clean_str(correct_ans) in clean_str(response)
                iter_tru += int(is_true)
                iter_result = {
                    "question": question,
                    "input_prompt": prompt,
                    "output": response,
                    "correct_answer": correct_ans
                }
            else:
                # Build top-k results
                topk_idx = list(results[qid].keys())[: args.top_k]
                topk_results = [
                    {"score": results[qid][idx], "context": corpus[idx]["text"]} for idx in topk_idx
                ]

                # Attack path: augment with adversarial examples
                if args.attack_method not in [None, "None"]:
                    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
                    query_input = {k: v.cuda() for k, v in query_input.items()}
                    with torch.no_grad():
                        query_emb = get_emb(model, query_input)
                    # compute similarity with each adv
                    for j, adv_text in enumerate(adv_text_list):
                        adv_emb = adv_embs[j].unsqueeze(0)
                        if args.score_function == "dot":
                            sim = torch.mm(adv_emb, query_emb.T).cpu().item()
                        elif args.score_function == "cos_sim":
                            sim = torch.cosine_similarity(adv_emb, query_emb).cpu().item()
                        topk_results.append({"score": sim, "context": adv_text})
                    topk_results = sorted(topk_results, key=lambda x: x["score"], reverse=True)

                    # Inject poison based on ratio
                    num_poison = int(args.top_k * args.poison_ratio)
                    num_clean = args.top_k - num_poison
                    adv_text_map = {txt: idx for idx, txt in enumerate(adv_text_groups[offset])}
                    adv_entries = [r for r in topk_results if r["context"] in adv_text_map]
                    clean_entries = [r for r in topk_results if r["context"] not in adv_text_map]
                    adv_entries = sorted(adv_entries, key=lambda x: -x["score"])[:num_poison]
                    clean_entries = sorted(clean_entries, key=lambda x: -x["score"])[:num_clean]
                    final_entries = sorted(adv_entries + clean_entries, key=lambda x: -x["score"])
                    topk_contents = [e["context"] for e in final_entries]
                    topk_scores = [e["score"] for e in final_entries]
                    adv_indices_in_final = [idx for idx, e in enumerate(final_entries) if e["context"] in adv_text_map]
                    cnt_from_adv = len(adv_indices_in_final)
                else:
                    topk_contents = [r["context"] for r in topk_results]
                    topk_scores = [r["score"] for r in topk_results]
                    cnt_from_adv = 0
                    adv_indices_in_final = []

                # Query RAG with aggregated contexts
                start = time.time()
                response, selected_contexts, d = llm.rag_with_diverse_selection_new(
                    topk_contents, topk_scores, question, cnt_from_adv, n=args.diverse_n, num=i
                )
                elapsed = time.time() - start
                iter_time_list.append(round(elapsed, 2))
                iter_d_list.append(d)
                d_list.append(d)

                # Evaluate poison retention and success
                adv_text_set = set(adv_text_groups[offset]) if adv_text_groups else set()
                injected_adv = [ctx for ctx in topk_contents if ctx in adv_text_set]
                final_injected_adv = [ctx for ctx in selected_contexts if ctx in adv_text_set]
                has_poison = len(injected_adv) > 0
                retained_poison = len(final_injected_adv) > 0
                if has_poison:
                    total_poisoned_samples += 1
                    if retained_poison:
                        poison_retained_cnt += 1
                        iter_retain += 1

                # Response check
                poison_check = chat_completion(response_check_prompt(question, response, incorrect_ans))
                clean_check = chat_completion(response_check_prompt(question, response, correct_ans))
                if "yes" in clean_str_to_words(poison_check) and retained_poison:
                    iter_asr += 1
                if "yes" in clean_str_to_words(clean_check):
                    iter_tru += 1

                iter_result = {
                    "id": qid,
                    "question": question,
                    "injected_adv": injected_adv,
                    "input_prompt": "[AutoPrompt-Diverse-Selected]",
                    "output_poison": response,
                    "incorrect_answer": incorrect_ans,
                    "correct_answer": correct_ans
                }

            # Accumulate per-query results
            iter_results = locals().get("iter_results", [])
            iter_results.append(iter_result)

        # Aggregate iteration metrics
        asr_list.append(iter_asr)
        tru_list.append(iter_tru)
        retain_poison_list.append(iter_retain)
        ret_list.append([])  # Placeholder if needed
        all_time.append(iter_time_list if "iter_time_list" in locals() else [])

        all_results = [{f"iter_{iter_idx}": iter_results}]
        save_results(all_results, args.query_results_dir, args.name)

    # Final statistics
    d_mean = round(float(np.mean(d_list)), 2)
    asr = np.array(asr_list) / args.M
    tru = np.array(tru_list) / args.M
    asr_mean = round(np.mean(asr), 2)
    tru_mean = round(np.mean(tru), 2)
    time_mean = round(float(np.mean(all_time)), 2)
    retain_poison = np.array(retain_poison_list) / args.M

    print("============== Final Poison Retention Statistics ==============")
    print(f"Total poisoned samples: {total_poisoned_samples}")
    print(f"Retained poisoned samples: {poison_retained_cnt}")
    print(f"Poison retention ratio: {retain_poison}")
    print(f"Mean d: {d_mean}")
    print(f"Mean time: {time_mean}")
    if total_poisoned_samples > 0:
        print(f"PRR: {poison_retained_cnt / total_poisoned_samples:.2f}")
    else:
        print("No poison injected. PRR undefined.")
