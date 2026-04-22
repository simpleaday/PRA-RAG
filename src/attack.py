import random
import torch
import torch.nn.functional as F
import os
import json
import re

from sentence_transformers import SentenceTransformer
from src.utils import load_json


class GradientStorage:
    def __init__(self, module):
        self._stored_gradient = None
        module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def get_embeddings(model):
    if isinstance(model, SentenceTransformer):
        return model[0].auto_model.embeddings.word_embeddings
    return model.embeddings.word_embeddings


def hotflip_attack(grad, embedding_matrix, increase_loss=False, num_candidates=1, filter=None):
    with torch.no_grad():
        dot = torch.matmul(embedding_matrix, grad)
        if filter is not None:
            dot = dot - filter
        if not increase_loss:
            dot = -dot
        _, top_k_ids = dot.topk(num_candidates)
    return top_k_ids


def clean_str_to_words(s):
    s = str(s).strip()
    if len(s) > 1 and s.endswith("."):
        s = s[:-1]
    s = s.lower()
    return re.findall(r"\b\w+\b", s)


class Attacker:
    def __init__(self, args, **kwargs):
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query

        self.model = kwargs.get("model")
        self.c_model = kwargs.get("c_model")
        self.tokenizer = kwargs.get("tokenizer")
        self.get_emb = kwargs.get("get_emb")

        self.max_seq_length = getattr(args, "max_seq_length", 128)
        self.pad_to_max_length = getattr(args, "pad_to_max_length", True)
        self.num_iter = getattr(args, "num_iter", 30)
        self.gold_init = getattr(args, "gold_init", True)
        self.early_stop = getattr(args, "early_stop", False)
        self.num_cand = getattr(args, "num_cand", 100)

        self.all_adv_texts = load_json(f"results/adv_targeted_results/{args.eval_dataset}.json")

    def get_attack(self, target_queries):
        adv_groups = []
        if self.attack_method == "LM_targeted":
            for tq in target_queries:
                question = tq["query"]
                qid = tq["id"]
                adv_texts_b = self.all_adv_texts[qid]["adv_texts"][: self.adv_per_query]
                base = question + "."
                adv_groups.append([base + t for t in adv_texts_b])
        elif self.attack_method == "hotflip":
            adv_groups = self.hotflip(target_queries)
        else:
            raise NotImplementedError
        return adv_groups

    def hotflip(self, target_queries):
        device = "cuda"
        adv_text_groups = []
        for tq in target_queries:
            query = tq["query"]
            top1_score = tq.get("top1_score", 0)
            qid = tq["id"]
            adv_texts_b = self.all_adv_texts[qid]["adv_texts"]
            group = []

            for j in range(self.adv_per_query):
                adv_b = adv_texts_b[j]
                adv_b_ids = self.tokenizer(adv_b, max_length=self.max_seq_length, truncation=True, padding=False)["input_ids"]

                if self.gold_init:
                    adv_a_ids = self.tokenizer(query, max_length=self.max_seq_length, truncation=True, padding=False)["input_ids"]
                else:
                    adv_a_ids = [self.tokenizer.mask_token_id] * self.args.num_adv_passage_tokens

                embeddings = get_embeddings(self.c_model)
                grad_storage = GradientStorage(embeddings)

                adv_passage_ids = torch.tensor(adv_a_ids + adv_b_ids, device=device).unsqueeze(0)
                attention_mask = torch.ones_like(adv_passage_ids)
                token_type_ids = torch.zeros_like(adv_passage_ids)

                q_inputs = self.tokenizer(
                    query,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length" if self.pad_to_max_length else False,
                    return_tensors="pt",
                )
                q_inputs = {k: v.cuda() for k, v in q_inputs.items()}
                q_emb = self.get_emb(self.model, q_inputs).detach()

                for _ in range(self.num_iter):
                    self.c_model.zero_grad()
                    p_sent = {
                        "input_ids": adv_passage_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                    }
                    p_emb = self.get_emb(self.c_model, p_sent)

                    if self.args.score_function == "dot":
                        sim = torch.mm(p_emb, q_emb.T)
                    elif self.args.score_function == "cos_sim":
                        sim = torch.cosine_similarity(p_emb, q_emb)
                    else:
                        raise KeyError

                    loss = sim.mean()
                    if self.early_stop and sim.item() > top1_score + 0.1:
                        break
                    loss.backward()

                    grad = grad_storage.get().sum(dim=0)
                    token_to_flip = random.randrange(len(adv_a_ids))
                    candidates = hotflip_attack(
                        grad[token_to_flip],
                        embeddings.weight,
                        increase_loss=True,
                        num_candidates=self.num_cand,
                    )

                    current_score = loss.item()
                    candidate_scores = torch.zeros(len(candidates), device=device)

                    for idx, candidate in enumerate(candidates):
                        temp_adv = adv_passage_ids.clone()
                        temp_adv[:, token_to_flip] = candidate
                        temp_p_sent = {
                            "input_ids": temp_adv,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                        }
                        temp_p_emb = self.get_emb(self.c_model, temp_p_sent)
                        with torch.no_grad():
                            if self.args.score_function == "dot":
                                temp_sim = torch.mm(temp_p_emb, q_emb.T)
                            elif self.args.score_function == "cos_sim":
                                temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                            else:
                                raise KeyError
                            candidate_scores[idx] = temp_sim.mean().item()

                    if (candidate_scores > current_score).any():
                        best_idx = candidate_scores.argmax()
                        adv_passage_ids[:, token_to_flip] = candidates[best_idx]

                adv_text = self.tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                group.append(adv_text)
            adv_text_groups.append(group)
        return adv_text_groups
