import os
import itertools
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .Model import Model
from src.prompts import wrap_prompt
from src.utils import load_models

MULTIPLE_PROMPT = (
    "You are a helpful assistant, below is a query from a user and some relevant contexts. "
    "Answer the question given the information in those contexts. Your answer should be short and concise. "
    "If you cannot find the answer to the question, just say \"I don't know\". "
    "\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:"
)

def fuse_context_embeddings(context_embeddings, strategy="inverse_rank", scores=None):
    if not context_embeddings:
        raise ValueError("context_embeddings cannot be empty")

    device = context_embeddings[0].device
    dtype = context_embeddings[0].dtype
    stacked = torch.stack(context_embeddings)  # (N, L, D)
    N = stacked.shape[0]

    if strategy == "inverse_rank":
        weights = 1.0 / (torch.arange(1, N + 1, device=device, dtype=dtype))
    elif strategy == "linear_decay":
        weights = torch.arange(N, 0, -1, device=device, dtype=dtype)
    elif strategy == "softmax_score":
        if scores is None or len(scores) != N:
            raise ValueError("scores length mismatch")
        weights = torch.tensor(scores, device=device, dtype=dtype)
        weights = torch.softmax(weights, dim=0)
    elif strategy == "custom":
        if scores is None or len(scores) != N:
            raise ValueError("scores length mismatch")
        weights = torch.tensor(scores, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy not in {"softmax_score", "custom"}:
        weights = weights / weights.sum()

    weights = weights.view(N, 1, 1)
    avg_context = (stacked * weights).sum(dim=0)  # (L, D)
    return avg_context


class Mistral(Model):
    def __init__(self, config):
        super().__init__(config)
        params = config["params"]
        self.max_output_tokens = int(params["max_output_tokens"])
        self.device = params["device"]
        self.temperature = params.get("temperature", 0.7)
        self.name = config["model_info"]["name"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.model.eval()

    def query(self, msg):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True
        )
        out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return out[len(msg):]

    def interpolate_tensor(self, tensor, target_len):
        L, D = tensor.shape
        if L == target_len:
            return tensor
        tensor = tensor.unsqueeze(0).permute(0, 2, 1)  # (1, D, L)
        tensor = F.interpolate(tensor, size=target_len, mode='linear', align_corners=False)
        return tensor.permute(0, 2, 1).squeeze(0)  # (target_len, D)

    def query_with_prompt_embed_avg_context(self, topk_contents, topk_scores, question):
        prefix, rest = MULTIPLE_PROMPT.split("[context]")
        suffix_before_q, suffix_after_q = rest.split("[question]")

        # Embed each context
        context_embeddings_raw = []
        max_len = 0
        for ctx in topk_contents:
            input_ids = self.tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            if input_ids.shape[-1] == 0:
                continue
            with torch.no_grad():
                embed = self.model.model.embed_tokens(input_ids).squeeze(0)
            context_embeddings_raw.append(embed)
            max_len = max(max_len, embed.shape[0])

        if not context_embeddings_raw:
            raise ValueError("All context inputs are empty.")

        aligned = [self.interpolate_tensor(e, max_len) for e in context_embeddings_raw]
        avg_context = fuse_context_embeddings(aligned, strategy="softmax_score", scores=topk_scores)

        def get_token_embed(text):
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            with torch.no_grad():
                return self.model.model.embed_tokens(ids).squeeze(0)

        prefix_embed = get_token_embed(prefix)
        before_q_embed = get_token_embed(suffix_before_q)
        question_embed = get_token_embed(question)
        after_q_embed = get_token_embed(suffix_after_q)

        content_embed = torch.cat([
            prefix_embed, avg_context, before_q_embed, question_embed, after_q_embed
        ], dim=0)

        bos_token_id = self.tokenizer.bos_token_id
        bos_embed = self.model.model.embed_tokens(torch.tensor([[bos_token_id]], device=self.device))
        full_embed = torch.cat([bos_embed.squeeze(0), content_embed], dim=0).unsqueeze(0)
        attention_mask = torch.ones(full_embed.shape[:2], device=self.device, dtype=torch.long)

        outputs = self.model.generate(
            inputs_embeds=full_embed,
            attention_mask=attention_mask,
            temperature=self.temperature,
            max_new_tokens=self.max_output_tokens,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def rag_with_diverse_selection_new(self, topk_contents, topk_scores, question, cnt_from_adv, n, num):
        device = self.device
        valid_indices = [i for i, ctx in enumerate(topk_contents) if ctx.strip()]
        if len(valid_indices) < n:
            raise ValueError(f"Not enough valid contexts ({len(valid_indices)} < {n}).")

        idx_combinations = list(itertools.combinations(range(len(valid_indices)), n))
        num_poison_combs = len(idx_combinations) - len(list(itertools.combinations(range(len(valid_indices) - cnt_from_adv), n)))

        # Generate embedding vectors for each combination using external retriever
        model, c_model, tokenizer, get_emb = load_models("contriever")
        model.eval().to(device)
        c_model.eval().to(device)

        embed_vectors = []
        all_comb_texts = []
        for idxs in idx_combinations:
            selected_texts = [topk_contents[valid_indices[i]] for i in idxs]
            joined = " ".join(selected_texts)
            all_comb_texts.append(selected_texts)

            inputs = tokenizer(joined, return_tensors="pt", add_special_tokens=False, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = get_emb(model, inputs).squeeze(0)
            embed_vectors.append(emb)

        compute_groupwise_cosine_sim_means(embed_vectors, idx_combinations, valid_indices, cnt_from_adv)
        visualize_embedding_combinations_tsne(embed_vectors, idx_combinations, valid_indices, cnt_from_adv, num)

        # Score combinations by median angle
        s_scores = []
        angle_dict = {}
        for i, e_i in enumerate(embed_vectors):
            angles = []
            angle_dict[i] = {}
            for j, e_j in enumerate(embed_vectors):
                if i == j:
                    continue
                cos_sim = F.cosine_similarity(e_i.unsqueeze(0), e_j.unsqueeze(0), dim=-1).clamp(-1, 1)
                angle = torch.acos(cos_sim).item()
                angles.append(angle)
                angle_dict[i][j] = angle
            angles.sort()
            angle_dict[i] = dict(sorted(angle_dict[i].items(), key=lambda x: x[1]))
            if angles:
                s_scores.append(angles[len(angles) // 2])

        if s_scores:
            best_idx = int(torch.tensor(s_scores).argmin().item())
        else:
            best_idx = 0
        best_comb = idx_combinations[best_idx]
        best_texts = all_comb_texts[best_idx]

        # Compute d for decision
        sorted_angles = list(angle_dict[best_idx].items())
        idx_d = len(sorted_angles) // 2 + num_poison_combs
        _, angle_kth = sorted_angles[min(idx_d, len(sorted_angles) - 1)]
        d = 3 * angle_kth

        best_score_indices = [valid_indices[i] for i in best_comb]
        best_comb_scores = [topk_scores[idx] for idx in best_score_indices]

        # Use fused prompt embedding path
        response = self.query_with_prompt_embed_avg_context(best_texts, best_comb_scores, question)
        return response, best_texts, d
