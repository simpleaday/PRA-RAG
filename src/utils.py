import os
import json
import random
import re

import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from .contriever_src.contriever import Contriever


# Mapping from model code to model identifier
MODEL_NAME_MAP = {
    "contriever": "facebook/contriever",
    "contriever-msmarco": "facebook/contriever-msmarco",
    "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"
}


def contriever_get_emb(model, inputs):
    return model(**inputs)


def ance_get_emb(model, inputs):
    inputs.pop("token_type_ids", None)
    return model(inputs)["sentence_embedding"]


def load_models(model_code):
    assert model_code in MODEL_NAME_MAP, f"Unsupported model code: {model_code}"
    name = MODEL_NAME_MAP[model_code]

    if "contriever" in model_code:
        model = Contriever.from_pretrained(name, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(name)
        get_emb = contriever_get_emb
        c_model = model
    elif "ance" in model_code:
        model = SentenceTransformer(name)
        tokenizer = model.tokenizer
        get_emb = ance_get_emb
        c_model = model
    else:
        raise NotImplementedError(f"Model code not supported: {model_code}")

    return model, c_model, tokenizer, get_emb


def load_beir_datasets(dataset_name, split):
    assert dataset_name in {"nq", "msmarco", "hotpotqa"}
    if dataset_name == "msmarco":
        split = "train"  # override as per convention

    out_dir = os.path.join(os.getcwd(), "datasets")
    data_path = os.path.join(out_dir, dataset_name)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, out_dir)

    loader = GenericDataLoader(data_path)
    corpus, queries, qrels = loader.load(split=split)
    return corpus, queries, qrels


# JSON serialization helpers
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=NpEncoder, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Utilities
def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_str(s):
    s = str(s).strip()
    if len(s) > 1 and s.endswith("."):
        s = s[:-1]
    return s.lower()


def clean_str_to_words(s):
    s = clean_str(s)
    return re.findall(r"\b\w+\b", s)

