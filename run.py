import os
import subprocess

def get_log_name(params):
    os.makedirs(f"logs/{params['query_results_dir']}_logs", exist_ok=True)

    if params.get('note'):
        base_name = params['note']
    else:
        if params.get('use_truth'):
            base_name = f"{params['eval_dataset']}-{params['eval_model_code']}-{params['model_name']}-Truth-M{params['M']}x{params['repeat_times']}"
        else:
            base_name = f"{params['eval_dataset']}-{params['eval_model_code']}-{params['model_name']}-Top{params['top_k']}-M{params['M']}x{params['repeat_times']}"

        if params.get('attack_method') is not None:
            base_name += f"-adv-{params['attack_method']}-{params['score_function']}-{params['adv_per_query']}-{params['top_k']}"

    log_file = f"logs/{params['query_results_dir']}_logs/{base_name}_{params['poison_ratio']}_{params['diverse_n']}.txt"
    return log_file, base_name

def run(params):
    log_file, log_name = get_log_name(params)

    cmd_parts = [
        "nohup", "python3", "-u", "main.py",
        "--eval_model_code", str(params["eval_model_code"]),
        "--eval_dataset", str(params["eval_dataset"]),
        "--split", str(params["split"]),
        "--query_results_dir", str(params["query_results_dir"]),
        "--model_name", str(params["model_name"]),
        "--response_check_model_name", str(params["response_check_model_name"]),
        "--top_k", str(params["top_k"]),
        "--diverse_n", str(params["diverse_n"]),
        "--poison_ratio", str(params["poison_ratio"]),
        "--use_truth", str(params["use_truth"]),
        "--gpu_id", str(params["gpu_id"]),
        "--attack_method", str(params["attack_method"]),
        "--adv_per_query", str(params["adv_per_query"]),
        "--score_function", str(params["score_function"]),
        "--repeat_times", str(params["repeat_times"]),
        "--M", str(params["M"]),
        "--seed", str(params["seed"]),
        "--name", log_name
    ]

    with open(log_file, "a") as lf:
        # Launch in background similarly to nohup
        subprocess.Popen(cmd_parts, stdout=lf, stderr=lf, close_fds=True)

if __name__ == "__main__":
    test_params = {
        "eval_model_code": "ance",
        "eval_dataset": None,  # will be set in loop
        "split": "test",
        "query_results_dir": "result",
        "model_name": "mistral-7b-v0.3",
        "response_check_model_name": "gpt4",
        "use_truth": False,
        "top_k": 8,
        "diverse_n": 3,
        "gpu_id": 0,
        "attack_method": "LM_targeted",
        "adv_per_query": 5,
        "poison_ratio": 0.2,
        "score_function": "dot",
        "repeat_times": 10,
        "M": 10,
        "seed": 12,
        "note": None,
    }

    for dataset in ["nq", "hotpotqa", "msmarco"]:
        test_params["eval_dataset"] = dataset
        run(test_params)
