import argparse
import itertools
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import tqdm
import yaml
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from smoothquant.fake_quant import quantize_model
from smoothquant.mod import (
    DATATYPE_MAPPING_3_BIT,
    DATATYPE_MAPPING_3_BIT_MX,
    DATATYPE_MAPPING_4_BIT,
    DATATYPE_MAPPING_4_BIT_MX,
    DATATYPE_MAPPING_6_BIT,
    DATATYPE_MAPPING_6_BIT_MX,
    DATATYPE_MAPPING_8_BIT,
    DATATYPE_MAPPING_8_BIT_MX,
)
from smoothquant.smooth import smooth_lm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DATATYPE_BY_BIT = {
    3: sorted(DATATYPE_MAPPING_3_BIT.keys()),
    4: sorted(DATATYPE_MAPPING_4_BIT.keys()),
    6: sorted(DATATYPE_MAPPING_6_BIT.keys()),
    8: sorted(DATATYPE_MAPPING_8_BIT.keys()),
}
MX_DATATYPE_BY_BIT = {
    3: sorted(DATATYPE_MAPPING_3_BIT_MX.keys()),
    4: sorted(DATATYPE_MAPPING_4_BIT_MX.keys()),
    6: sorted(DATATYPE_MAPPING_6_BIT_MX.keys()),
    8: sorted(DATATYPE_MAPPING_8_BIT_MX.keys()),
}
SUPPORTED_WQ_BITS = sorted(BASE_DATATYPE_BY_BIT.keys())
BITS_WITH_MX_SUPPORT = {bit for bit, names in MX_DATATYPE_BY_BIT.items() if names}
SUPPORTED_DATATYPE_RANGE = sorted(
    {dtype for dtypes in BASE_DATATYPE_BY_BIT.values() for dtype in dtypes}
)
DATATYPE_TO_BITS = {
    dtype: bit for bit, dtypes in BASE_DATATYPE_BY_BIT.items() for dtype in dtypes
}

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


def parse_int_candidates(candidate_str: Optional[str]) -> List[int]:
    if not candidate_str:
        return []
    candidates: List[int] = []
    seen = set()
    for part in candidate_str.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in seen:
            candidates.append(value)
            seen.add(value)
    return candidates


def parse_bool_candidates(candidate_str: Optional[str]) -> List[bool]:
    if not candidate_str:
        return []
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
    }
    values: List[bool] = []
    seen = set()
    for part in candidate_str.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Invalid boolean candidate: {part}")
        value = mapping[key]
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def infer_wq_bits_list_length(model_config) -> int:#NOTICE: if adjust to more models, this function may need to be modified 
    if model_config.model_type == "mixtral":
        return 4
    return 3


def load_hardware_configs(path: Optional[str], default_mx: bool) -> List[Dict]:
    def build_entry(name: str, datatypes: List[str], mx_support: bool) -> Optional[Dict]:
        unique = []
        seen = set()
        for dtype in datatypes:
            dtype_str = str(dtype)
            if dtype_str not in seen:
                unique.append(dtype_str)
                seen.add(dtype_str)

        invalid = sorted(set(unique) - set(SUPPORTED_DATATYPE_RANGE))
        if invalid:
            raise ValueError(f"{name}: unsupported datatype(s) {invalid}.")

        bit_to_dtypes: Dict[int, List[str]] = {}
        for dtype in unique:
            bit = DATATYPE_TO_BITS.get(dtype)
            if bit is None:
                continue
            bit_to_dtypes.setdefault(bit, []).append(dtype)

        supported_bits = sorted(bit_to_dtypes.keys())
        if not supported_bits:
            print(
                f"Warning: hardware '{name}' has no supported wq_bits inferred from datatype_support; skipping."
            )
            return None

        for bit in supported_bits:
            bit_to_dtypes[bit] = sorted(bit_to_dtypes[bit])

        return {
            "name": name,
            "datatype_support": sorted(unique),
            "mx_support": bool(mx_support),
            "supported_bits": supported_bits,
            "bit_to_dtypes": bit_to_dtypes,
        }

    if not path:
        entry = build_entry("default", SUPPORTED_DATATYPE_RANGE, bool(default_mx))
        return [entry] if entry else []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError("Hardware YAML must be a list of hardware entries.")

    hardware_configs: List[Dict] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Hardware entry #{idx} must be a mapping.")
        name = str(entry.get("name") or f"hardware_{idx}")
        datatypes = entry.get("datatype_support")
        if not isinstance(datatypes, list) or not datatypes:
            raise ValueError(f"{name}: 'datatype_support' must be a non-empty list.")
        mx_support = bool(entry.get("mx_support", False))
        hardware_entry = build_entry(name, datatypes, mx_support)
        if hardware_entry:
            hardware_configs.append(hardware_entry)

    if not hardware_configs:
        raise ValueError("No valid hardware configurations found in YAML.")
    return hardware_configs


def build_datatype_support_used(
    bits_list: List[int],
    hardware: Dict,
    sample: bool = False,
    rng: Optional[random.Random] = None,
) -> List[str]:
    support = set()
    bit_to_dtypes = hardware.get("bit_to_dtypes", {})
    for bit in bits_list:
        candidates = bit_to_dtypes.get(bit, [])
        if not candidates:
            continue
        if sample and rng is not None and len(candidates) > 1:
            sample_size = rng.randint(1, len(candidates))
            chosen = rng.sample(candidates, sample_size)
        else:
            chosen = candidates
        support.update(chosen)
    return sorted(support)


def generate_configs_for_hardware(
    args,
    bits_len: int,
    rng: random.Random,
    hardware: Dict,
) -> List[Dict]:

    hardware_bits = hardware["supported_bits"]
    candidate_bits =  list(hardware_bits)
    if not candidate_bits:
        print(
            f"Warning: hardware '{hardware['name']}' has no candidate bits after applying CLI filters."
        )
        return []

    def bits_have_datatypes(bits_list: List[int]) -> bool:
        return all(hardware["bit_to_dtypes"].get(bit) for bit in bits_list)

    if args.search_mode == "fixed":
        if args.wq_bits not in candidate_bits:
            print(
                f"Warning: hardware '{hardware['name']}' does not support fixed wq_bits={args.wq_bits}; skipping."
            )
            return []
        bits_list = [args.wq_bits] * bits_len
        if not bits_have_datatypes(bits_list):
            print(
                f"Warning: hardware '{hardware['name']}' lacks datatype coverage for bits {bits_list}; skipping."
            )
            return []
        datatype_support_used = build_datatype_support_used(bits_list, hardware)
        if not datatype_support_used:
            print(
                f"Warning: hardware '{hardware['name']}' produced empty datatype support for bits {bits_list}."
            )
            return []
        return [
            {
                "hardware": hardware,
                "wq_bits_list": bits_list,
                "datatype_support_used": datatype_support_used,
                "if_mx_support": hardware["mx_support"],
            }
        ]

    configs: List[Dict] = []

    if args.search_mode == "grid":
        for bits_tuple in itertools.product(candidate_bits, repeat=bits_len):
            bits_list = list(bits_tuple)
            if not bits_have_datatypes(bits_list):
                continue
            datatype_support_used = build_datatype_support_used(bits_list, hardware)
            if not datatype_support_used:
                continue
            configs.append(
                {
                    "hardware": hardware,
                    "wq_bits_list": bits_list,
                    "datatype_support_used": datatype_support_used,
                    "if_mx_support": hardware["mx_support"],
                }
            )
        return configs

    if args.search_mode == "random":
        target = max(args.num_trials, 1)
        seen = set()
        attempts = 0
        max_attempts = target * 20
        while len(configs) < target and attempts < max_attempts:
            attempts += 1
            bits_list = [rng.choice(candidate_bits) for _ in range(bits_len)]
            if not bits_have_datatypes(bits_list):
                continue
            datatype_support_used = build_datatype_support_used(
                bits_list, hardware, sample=False, rng=rng#TODO sample is unnecessary?
            )
            if not datatype_support_used:
                continue
            key = (tuple(bits_list), tuple(datatype_support_used))
            if key in seen:
                continue
            seen.add(key)
            configs.append(
                {
                    "hardware": hardware,
                    "wq_bits_list": bits_list,
                    "datatype_support_used": datatype_support_used,
                    "if_mx_support": hardware["mx_support"],
                }
            )
        if len(configs) < target:
            print(
                f"Warning: hardware '{hardware['name']}': requested {target} random configs but generated {len(configs)} unique ones."
            )
        return configs

    raise ValueError(f"Unsupported search mode: {args.search_mode}")


def describe_config(config: Dict) -> str:
    bits_str = ",".join(str(bit) for bit in config["wq_bits_list"])
    datatype_str = ",".join(config["datatype_support_used"])
    mx_str = "True" if config["if_mx_support"] else "False"
    hardware = config["hardware"]
    return (
        f"hardware={hardware['name']} "
        f"(bits={hardware['supported_bits']}, "
        f"datatype_support={','.join(hardware['datatype_support'])}, "
        f"mx_support={hardware['mx_support']}), "
        f"wq_bits_list=[{bits_str}], datatype_support_used=[{datatype_str}], if_mx_support={mx_str}"
    )


def init_results_db(db_path: str) -> sqlite3.Connection:
    path = Path(db_path).expanduser()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS quant_results (
            model_path TEXT NOT NULL,
            config_signature TEXT NOT NULL,
            hardware_signature TEXT NOT NULL,
            hardware_name TEXT NOT NULL,
            wquantization TEXT,
            datatype TEXT,
            group_size INTEGER,
            alpha REAL,
            smooth INTEGER,
            quantize INTEGER,
            act_scales_path TEXT,
            n_samples INTEGER,
            perplexity REAL,
            layer_datatypes TEXT,
            created_at TEXT,
            UNIQUE (
                model_path,
                config_signature,
                wquantization,
                datatype,
                group_size,
                alpha,
                smooth,
                quantize,
                act_scales_path,
                n_samples
            )
        )
        """
    )
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(quant_results)")
    }
    if "layer_datatypes" not in columns:
        conn.execute("ALTER TABLE quant_results ADD COLUMN layer_datatypes TEXT")
    conn.commit()
    return conn


def make_config_signature(config: Dict) -> str:
    payload = {
        "wq_bits_list": config["wq_bits_list"],
        "datatype_support_used": config["datatype_support_used"],
        "if_mx_support": config["if_mx_support"],
    }
    return json.dumps(payload, sort_keys=True)


def make_hardware_signature(hardware: Dict) -> str:
    payload = {
        "supported_bits": hardware["supported_bits"],
        "mx_support": bool(hardware["mx_support"]),
        "datatype_support": hardware.get("datatype_support", []),
        "bit_to_dtypes": hardware.get("bit_to_dtypes", {}),
    }
    return json.dumps(payload, sort_keys=True)


def lookup_cached_result(
    conn: sqlite3.Connection,
    model_path: str,
    config_signature: str,
    args,
    act_scales_path: Optional[str],
    n_samples: Optional[int],
    hardware_signature: str = None,
) -> Optional[Dict[str, Any]]:
    query = """
        SELECT perplexity, layer_datatypes
        FROM quant_results
        WHERE model_path = ?
            AND config_signature = ?
            {hardware_clause}
            AND wquantization = ?
            AND datatype = ?
            AND group_size = ?
            AND alpha = ?
            AND smooth = ?
            AND quantize = ?
            AND act_scales_path IS ?
            AND n_samples IS ?
    """
    if hardware_signature is None:
        hardware_clause = ""
        params = (
            model_path,
            config_signature,
            args.wquantization,
            args.datatype,
            args.group_size,
            args.alpha,
            int(args.smooth),
            int(args.quantize),
            act_scales_path,
            n_samples,
        )
    else:
        hardware_clause = "AND hardware_signature = ?"
        params = (
            model_path,
            config_signature,
            hardware_signature,
            args.wquantization,
            args.datatype,
            args.group_size,
            args.alpha,
            int(args.smooth),
            int(args.quantize),
            act_scales_path,
            n_samples,
        )

    cursor = conn.execute(query.format(hardware_clause=hardware_clause), params)
    row = cursor.fetchone()
    if not row:
        return None
    perplexity = float(row[0])
    layer_meta = json.loads(row[1]) if row[1] else None
    return {"perplexity": perplexity, "layer_datatypes": layer_meta}


def store_result(
    conn: sqlite3.Connection,
    model_path: str,
    config_signature: str,
    hardware_signature: str,
    hardware_name: str,
    args,
    act_scales_path: Optional[str],
    n_samples: Optional[int],
    perplexity: float,
    layer_datatypes: Optional[Dict[str, Any]],
) -> None:
    layer_datatypes_json = (
        json.dumps(layer_datatypes, sort_keys=True) if layer_datatypes else None
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO quant_results (
            model_path,
            config_signature,
            hardware_signature,
            hardware_name,
            wquantization,
            datatype,
            group_size,
            alpha,
            smooth,
            quantize,
            act_scales_path,
            n_samples,
            perplexity,
            layer_datatypes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_path,
            config_signature,
            hardware_signature,
            hardware_name,
            args.wquantization,
            args.datatype,
            args.group_size,
            args.alpha,
            int(args.smooth),
            int(args.quantize),
            act_scales_path,
            n_samples,
            perplexity,
            layer_datatypes_json,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument(
        "--act_scales_path",
        type=str,
        default="act_scales/llama-2-7b.pt",
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--results_path", type=str)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--wquantization", type=str, default="fmadse")
    parser.add_argument("--datatype", type=str, default="wrong")
    parser.add_argument("--wq_bits", type=int, default=4)
    parser.add_argument(
        "--search_mode", choices=["fixed", "grid", "random"], default="fixed"
    )

    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of random samples when search_mode is 'random'.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling in random search mode.",
    )

    parser.add_argument(
        "--hardware_yaml",
        type=str,
        default=None,
        help="Path to a YAML file describing hardware configurations.",
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default="results_mod/quant_results.db",
        help="SQLite database file for caching quantization results.",
    )
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

    model_config = AutoConfig.from_pretrained(args.model_path)
    wq_bits_list_len = infer_wq_bits_list_length(model_config)

    hardware_configs = load_hardware_configs(args.hardware_yaml, True)

    rng = random.Random(args.random_seed)

    quant_configs: List[Dict] = []
    for hardware in hardware_configs:
        configs = generate_configs_for_hardware(
            args,
            wq_bits_list_len,
            rng,
            hardware,
        )
        if not configs:
            continue
        quant_configs.extend(configs)

    if not quant_configs:
        raise ValueError("No quantization configurations were generated.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=args.n_samples)

    act_scales = torch.load(args.act_scales_path) if args.smooth else None
    act_scales_key = args.act_scales_path if args.smooth else None
    n_samples_key = args.n_samples if args.n_samples is not None else None

    db_conn = init_results_db(args.results_db)

    def prepare_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto"
        )

    if not args.quantize and len(quant_configs) > 1 or args.wquantization!='fmadse':
        print("Quantization disabled or wquantization is not fmadse; evaluating only the first configuration.")
        quant_configs = quant_configs[:1]

    results = []

    total = len(quant_configs)
    print(f'total {total} quant_configs to be run')
    for idx, config in enumerate(quant_configs, start=1):
        config_desc = describe_config(config)
        print(f"[{idx}/{total}] {config_desc}")

        config_signature = make_config_signature(config)
        hardware_signature = make_hardware_signature(config["hardware"])

        cached_record = lookup_cached_result(
            db_conn,
            args.model_path,
            config_signature,
            # hardware_signature,
            args,
            act_scales_key,
            n_samples_key,
        )
        if cached_record is not None:
            cached_value = cached_record["perplexity"]
            cached_layer_datatypes = cached_record.get("layer_datatypes")
            if lookup_cached_result(
                db_conn,
                args.model_path,
                config_signature,
                args,
                act_scales_key,
                n_samples_key,
                hardware_signature,
            ) is None:
                store_result(
                    db_conn,
                    args.model_path,
                    config_signature,
                    hardware_signature,
                    config["hardware"]["name"],
                    args,
                    act_scales_key,
                    n_samples_key,
                    cached_value,
                    cached_layer_datatypes,
                )
            print("  -> Using cached result.")
            print(f"  -> Perplexity: {cached_value}")
            results.append(
                {
                    "config": config,
                    "perplexity": cached_value,
                    "cached": True,
                    "layer_datatypes": cached_layer_datatypes,
                }
            )

            continue

        model = prepare_model()
        if act_scales is not None:
            smooth_lm(model, act_scales, args.alpha)
        if args.quantize:
            model = quantize_model(
                model,
                weight_quant=args.wquantization,
                act_quant="per_token",
                group_size=args.group_size,
                quantize_bmm_input=True,
                datatype=args.datatype,
                wq_bits_list=config["wq_bits_list"],
                datatype_support=config['hardware']["datatype_support"],
                if_mx_support=config["if_mx_support"],
            )
        layer_datatypes = getattr(model, "layer_quant_datatypes", None)

        ppl_value = evaluator.evaluate(model)
        print(f"  -> Perplexity: {ppl_value}")

        store_result(
            db_conn,
            args.model_path,
            config_signature,
            hardware_signature,
            config["hardware"]["name"],
            args,
            act_scales_key,
            n_samples_key,
            float(ppl_value),
            layer_datatypes,
        )

        results.append(
            {
                "config": config,
                "perplexity": float(ppl_value),
                "cached": False,
                "layer_datatypes": layer_datatypes,
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for entry in results:
        status = "cached" if entry["cached"] else "computed"
        config_desc = describe_config(entry["config"])
        print(f"{status.upper()}: {config_desc} -> Perplexity: {entry['perplexity']}")

    if args.results_path:
        path = Path(args.results_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in results:
                status = "cached" if entry["cached"] else "computed"
                config_desc = describe_config(entry["config"])
                f.write(f"{status.upper()}: {config_desc}\n")
                f.write(f"Perplexity: {entry['perplexity']}\n")

    db_conn.close()


if __name__ == "__main__":
    main()
