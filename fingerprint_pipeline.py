# run_fingerprint_pipeline.py
"""
批量跑 fingerprint + 在固定 suspect 模型上测试。
核心逻辑全部来自原来的 test_batch.py（这里假设你把它改名为 fingerprint_tools.py）。
"""

from __future__ import annotations
import argparse

from test_batch import (
    # 公共工具
    set_seed,
    load_hf_model,
    unload_hf_model,
    SuspectFromLoadedHF,
    generate_fingerprints_batch,
    evaluate_fingerprint_unclean,   # ✅ 用 unclean 版本
    evaluate_and_append_simple,
    load_model_ids,
    _safe_model_label,
    MODEL_LIST_CSV,
    DEFAULT_BASE_MODEL,
    RELATION,
)


def run_single_candidate_unclean(
    model_name: str,
    suspect_wrapper: SuspectFromLoadedHF,
    suspect_model_name: str,
    *,
    num_pairs: int = 5,
    prompt_style: str = "raw",
    save_report_dir: str = ".",
    csv_path: str = "lineage_scores.csv",
    k_prefix: int = 30,
    relation: str | None = None,
) -> None:
    """
    和你原来的 run_single_candidate 逻辑一样，只是：
    - 调用 evaluate_fingerprint_unclean（不再用 clean 版本）
    - 其它逻辑尽量保持不变
    """
    relation = relation or RELATION
    model_origin = _safe_model_label(model_name)
    print(f"\n===== Testing candidate: {model_name} (label: {model_origin}) =====")

    # 1. 加载 candidate 作为 base（用来生成 fingerprint）
    try:
        model1, tok1, dev1 = load_hf_model(model_name, fourbit=False)
    except Exception as exc:
        print(f"[error] Failed to load candidate model {model_name}: {exc}")
        return

    try:
        print("Loaded candidate on", dev1)
        set_seed(42)

        # 2. 用 candidate 生成 (x', y) fingerprint
        pairs, pairs_path = generate_fingerprints_batch(
            model=model1,
            model_name=model_origin,
            tokenizer=tok1,
            num_pairs=num_pairs,
            prompt_style=prompt_style,
            l_random_prefix=8,
            total_len=64,
            k_bottom=50,
            max_new_tokens=64,
            save_json_path=None,  # 保持你原来内部逻辑（函数里会自己决定存哪里的话就留 None）
        )

        if not pairs_path:
            print("[warn] No fingerprint file produced, skipping evaluation.")
            return

        # 3. 在 suspect 模型上跑 eval（✅ 用 unclean 版本）
        suspect_label = f"{suspect_model_name}-on-{model_origin}"
        report_path = f"{save_report_dir}/eval_report_{model_origin}.json"

        _, final_report_path = evaluate_fingerprint_unclean(
            pairs_json_path=pairs_path,
            model_name=model_origin,
            suspect_base_name=suspect_model_name,
            suspect_model=suspect_wrapper,
            suspect_label=suspect_label,
            save_report_path=report_path,
            sig_min_tok_len=6,
            use_timestamp_in_name=False,
        )

        if not final_report_path:
            print("[warn] No evaluation report path returned, skipping lineage score logging.")
            return

        # 4. 计算 lineage 分数并 append 到 CSV
        res = evaluate_and_append_simple(
            final_report_path,
            csv_path=csv_path,
            k_prefix=k_prefix,
            base_model_name=model_name,
            suspect_model_name=suspect_model_name,
            relation=relation,
        )
        print(res["summary"])
    finally:
        unload_hf_model(model1, tok1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_list_csv",
        type=str,
        default=MODEL_LIST_CSV,
        help="包含 candidate/base 模型列表的 CSV（有 model_id 列）",
    )
    parser.add_argument(
        "--suspect_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="固定的 suspect/base 模型名称（HF repo id）",
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=20,
        help="每个 candidate 生成多少 fingerprint pair",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="raw",
        choices=["raw", "oneshot", "chatml"],
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="lineage_scores.csv",
    )
    parser.add_argument(
        "--save_report_dir",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--k_prefix",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--relation",
        type=str,
        default=RELATION,
        help="关系标签，比如 same/diff/pos/neg",
    )

    args = parser.parse_args()

    candidate_models = load_model_ids(args.model_list_csv)
    if not candidate_models:
        print("[info] No candidate models found to evaluate.")
        return

    model_base_name = args.suspect_model
    print(f"[setup] Loading suspect/base model: {model_base_name}")
    model2, tok2, dev2 = load_hf_model(model_base_name, fourbit=False)
    print("Loaded suspect/base on", dev2)
    suspect = SuspectFromLoadedHF(model2, tok2)

    try:
        for idx, candidate in enumerate(candidate_models, 1):
            print(f"\n### ({idx}/{len(candidate_models)}) Processing {candidate}")
            try:
                run_single_candidate_unclean(
                    candidate,
                    suspect,
                    model_base_name,
                    num_pairs=args.num_pairs,
                    prompt_style=args.prompt_style,
                    save_report_dir=args.save_report_dir,
                    csv_path=args.csv_path,
                    k_prefix=args.k_prefix,
                    relation=args.relation,
                )
            except Exception as exc:
                print(f"[error] Unexpected failure while processing {candidate}: {exc}")
    finally:
        unload_hf_model(model2, tok2)


if __name__ == "__main__":
    main()