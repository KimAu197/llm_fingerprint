#!/usr/bin/env python3
"""Transitive closure of Hugging Face model card `base_model` starting from overlap CSV row labels."""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from paths import result_root


def load_seed_models_from_overlap_csv(path: Path) -> list[str]:
    """Row labels (first column), same 103 models as symmetric matrix index."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        out: list[str] = []
        for row in reader:
            if row and row[0].strip():
                out.append(row[0].strip())
        return out


def normalize_base_model(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        acc: list[str] = []
        for item in raw:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    acc.append(s)
            elif isinstance(item, dict) and "name" in item:
                s = str(item["name"]).strip()
                if s:
                    acc.append(s)
        return acc
    return []


def fetch_card_bases(api: HfApi, model_id: str, retries: int = 4) -> tuple[list[str], str | None]:
    last_err = None
    for attempt in range(retries):
        try:
            info = api.model_info(model_id, expand=["cardData"])
            cd = info.card_data
            if cd is None:
                return [], None
            return normalize_base_model(cd.base_model), None
        except RepositoryNotFoundError as e:
            return [], f"not_found: {e}"
        except HfHubHTTPError as e:
            last_err = str(e)
            code = e.response.status_code if e.response is not None else None
            if code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            return [], f"http_error: {e}"
        except Exception as e:
            last_err = str(e)
            time.sleep(0.5 * (attempt + 1))
    return [], last_err or "unknown_error"


def is_repo_id(s: str) -> bool:
    return "/" in s and not s.startswith("http") and " " not in s


def transitive_closure_card_bases(
    seeds: list[str],
    api: HfApi,
    *,
    sleep_s: float,
    log_path: Path | None,
) -> tuple[list[str], dict[str, str]]:
    """Return sorted model ids and fetch errors for ids we attempted."""
    closed: set[str] = set(seeds)
    errors: dict[str, str] = {}
    q: deque[str] = deque(seeds)
    seen_fetch: set[str] = set()

    log_lines: list[str] = []
    while q:
        mid = q.popleft()
        if mid in seen_fetch:
            continue
        seen_fetch.add(mid)
        bases, err = fetch_card_bases(api, mid)
        if err:
            errors[mid] = err
        for b in bases:
            if not is_repo_id(b):
                continue
            if b not in closed:
                closed.add(b)
                q.append(b)
        if sleep_s > 0:
            time.sleep(sleep_s)
        if log_path:
            log_lines.append(f"{mid}\t{repr(bases)}\t{err or ''}\n")

    if log_path and log_lines:
        log_path.write_text("model_id\tbases\terr\n" + "".join(log_lines), encoding="utf-8")

    return sorted(closed), errors


def main() -> None:
    root = result_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=root / "overlap_matrix_300_clean.csv")
    ap.add_argument("--out", type=Path, default=root / "closure" / "model_ids_hf_card_closure.csv")
    ap.add_argument("--log", type=Path, default=None, help="Optional TSV of each HF fetch")
    ap.add_argument("--sleep", type=float, default=0.06)
    args = ap.parse_args()

    seeds = load_seed_models_from_overlap_csv(args.csv)
    api = HfApi()
    log_path = args.log
    closed, errs = transitive_closure_card_bases(seeds, api, sleep_s=args.sleep, log_path=log_path)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_id"])
        for mid in closed:
            w.writerow([mid])

    print(f"seeds: {len(seeds)}")
    print(f"closure size: {len(closed)}")
    print(f"wrote: {args.out}")
    if errs:
        print(f"fetch issues: {len(errs)} (see log)")
        for k in list(errs)[:8]:
            print(f"  {k}: {errs[k]}")


if __name__ == "__main__":
    main()
