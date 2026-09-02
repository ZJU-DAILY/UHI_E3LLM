#!/usr/bin/env python3
"""Sample training records and generate DeepSeek explanation-distillation data."""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_from_disk
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import format_numeric_plain_prompt_PPO, format_prompt_PPO  # noqa: E402


SECTION_HEADINGS = (
    "Geographical Context:",
    "Structural Contextual Influence:",
    "Dynamic Environmental Modulation:",
    "Cross-Factor Synergy:",
    "Prediction Synthesis:",
    "Planning-Oriented Recommendations:",
)
EXPLANATION_MARKER = SECTION_HEADINGS[0]
EXPLANATION_GUIDELINES = """
Considering the important features related to UHI prediction, think step by step and explain the next month's UHI result.

Explanation Guidelines:
Structure your analysis into exactly 6 paragraphs with the following headings:
1. Geographical Context:
Analyze the regional geographic setting based on the provided coordinates. Explain how these baseline geographic conditions influence UHI.
2. Structural Contextual Influence:
Identify primary Structure Factors. Describe their impact on UHI.
3. Dynamic Environmental Modulation:
Identify critical Liquidity Factors. Analyze their monthly trends and interactions with structural features to modulate UHI.
4. Cross-Factor Synergy:
Highlight key interactions between structural and dynamic factors. Specify whether these interactions amplify or counteract UHI.
5. Prediction Synthesis:
Summarize the primary explanatory pathway linking geographic context, Structure Factors, and Liquidity Factors. Conclude with the dominant model-identified associations underlying the predicted UHI outcome (increase or decrease compared to the last few months).
6. Planning-Oriented Recommendations:
Based on the above model-identified factors influencing UHI Intensity, produce a locally targeted UHI mitigation plan. You should design your plan customized to the specific environment and reference related primary structural/liquidity factors, rather than listing general strategies like expanding green infrastructure. Provide one or two key recommendations and elaborate on them in detail.

Response Format Requirements:
Prediction: Start with "[Result:] Predicted UHI intensity for the next month: [value] °C".
Explanation: Use exactly 6 paragraphs matching the section headings above.
Language: Use semicolons to separate related observations within sentences.
""".strip()

RESULT_LINE_RE = re.compile(
    r"^\s*(?:\*\*)?\[Result:\]\s*Predicted UHI intensity for the next month:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*°C(?:\*\*)?\s*(?:\r?\n|$)",
    flags=re.IGNORECASE,
)
_thread_local = threading.local()


def _scalar(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return value


def _month_key(sample):
    return int(_scalar(sample["year_info"])), int(_scalar(sample["month_info"]))


def _parse_month(value):
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM, received {value!r}."
        ) from exc
    return parsed.year, parsed.month


def validate_city_samples(
    samples,
    city_id,
    expected_count,
    first_target,
    last_target,
):
    ordered = sorted(samples, key=_month_key)
    months = [_month_key(sample) for sample in ordered]
    if len(ordered) != expected_count:
        raise ValueError(
            f"City {city_id}: expected {expected_count} samples, got {len(ordered)}."
        )
    if len(set(months)) != len(months):
        raise ValueError(f"City {city_id}: duplicate target months found.")
    if months[0] != first_target or months[-1] != last_target:
        raise ValueError(
            f"City {city_id}: expected target range "
            f"{first_target[0]:04d}-{first_target[1]:02d} to "
            f"{last_target[0]:04d}-{last_target[1]:02d}, got "
            f"{months[0][0]:04d}-{months[0][1]:02d} to "
            f"{months[-1][0]:04d}-{months[-1][1]:02d}."
        )
    return ordered


def load_training_city_ids(path, expected_count):
    if not path.is_file():
        raise FileNotFoundError(f"Training-city metadata not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(
                f"Training-city metadata must contain an 'id' column: {path}"
            )
        raw_ids = [row["id"].strip() for row in reader]
    if any(not value for value in raw_ids):
        raise ValueError(f"Training-city metadata contains an empty id: {path}")
    try:
        city_ids = [int(value) for value in raw_ids]
    except ValueError as exc:
        raise ValueError(
            f"Training-city metadata contains a non-integer id: {path}"
        ) from exc
    if len(city_ids) != expected_count:
        raise ValueError(
            f"Expected {expected_count} training cities, found {len(city_ids)} in {path}."
        )
    if len(set(city_ids)) != len(city_ids):
        duplicates = sorted(
            city_id for city_id, count in Counter(city_ids).items() if count > 1
        )
        raise ValueError(f"Duplicate training city ids found in {path}: {duplicates}")
    return sorted(city_ids)


def select_candidates(candidates, sample_count, sampling_seed):
    if sample_count > len(candidates):
        raise ValueError(
            f"Cannot sample {sample_count} records without replacement from "
            f"a pool of {len(candidates)}."
        )
    rng = random.Random(sampling_seed)
    selected = rng.sample(candidates, sample_count)
    for selection_rank, candidate in enumerate(selected, start=1):
        candidate["selection_rank"] = selection_rank
    return selected


def build_prompts(sample, student_input_format="prompt"):
    structured = format_prompt_PPO(sample)
    if student_input_format == "prompt":
        student = structured
    elif student_input_format == "numeric_plain":
        student = format_numeric_plain_prompt_PPO(sample)
    else:
        raise ValueError(f"Unsupported student input format: {student_input_format}")
    target = float(_scalar(structured["ground_truth"]))
    student_prompt = student["prompt"]
    supervised_completion = student["completion"]
    teacher_prompt = (
        structured["prompt"]
        + structured["completion"]
        + "\n\n"
        + EXPLANATION_GUIDELINES
    )
    return student_prompt, supervised_completion, teacher_prompt, target


def parse_teacher_content(content, expected_target, tolerance=5e-4):
    if not content or not content.strip():
        raise ValueError("The API returned empty content.")
    match = RESULT_LINE_RE.match(content)
    if not match:
        raise ValueError("The API response does not start with the required result line.")
    generated_target = float(match.group(1))
    rendered_target = float(f"{expected_target:.3f}")
    if abs(generated_target - rendered_target) > tolerance:
        raise ValueError(
            f"The API changed the supplied prediction from {rendered_target:.3f} "
            f"to {generated_target:.3f}."
        )
    explanation = content[match.end() :].strip()
    missing = [heading for heading in SECTION_HEADINGS if heading not in explanation]
    if missing:
        raise ValueError(f"The API response is missing headings: {', '.join(missing)}")
    return explanation


def build_distillation_text(student_prompt, completion, explanation):
    return (
        student_prompt
        + completion.rstrip()
        + "\n\n"
        + explanation.strip()
        + "\n"
    )


def _client(api_key, base_url, timeout):
    metadata = (api_key, base_url, timeout)
    if getattr(_thread_local, "metadata", None) != metadata:
        _thread_local.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        _thread_local.metadata = metadata
    return _thread_local.client


def _atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _generate_job(job, args):
    student_prompt, completion, teacher_prompt, target = build_prompts(
        job["sample"], args.student_input_format
    )
    if args.reuse_content_dir is not None:
        selection_rank = job.get("selection_rank", job["sample_index"])
        source_candidates = [
            args.reuse_content_dir / f"sample_{selection_rank}.txt",
            args.reuse_content_dir
            / f"City{job['city_id']}"
            / f"sample_{job['sample_index']}.txt",
            args.reuse_content_dir
            / "content"
            / f"City{job['city_id']}"
            / f"sample_{job['sample_index']}.txt",
        ]
        source = next((path for path in source_candidates if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                "Missing reusable explanation; checked: "
                + ", ".join(str(path) for path in source_candidates)
            )
        source_text = source.read_text(encoding="utf-8")
        explanation_start = source_text.find(EXPLANATION_MARKER)
        if explanation_start < 0:
            raise ValueError(f"Reusable explanation has no {EXPLANATION_MARKER!r}: {source}")
        explanation = source_text[explanation_start:].strip()
        missing = [heading for heading in SECTION_HEADINGS if heading not in explanation]
        if missing:
            raise ValueError(
                f"Reusable explanation is missing headings in {source}: {', '.join(missing)}"
            )
        training_text = build_distillation_text(
            student_prompt, completion, explanation
        )
        _atomic_write(job["content_path"], training_text)
        _atomic_write(job["reasoning_path"], "\n")
        return {
            "city_id": job["city_id"],
            "sample_index": job["sample_index"],
            "target_month": job["target_month"],
            "status": "reserialized",
        }

    last_error = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            response = _client(
                args.api_key,
                args.base_url,
                args.timeout,
            ).chat.completions.create(
                model=args.api_model,
                messages=[{"role": "user", "content": teacher_prompt}],
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
            )
            message = response.choices[0].message
            raw_content = message.content or ""
            explanation = parse_teacher_content(raw_content, target)
            training_text = build_distillation_text(
                student_prompt,
                completion,
                explanation,
            )
            reasoning = getattr(message, "reasoning_content", None) or ""
            _atomic_write(job["content_path"], training_text)
            _atomic_write(job["reasoning_path"], reasoning.rstrip() + "\n")
            return {
                "city_id": job["city_id"],
                "sample_index": job["sample_index"],
                "target_month": job["target_month"],
                "status": "generated",
            }
        except Exception as exc:  # Network and response-format failures are retried together.
            last_error = exc
            if attempt < args.max_attempts:
                time.sleep(args.retry_delay * (2 ** (attempt - 1)))
    raise RuntimeError(
        f"City {job['city_id']} sample {job['sample_index']} "
        f"({job['target_month']}) failed after {args.max_attempts} attempts: {last_error}"
    )


def _build_jobs(args):
    candidates = []
    city_ids = load_training_city_ids(
        args.train_city_metrics,
        args.expected_train_cities,
    )
    for city_id in city_ids:
        city_path = args.data_dir / "train" / f"UHI_city_{city_id}"
        if not city_path.is_dir():
            raise FileNotFoundError(f"Missing training city dataset: {city_path}")
        dataset = load_from_disk(str(city_path))
        samples = validate_city_samples(
            list(dataset),
            city_id,
            args.expected_train_samples_per_city,
            args.first_target,
            args.last_target,
        )
        for index, sample in enumerate(samples, start=1):
            year, month = _month_key(sample)
            candidates.append(
                {
                    "city_id": city_id,
                    "sample_index": index,
                    "target_month": f"{year:04d}-{month:02d}",
                    "sample": sample,
                }
            )
    selected = select_candidates(candidates, args.sample_count, args.sampling_seed)
    for job in selected:
        filename = f"sample_{job['sample_index']}.txt"
        job["content_path"] = (
            args.output_dir / "content" / f"City{job['city_id']}" / filename
        )
        job["reasoning_path"] = (
            args.output_dir / "reasoning" / f"City{job['city_id']}" / filename
        )
    return selected, city_ids, len(candidates)


def _is_complete(job, student_input_format="prompt"):
    if not job["content_path"].is_file() or not job["reasoning_path"].is_file():
        return False
    if job["content_path"].stat().st_size == 0:
        return False
    try:
        student_prompt, completion, _, _ = build_prompts(
            job["sample"], student_input_format
        )
        text = job["content_path"].read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return False
    expected_prefix = student_prompt + completion.rstrip()
    return text.startswith(expected_prefix) and all(
        heading in text for heading in SECTION_HEADINGS
    )


def _selection_payload(args, jobs, city_ids, candidate_count):
    selected_counts = Counter(job["city_id"] for job in jobs)
    samples = [
        {
            "selection_rank": job["selection_rank"],
            "city_id": job["city_id"],
            "source_sample_index": job["sample_index"],
            "target_month": job["target_month"],
            "content_file": str(job["content_path"].relative_to(args.output_dir)),
            "reasoning_file": str(job["reasoning_path"].relative_to(args.output_dir)),
        }
        for job in jobs
    ]
    digest = hashlib.sha256(
        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "source_split": "train",
        "training_city_metrics": str(args.train_city_metrics.resolve()),
        "expected_train_cities": args.expected_train_cities,
        "training_city_ids": city_ids,
        "target_range": {
            "first": f"{args.first_target[0]:04d}-{args.first_target[1]:02d}",
            "last": f"{args.last_target[0]:04d}-{args.last_target[1]:02d}",
        },
        "candidate_samples": candidate_count,
        "sampling_method": "simple_random_without_replacement",
        "sampling_seed": args.sampling_seed,
        "selected_samples": len(jobs),
        "selected_cities": len(selected_counts),
        "selected_samples_per_city": {
            str(city_id): selected_counts[city_id] for city_id in sorted(selected_counts)
        },
        "selection_sha256": digest,
        "samples": samples,
    }


def _ensure_selection_manifest(args, payload):
    path = args.output_dir / "selection_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Existing selection manifest does not match the requested sampling "
                f"configuration: {path}. Use a new EXPLANATION_DATA_ROOT."
            )
        return
    existing_samples = list((args.output_dir / "content").rglob("sample_*.txt"))
    if existing_samples:
        raise RuntimeError(
            f"Found explanation files without a selection manifest in {args.output_dir}. "
            "Use a new EXPLANATION_DATA_ROOT so samples cannot be mixed."
        )
    _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_manifest(args, selection_payload, generated, skipped):
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_model": args.api_model,
        "student_input_format": args.student_input_format,
        "reuse_content_dir": (
            str(args.reuse_content_dir.resolve())
            if args.reuse_content_dir is not None
            else None
        ),
        "source_split": "train",
        "selection_manifest": "selection_manifest.json",
        "selection_sha256": selection_payload["selection_sha256"],
        "sampling_seed": args.sampling_seed,
        "candidate_samples": selection_payload["candidate_samples"],
        "selected_cities": selection_payload["selected_cities"],
        "expected_total_samples": args.sample_count,
        "generated_this_run": generated,
        "reused_this_run": skipped,
        "content_files": sum(1 for _ in (args.output_dir / "content").rglob("*.txt")),
        "reasoning_files": sum(1 for _ in (args.output_dir / "reasoning").rglob("*.txt")),
        "generation_parameters": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
        },
        "guidelines_sha256": hashlib.sha256(
            EXPLANATION_GUIDELINES.encode("utf-8")
        ).hexdigest(),
    }
    _atomic_write(
        args.output_dir / "generation_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "explanation_samples",
    )
    parser.add_argument(
        "--train-city-metrics",
        type=Path,
        default=ROOT / "data" / "city_metrics" / "city_metrics_train.csv",
    )
    parser.add_argument("--expected-train-cities", type=int, default=500)
    parser.add_argument("--expected-train-samples-per-city", type=int, default=168)
    parser.add_argument("--sample-count", type=int, default=900)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--first-target", type=_parse_month, default=(2004, 1))
    parser.add_argument("--last-target", type=_parse_month, default=(2017, 12))
    parser.add_argument("--api-model", default="deepseek-reasoner")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="Defaults to DEEPSEEK_API_KEY. Prefer the environment variable.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--presence-penalty", type=float, default=0.1)
    parser.add_argument(
        "--student-input-format",
        choices=("prompt", "numeric_plain"),
        default="prompt",
    )
    parser.add_argument(
        "--reuse-content-dir",
        type=Path,
        default=None,
        help=(
            "Reuse the explanation sections from an existing content directory "
            "and only reserialize the student prompt. No API calls are made."
        ),
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all datasets and prompts without calling the API or writing files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for name in (
        "expected_train_cities",
        "expected_train_samples_per_city",
        "sample_count",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.concurrency <= 0 or args.max_attempts <= 0:
        raise ValueError("--concurrency and --max-attempts must be positive.")

    jobs, city_ids, candidate_count = _build_jobs(args)
    if len(jobs) != args.sample_count:
        raise RuntimeError(f"Expected {args.sample_count} jobs, built {len(jobs)}.")
    selection_payload = _selection_payload(args, jobs, city_ids, candidate_count)

    # Render every prompt before any paid request so serialization failures are free.
    for job in jobs:
        build_prompts(job["sample"], args.student_input_format)
    selected_city_count = len({job["city_id"] for job in jobs})
    print(
        f"Validated {len(city_ids)} training cities and {candidate_count} candidates: "
        f"{args.first_target[0]:04d}-{args.first_target[1]:02d} to "
        f"{args.last_target[0]:04d}-{args.last_target[1]:02d}. "
        f"Selected {len(jobs)} samples across all {selected_city_count} cities "
        f"without replacement using seed {args.sampling_seed}."
    )
    if args.dry_run:
        print("Dry run complete; no API requests or files were written.")
        return

    _ensure_selection_manifest(args, selection_payload)

    pending = (
        jobs
        if args.overwrite
        else [
            job
            for job in jobs
            if not _is_complete(job, args.student_input_format)
        ]
    )
    skipped = len(jobs) - len(pending)
    if pending and args.reuse_content_dir is None and not args.api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required because explanation samples are missing."
        )
    print(f"Explanation generation: pending={len(pending)}, reusable={skipped}")

    generated = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_generate_job, job, args): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                generated += 1
                print(
                    f"Generated City{result['city_id']} sample_{result['sample_index']} "
                    f"({result['target_month']}); {generated}/{len(pending)}"
                )
            except Exception as exc:
                failures.append(str(exc))
                print(f"ERROR: {exc}", file=sys.stderr)

    if failures:
        raise RuntimeError(
            f"{len(failures)} explanation request(s) failed. Rerun the command to resume."
        )
    incomplete = [
        job
        for job in jobs
        if not _is_complete(job, args.student_input_format)
    ]
    if incomplete:
        raise RuntimeError(f"Generation ended with {len(incomplete)} incomplete samples.")
    _write_manifest(args, selection_payload, generated, skipped)
    print(f"Explanation dataset complete: {len(jobs)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
