import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from output_parsing import parse_prediction  # noqa: E402
from sequence_dataset import load_city_dataset  # noqa: E402
from utils import format_prompt_PPO  # noqa: E402


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

PREDICTION_SUFFIX = "\n\nPrediction\n\n[Result:] Predicted UHI intensity for the next month:"

def add_explanation_guidelines(prompt):
    if not prompt.endswith(PREDICTION_SUFFIX):
        raise ValueError("Structured prompt has an unexpected prediction suffix.")
    task_prompt = prompt[: -len(PREDICTION_SUFFIX)]
    return task_prompt + "\n\n" + EXPLANATION_GUIDELINES + PREDICTION_SUFFIX


def scalar(value):
    if isinstance(value, (list, tuple)):
        return scalar(value[0])
    if hasattr(value, "reshape"):
        return value.reshape(-1)[0].item()
    if hasattr(value, "item"):
        return value.item()
    return value


def selected_sample(data_root, city_id, year, month):
    dataset = load_city_dataset(data_root, city_id, "valid")
    matches = [
        sample
        for sample in dataset
        if int(scalar(sample["year_info"])) == year
        and int(scalar(sample["month_info"])) == month
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one sample for City{city_id} {year:04d}-{month:02d}; "
            f"found {len(matches)}."
        )
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city-ids", type=int, nargs="+", required=True)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=1300)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--include-explanation-guidelines",
        action="store_true",
        help="Insert the six-section response guidelines before the prediction cue.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir).to(args.device)
    model.eval()

    summary = []
    for sample_index, city_id in enumerate(args.city_ids, start=1):
        sample = selected_sample(
            args.data_root, city_id, args.year, args.month
        )
        formatted = format_prompt_PPO(sample)
        prompt = formatted["prompt"]
        if args.include_explanation_guidelines:
            prompt = add_explanation_guidelines(prompt)
        true_uhi = float(scalar(formatted["ground_truth"]))
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, encoded.input_ids.size(1) :]
        output = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        parsed = parse_prediction(
            output,
            valid_min=-5.0,
            valid_max=5.0,
            max_decimal_places=3,
        )
        record = {
            "sample": sample_index,
            "city_id": city_id,
            "target_month": f"{args.year:04d}-{args.month:02d}",
            "explanation_guidelines_in_prompt": args.include_explanation_guidelines,
            "true_uhi": true_uhi,
            "parsed_prediction": parsed.parsed_value,
            "valid_prediction": parsed.is_valid,
            "generated_characters": len(output),
        }
        summary.append(record)
        rendered = (
            f"City ID: {city_id}\n"
            f"Target month: {args.year:04d}-{args.month:02d}\n"
            f"True UHI: {true_uhi:.6f} °C\n"
            f"Parsed prediction: {parsed.parsed_value}\n"
            f"Valid prediction: {parsed.is_valid}\n\n"
            f"{output}\n"
        )
        output_path = args.output_dir / f"sample_{sample_index}.txt"
        output_path.write_text(rendered, encoding="utf-8")
        print(
            f"Saved {output_path}: city={city_id}, prediction={parsed.parsed_value}, "
            f"valid_prediction={parsed.is_valid}",
            flush=True,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    valid_predictions = sum(record["valid_prediction"] for record in summary)
    prediction_summary = {
        "samples": len(summary),
        "valid_predictions": valid_predictions,
        "invalid_predictions": len(summary) - valid_predictions,
    }
    (args.output_dir / "format_validation.json").write_text(
        json.dumps(prediction_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(prediction_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
