
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta
from datasets import load_from_disk
from openai import OpenAI
from torch.utils.data import DataLoader

from utils import create_prompt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_length', type=int, default=1300)
    parser.add_argument('--model', type=str, default='deepseek-8b')
    parser.add_argument('--target_modules', nargs='*', default=["q_proj", "v_proj"])
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, "Dataset"),
    )
    parser.add_argument('--city_ids', type=int, nargs='+')
    args = parser.parse_args()


    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set the DEEPSEEK_API_KEY environment variable for DeepSeek access.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    df_train = pd.read_csv(args.dataset_path + "/city_metrics/city_metrics_train.csv")
    city_id_list = df_train['id'].tolist()
    if args.city_ids is not None:
        if len(args.city_ids) < 1:
            raise ValueError("city_ids must contain at least one city id.")
        city_id_list = args.city_ids

    for city_id in city_id_list:
        train_dataset = load_from_disk(f'{args.dataset_path}/train/UHI_city_{city_id}')
        train_loader = DataLoader(train_dataset, batch_size=1, drop_last=True)
        content_folder = f"./explanation_samples/content/City{city_id}"
        reasoning_folder = f"./explanation_samples/reasoning/City{city_id}"
        os.makedirs(content_folder, exist_ok=True)
        os.makedirs(reasoning_folder, exist_ok=True)

        for index, info in enumerate(train_loader):
            print(f"Processing city {index}")
            prompt = create_prompt(info, 0)
            prompt += '''
Considering the important features related to UHI prediction, please think step by step, and give the explaination for the next month's UHI result
Explanation Guidelines:
Structure your analysis into 5 sections with headings (no markdown):
1. Geographical Context:
    Analyze the regional geographic setting based on the provided coordinates. Explain how these baseline geographic conditions influence UHI.
2. Structural Contextual Influence
    Identify dominant Structure Factors. Describe their impact on UHI.
3. Dynamic Environmental Modulation
    Identify critical Liquidity Factors. Analyze their monthly trends and interactions with structural features to modulate UHI.
4. Cross-Factor Synergy
    Highlight key interactions between structural and dynamic factors. Specify whether these interactions amplify or counteract UHI.
5. Prediction Synthesis
    Summarize the primary causal chain linking geographic context, Structure Factors, and Liquidity Factors. Conclude with the dominant mechanisms governing the predicted UHI outcome(increase or decrease compared to the last few months).

Response Format Requirements:
Prediction: Start your answer with "Result: Predicted UHI intensity for the next month: [value]"
Explanation: Use exactly 5 paragraphs matching the section headings above.
Language: Use semicolons (;) to separate related observations within sentences. Avoid bullet points.
Don't quote data directly.

Give the prediction and explanation without knowing the ground truth UHI value. Your analysis must strictly derive from the input features, as if you were making a real-time forecast. Do not retroactively adjust reasoning based on the final answer.

'''

            print("Generating response...")
            

            # Round 1
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )

            reasoning_content = response.choices[0].message.reasoning_content
            content = response.choices[0].message.content
            print(reasoning_content)
            print(content)

            output_file = f"sample_{index + 1}.txt"
            with open(f"{content_folder}/{output_file}", 'w', encoding='utf-8') as f:
                f.write(prompt)
                f.write(content)
                print(f"Save content to {content_folder}/{output_file} done.\n\n")

            with open(f"{reasoning_folder}/{output_file}", 'w', encoding='utf-8') as f:
                f.write(reasoning_content)
                print(f"Save reasoning content to {reasoning_folder}/{output_file} done.\n\n")

            

