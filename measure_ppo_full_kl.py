"""Measure exact vocabulary KL on fixed held-out response contexts.

Reference = base policy with LoRA disabled, matching common.models.reference_mode.
Run once per PPO adapter, with identical --contexts for controlled comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.data import load_yaml
from common.models import load_policy, load_tokenizer, reference_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ppo.yaml')
    parser.add_argument('--adapter', required=True, help='Saved PPO LoRA adapter directory')
    parser.add_argument('--contexts', default='results/task2_ppo/kl_0_eval_generations.jsonl', help='Fixed held-out prompt/response JSONL')
    parser.add_argument('--max-examples', type=int, default=50)
    parser.add_argument('--output', required=True, help='Results CSV path')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg['base_model'])
    model = load_policy(cfg, adapter_path=args.adapter, trainable=False)
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with open(args.contexts, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = rows[:args.max_examples]
    if not rows:
        raise ValueError('No evaluation records found.')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_kl, total_tokens = 0., 0
    results = []
    max_prompt = int(cfg['max_prompt_length'])
    # The saved evaluation uses a maximum response length of 768.
    max_resp = int(cfg['eval_max_response_length'])
    for idx, row in enumerate(rows, 1):
        prompt = tok.apply_chat_template(row['prompt_messages'], tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt)['input_ids']
        response_ids = tok(row['response'], add_special_tokens=False)['input_ids'][:max_resp]
        if not prompt_ids or not response_ids:
            continue
        # Calculate p(next token | prompt, preceding response tokens).
        ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        start = len(prompt_ids) - 1
        end = start + len(response_ids)
        with torch.inference_mode():
            logits_p = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, start:end, :]
            logp = F.log_softmax(logits_p.float(), dim=-1)
            del logits_p
            with reference_mode(model):
                logits_q = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, start:end, :]
                logq = F.log_softmax(logits_q.float(), dim=-1)
                del logits_q
            # KL(p || q) = sum_v p(v) [log p(v) - log q(v)].
            token_kl = (logp.exp() * (logp - logq)).sum(dim=-1).squeeze(0)
            # Float rounding may yield tiny negative values; exact KL is >= 0.
            per_example = float(token_kl.mean().item())
            if per_example < -1e-5:
                raise RuntimeError(f'Unexpected negative KL: {per_example}')
            per_example = max(0.0, per_example)
            n = len(response_ids)
            total_kl += per_example * n
            total_tokens += n
        results.append({'evaluation_index':row.get('evaluation_index',idx-1), 'prompt_id':row.get('prompt_id'), 'tokens':n, 'full_kl_per_token_nats':per_example})
        if idx % 10 == 0 or idx == len(rows):
            print(f'{idx}/{len(rows)} prompts | token-weighted full KL = {total_kl/total_tokens:.9f} nats/token', flush=True)
        del ids, mask, logp, logq, token_kl
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['evaluation_index','prompt_id','tokens','full_kl_per_token_nats'])
        writer.writeheader()
        writer.writerows(results)
    print(f'\nFINAL full-distribution KL: {total_kl/total_tokens:.9f} nats/token')
    print(f'Scored: {len(results)} responses; {total_tokens} response tokens')
    print(f'Saved: {output}')
    print('Reference: base model (LoRA disabled); contexts: fixed source response texts.')

if __name__ == '__main__':
    main()
