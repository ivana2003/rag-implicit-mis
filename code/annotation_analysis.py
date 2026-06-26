import json
import re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter


def normalize_value(val):
    try:
        return str(int(float(val))).strip()
    except Exception:
        return str(val).strip().lower()


def normalize_hp(val):
    if pd.isna(val):
        return None
    val = str(val).strip().upper()
    val = val.replace(' ', '_').replace('SKANG', 'SLANG').replace('BASELIINE', 'BASELINE')
    if val in ('TIE', 'TIE_'):
        return 'TIE'
    if val.startswith('TIE'):
        parts = val.split('_')
        tie_parts = sorted([p for p in parts if p not in ('TIE', '')])
        return 'TIE_' + '_'.join(tie_parts) if tie_parts else 'TIE'
    return val


def find_disagreements(ann1, ann2):
    faith_cols = [c for c in ann1.columns if 'faithfulness' in c.lower()]
    hp_cols = [c for c in ann1.columns if 'human_preference' in c.lower()]
    ann_cols = faith_cols + hp_cols

    disagreements = []
    for idx in range(min(len(ann1), len(ann2))):
        for col in ann_cols:
            if col not in ann2.columns:
                continue
            v1 = normalize_value(ann1.iloc[idx].get(col, ''))
            v2 = normalize_value(ann2.iloc[idx].get(col, ''))
            if v1 != v2 and v1 != 'nan' and v2 != 'nan':
                disagreements.append({
                    'idx': idx,
                    'id': ann1.iloc[idx].get('id', idx),
                    'column': col,
                    'annotator1': v1,
                    'annotator2': v2,
                })
    return pd.DataFrame(disagreements)


def merge_annotations(df_sample, ann1, ann2, lang='it'):
    df = df_sample.copy()

    kb_map = {'A': 'WIKI', 'B': 'SLANG', 'C': 'LITERATURE', 'D': 'BASELINE'}

    for letter, kb in kb_map.items():
        kb_lower = kb.lower()
        faith_col = f'faithfulness_{letter}'
        if faith_col in ann1.columns:
            df[f'{kb_lower}_faithfulness_annotator1'] = ann1.set_index('id').reindex(df['id'])[faith_col].values
        if faith_col in ann2.columns:
            df[f'{kb_lower}_faithfulness_annotator2'] = ann2.set_index('id').reindex(df['id'])[faith_col].values

    hp_col = 'Human Preference'
    if hp_col in ann1.columns:
        df['human_preference_annotator1'] = ann1.set_index('id').reindex(df['id'])[hp_col].apply(normalize_hp).values
    if hp_col in ann2.columns:
        df['human_preference_annotator2'] = ann2.set_index('id').reindex(df['id'])[hp_col].apply(normalize_hp).values

    return df


def faithfulness_stats(df, kbs=None):
    if kbs is None:
        kbs = ['wiki', 'slang', 'literature', 'baseline']
    stats = {}
    for kb in kbs:
        for ann in ['annotator1', 'annotator2']:
            col = f'{kb}_faithfulness_{ann}'
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            stats[f'{kb}_{ann}'] = {
                'n': len(vals),
                'mean': vals.mean(),
                'std': vals.std(),
                'sum': vals.sum(),
            }
    return pd.DataFrame(stats).T


def human_preference_stats(df, rag_kbs=None):
    if rag_kbs is None:
        rag_kbs = ['WIKI', 'SLANG', 'LITERATURE']
    results = {}
    for ann in ['annotator1', 'annotator2']:
        col = f'human_preference_{ann}'
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        rag_wins = sum(1 for v in vals
                       if any(kb in str(v) for kb in rag_kbs)
                       and 'BASELINE' not in str(v)
                       and 'TIE' not in str(v)
                       and v != 'NEITHER')
        base_wins = sum(1 for v in vals if v == 'BASELINE')
        ties = sum(1 for v in vals
                   if 'TIE' in str(v) or v == 'TIE'
                   or (any(kb in str(v) for kb in rag_kbs) and 'BASELINE' in str(v)))
        neither = sum(1 for v in vals if v == 'NEITHER')
        results[ann] = {
            'rag_wins': rag_wins,
            'baseline_wins': base_wins,
            'ties': ties,
            'neither': neither,
            'total': len(vals),
        }
    return pd.DataFrame(results).T


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-1', required=True, help='Annotation file for annotator 1')
    parser.add_argument('--ann-2', required=True, help='Annotation file for annotator 2')
    parser.add_argument('--output-disagreements', default='disagreements.csv')
    parser.add_argument('--output-stats', default='stats.csv')
    args = parser.parse_args()

    ext1 = args.ann_1.split('.')[-1]
    ext2 = args.ann_2.split('.')[-1]
    ann1 = pd.read_excel(args.ann_1) if ext1 == 'xlsx' else pd.read_csv(args.ann_1)
    ann2 = pd.read_excel(args.ann_2) if ext2 == 'xlsx' else pd.read_csv(args.ann_2)

    disagreements = find_disagreements(ann1, ann2)
    disagreements.to_csv(args.output_disagreements, index=False)
    print(f"Disagreements: {len(disagreements)} → {args.output_disagreements}")

    merged = merge_annotations(ann1, ann1, ann2)
    stats = faithfulness_stats(merged)
    stats.to_csv(args.output_stats)
    print(f"Stats saved to {args.output_stats}")
    print(stats.to_string())
