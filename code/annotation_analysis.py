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


def find_disagreements(ari, ivi):
    faith_cols = [c for c in ari.columns if 'faithfulness' in c.lower()]
    hp_cols = [c for c in ari.columns if 'human_preference' in c.lower()]
    ann_cols = faith_cols + hp_cols

    disagreements = []
    for idx in range(min(len(ari), len(ivi))):
        for col in ann_cols:
            if col not in ivi.columns:
                continue
            v_ari = normalize_value(ari.iloc[idx].get(col, ''))
            v_ivi = normalize_value(ivi.iloc[idx].get(col, ''))
            if v_ari != v_ivi and v_ari != 'nan' and v_ivi != 'nan':
                disagreements.append({
                    'idx': idx,
                    'id': ari.iloc[idx].get('id', idx),
                    'column': col,
                    'ARI': v_ari,
                    'IVANA': v_ivi,
                })
    return pd.DataFrame(disagreements)


def merge_annotations(df_sample, ann_ari, ann_ivi, lang='it'):
    df = df_sample.copy()

    if lang == 'it':
        kb_map = {'A': 'WIKI', 'B': 'SLANG', 'C': 'LITERATURE', 'D': 'BASELINE'}
    else:
        kb_map = {'A': 'WIKI', 'B': 'SLANG', 'C': 'LITERATURE', 'D': 'BASELINE'}

    for letter, kb in kb_map.items():
        kb_lower = kb.lower()
        faith_col = f'faithfulness_{letter}'
        if faith_col in ann_ari.columns:
            df[f'{kb_lower}_faithfulness_ARI'] = ann_ari.set_index('id').reindex(df['id'])[faith_col].values
        if faith_col in ann_ivi.columns:
            df[f'{kb_lower}_faithfulness_IVANA'] = ann_ivi.set_index('id').reindex(df['id'])[faith_col].values

    hp_col = 'Human Preference'
    if hp_col in ann_ari.columns:
        df['human_preference_ARI'] = ann_ari.set_index('id').reindex(df['id'])[hp_col].apply(normalize_hp).values
    if hp_col in ann_ivi.columns:
        df['human_preference_IVANA'] = ann_ivi.set_index('id').reindex(df['id'])[hp_col].apply(normalize_hp).values

    return df


def faithfulness_stats(df, kbs=None):
    if kbs is None:
        kbs = ['wiki', 'slang', 'literature', 'baseline']
    stats = {}
    for kb in kbs:
        for ann in ['ARI', 'IVANA']:
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
    for ann in ['ARI', 'IVANA']:
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
    parser.add_argument('--ann-ari', required=True)
    parser.add_argument('--ann-ivi', required=True)
    parser.add_argument('--output-disagreements', default='disagreements.csv')
    parser.add_argument('--output-stats', default='stats.csv')
    args = parser.parse_args()

    ext_ari = args.ann_ari.split('.')[-1]
    ext_ivi = args.ann_ivi.split('.')[-1]
    ann_ari = pd.read_excel(args.ann_ari) if ext_ari == 'xlsx' else pd.read_csv(args.ann_ari)
    ann_ivi = pd.read_excel(args.ann_ivi) if ext_ivi == 'xlsx' else pd.read_csv(args.ann_ivi)

    disagreements = find_disagreements(ann_ari, ann_ivi)
    disagreements.to_csv(args.output_disagreements, index=False)
    print(f"Disagreements: {len(disagreements)} → {args.output_disagreements}")

    merged = merge_annotations(ann_ari, ann_ari, ann_ivi)
    stats = faithfulness_stats(merged)
    stats.to_csv(args.output_stats)
    print(f"Stats saved to {args.output_stats}")
    print(stats.to_string())
