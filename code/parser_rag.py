import json
import re
import os
import numpy as np
import pandas as pd
from pathlib import Path


def parse_raw(raw_str, lang='it'):
    if not isinstance(raw_str, str) or raw_str.strip() == '':
        return '', ''
    try:
        d = json.loads(raw_str)
        if lang == 'it':
            assumption = d.get('assunto_implicito', d.get('implied_assumption', ''))
            label = d.get('risposta', d.get('answer', ''))
        else:
            assumption = d.get('implied_assumption', d.get('assunto_implicito', ''))
            label = d.get('answer', d.get('risposta', ''))
        return assumption, label
    except Exception:
        try:
            if lang == 'it':
                a = re.search(r'"assunto_implicito"\s*:\s*"(.+?)"(?=\s*,|\s*\})', raw_str, re.DOTALL)
                l = re.search(r'"risposta"\s*:\s*"(.+?)"(?=\s*\})', raw_str, re.DOTALL)
            else:
                a = re.search(r'"implied_assumption"\s*:\s*"(.+?)"(?=\s*,|\s*\})', raw_str, re.DOTALL)
                l = re.search(r'"answer"\s*:\s*"(.+?)"(?=\s*\})', raw_str, re.DOTALL)
            return (a.group(1) if a else ''), (l.group(1) if l else '')
        except Exception:
            return '', ''


def clean_csv(filepath, id_col='id'):
    df = pd.read_csv(filepath, quoting=1)
    df['_id_num'] = pd.to_numeric(df[id_col], errors='coerce')
    df = df[df['_id_num'].notna()].copy()
    df[id_col] = df['_id_num'].astype(int)
    df.drop(columns=['_id_num'], inplace=True)
    return df.reset_index(drop=True)


def concat_batches(batch_dir, prefix, n_batches=4, output_path=None):
    dfs = []
    for i in range(1, n_batches + 1):
        path = os.path.join(batch_dir, f'{prefix}_b{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=['text'])
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No batch files found for prefix '{prefix}' in {batch_dir}")
    result = pd.concat(dfs, ignore_index=True)
    if output_path:
        result.to_csv(output_path, index=False)
        print(f"Saved {len(result)} rows to {output_path}")
    return result


def extract_labels_assumptions(df, lang='it',
                                raw_rag_col='raw_RAG',
                                raw_norag_col='raw_NORAG'):
    df = df.copy()
    if raw_rag_col in df.columns:
        df['assumption_RAG'] = df[raw_rag_col].apply(lambda x: parse_raw(x, lang)[0])
        df['label_RAG'] = df[raw_rag_col].apply(lambda x: parse_raw(x, lang)[1])
    if raw_norag_col in df.columns:
        df['assumption_NORAG'] = df[raw_norag_col].apply(lambda x: parse_raw(x, lang)[0])
        df['label_NORAG'] = df[raw_norag_col].apply(lambda x: parse_raw(x, lang)[1])
    return df


def process_directory(input_dir, output_dir, lang='it',
                       raw_rag_col='raw_RAG', raw_norag_col='raw_NORAG'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        df = clean_csv(os.path.join(input_dir, filename))
        df = extract_labels_assumptions(df, lang, raw_rag_col, raw_norag_col)
        out_path = os.path.join(output_dir, filename)
        df.to_csv(out_path, index=False)
        n_rag = (df.get('label_RAG', pd.Series()) == ('SÌ' if lang == 'it' else 'YES')).sum()
        n_norag = (df.get('label_NORAG', pd.Series()) == ('SÌ' if lang == 'it' else 'YES')).sum()
        print(f"{filename}: {len(df)} rows | RAG YES={n_rag} | NORAG YES={n_norag}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p_process = subparsers.add_parser('process', help='Process a directory of CSVs')
    p_process.add_argument('--input-dir', required=True)
    p_process.add_argument('--output-dir', required=True)
    p_process.add_argument('--lang', default='it', choices=['it', 'en'])
    p_process.add_argument('--raw-rag-col', default='raw_RAG')
    p_process.add_argument('--raw-norag-col', default='raw_NORAG')

    p_concat = subparsers.add_parser('concat', help='Concatenate batch files')
    p_concat.add_argument('--batch-dir', required=True)
    p_concat.add_argument('--prefix', required=True)
    p_concat.add_argument('--n-batches', type=int, default=4)
    p_concat.add_argument('--output', required=True)

    p_single = subparsers.add_parser('single', help='Process a single CSV file')
    p_single.add_argument('--input', required=True)
    p_single.add_argument('--output', required=True)
    p_single.add_argument('--lang', default='it', choices=['it', 'en'])

    args = parser.parse_args()

    if args.command == 'process':
        process_directory(args.input_dir, args.output_dir, args.lang,
                          args.raw_rag_col, args.raw_norag_col)
    elif args.command == 'concat':
        concat_batches(args.batch_dir, args.prefix, args.n_batches, args.output)
    elif args.command == 'single':
        df = clean_csv(args.input)
        df = extract_labels_assumptions(df, args.lang)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
