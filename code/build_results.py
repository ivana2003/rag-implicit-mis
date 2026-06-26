import json
import re
import numpy as np
import pandas as pd


def parse_raw_ita(raw_str):
    if pd.isna(raw_str) or raw_str == '':
        return '', ''
    try:
        d = json.loads(str(raw_str))
        return d.get('assunto_implicito', d.get('implied_assumption', '')), \
               d.get('risposta', d.get('answer', ''))
    except Exception:
        try:
            assumption = re.search(
                r'"assunto_implicito"\s*:\s*"(.+?)"(?=\s*,|\s*\})', str(raw_str), re.DOTALL)
            label = re.search(
                r'"risposta"\s*:\s*"(.+?)"(?=\s*\})', str(raw_str), re.DOTALL)
            return (assumption.group(1) if assumption else '',
                    label.group(1) if label else '')
        except Exception:
            return '', ''


def parse_raw_eng(raw_str):
    if pd.isna(raw_str) or raw_str == '':
        return '', ''
    try:
        d = json.loads(str(raw_str))
        return d.get('implied_assumption', ''), d.get('answer', '')
    except Exception:
        return '', ''


def build_full_results_ita(norag, wiki, slang, lit, ann=None, truncated_ids=None):
    df = norag[['id', 'text', 'Contesto']].copy()

    df['baseline_assumption'] = norag['raw_NORAG'].apply(lambda x: parse_raw_ita(x)[0])
    df['baseline_label'] = norag['raw_NORAG'].apply(lambda x: parse_raw_ita(x)[1])
    df['baseline_raw'] = norag['raw_NORAG']

    df = df.merge(wiki[['id', 'retrieval_text_RAG', 'assumption_RAG', 'label_RAG', 'raw_RAG']],
                  on='id', how='left')
    df.rename(columns={'retrieval_text_RAG': 'wiki_retrieval',
                       'assumption_RAG': 'wiki_assumption',
                       'label_RAG': 'wiki_label',
                       'raw_RAG': 'wiki_raw'}, inplace=True)

    df = df.merge(slang[['id', 'slang_compressed', 'assumption_RAG', 'label_RAG', 'raw_RAG']],
                  on='id', how='left')
    df.rename(columns={'slang_compressed': 'slang_retrieval',
                       'assumption_RAG': 'slang_assumption',
                       'label_RAG': 'slang_label',
                       'raw_RAG': 'slang_raw'}, inplace=True)

    df = df.merge(lit[['id', 'lit_compressed', 'assumption', 'label', 'raw_output']],
                  on='id', how='left')
    df.rename(columns={'lit_compressed': 'lit_retrieval',
                       'assumption': 'lit_assumption',
                       'label': 'lit_label',
                       'raw_output': 'lit_raw'}, inplace=True)

    if truncated_ids:
        mask = df['id'].isin(truncated_ids)
        df.loc[mask, ['baseline_assumption', 'baseline_label']] = np.nan

    if ann is not None:
        ann_cols = ['id',
                    'wiki_faithfulness_annotator1', 'wiki_faithfulness_annotator2',
                    'slang_faithfulness_annotator1', 'slang_faithfulness_annotator2',
                    'literature_faithfulness_annotator1', 'literature_faithfulness_annotator2',
                    'baseline_faithfulness_annotator1', 'baseline_faithfulness_annotator2',
                    'human_preference_annotator1', 'human_preference_annotator2']
        df = df.merge(ann[[c for c in ann_cols if c in ann.columns]],
                      on='id', how='left')

    ordered = ['id', 'text', 'Contesto',
               'baseline_assumption', 'baseline_label', 'baseline_raw',
               'baseline_faithfulness_annotator1', 'baseline_faithfulness_annotator2',
               'wiki_retrieval', 'wiki_assumption', 'wiki_label', 'wiki_raw',
               'wiki_faithfulness_annotator1', 'wiki_faithfulness_annotator2',
               'slang_retrieval', 'slang_assumption', 'slang_label', 'slang_raw',
               'slang_faithfulness_annotator1', 'slang_faithfulness_annotator2',
               'lit_retrieval', 'lit_assumption', 'lit_label', 'lit_raw',
               'literature_faithfulness_annotator1', 'literature_faithfulness_annotator2',
               'human_preference_annotator1', 'human_preference_annotator2']

    return df[[c for c in ordered if c in df.columns]]


def build_full_results_eng(norag, wiki, slang, lit, ann=None):
    df = norag[['id', 'text']].copy()

    df['baseline_assumption'] = norag['raw_NORAG'].apply(lambda x: parse_raw_eng(x)[0])
    df['baseline_label'] = norag['raw_NORAG'].apply(lambda x: parse_raw_eng(x)[1])
    df['baseline_raw'] = norag['raw_NORAG']

    df = df.merge(wiki[['id', 'retrieval_text_RAG', 'assumption_RAG', 'label_RAG', 'raw_RAG']],
                  on='id', how='left')
    df.rename(columns={'retrieval_text_RAG': 'wiki_retrieval',
                       'assumption_RAG': 'wiki_assumption',
                       'label_RAG': 'wiki_label',
                       'raw_RAG': 'wiki_raw'}, inplace=True)

    df = df.merge(slang[['id', 'slang_compressed', 'assumption_RAG', 'label_RAG', 'raw_RAG']],
                  on='id', how='left')
    df.rename(columns={'slang_compressed': 'slang_retrieval',
                       'assumption_RAG': 'slang_assumption',
                       'label_RAG': 'slang_label',
                       'raw_RAG': 'slang_raw'}, inplace=True)

    df = df.merge(lit[['id', 'lit_compressed', 'assumption', 'label', 'raw_output']],
                  on='id', how='left')
    df.rename(columns={'lit_compressed': 'lit_retrieval',
                       'assumption': 'lit_assumption',
                       'label': 'lit_label',
                       'raw_output': 'lit_raw'}, inplace=True)

    if ann is not None:
        ann_cols = ['id',
                    'wiki_faithfulness_annotator1', 'wiki_faithfulness_annotator2',
                    'slang_faithfulness_annotator1', 'slang_faithfulness_annotator2',
                    'literature_faithfulness_annotator1', 'literature_faithfulness_annotator2',
                    'baseline_faithfulness_annotator1', 'baseline_faithfulness_annotator2',
                    'human_preference_annotator1', 'human_preference_annotator2']
        df = df.merge(ann[[c for c in ann_cols if c in ann.columns]],
                      on='id', how='left')

    ordered = ['id', 'text',
               'baseline_assumption', 'baseline_label', 'baseline_raw',
               'baseline_faithfulness_annotator1', 'baseline_faithfulness_annotator2',
               'wiki_retrieval', 'wiki_assumption', 'wiki_label', 'wiki_raw',
               'wiki_faithfulness_annotator1', 'wiki_faithfulness_annotator2',
               'slang_retrieval', 'slang_assumption', 'slang_label', 'slang_raw',
               'slang_faithfulness_annotator1', 'slang_faithfulness_annotator2',
               'lit_retrieval', 'lit_assumption', 'lit_label', 'lit_raw',
               'literature_faithfulness_annotator1', 'literature_faithfulness_annotator2',
               'human_preference_annotator1', 'human_preference_annotator2']

    return df[[c for c in ordered if c in df.columns]]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True, choices=['it', 'en'])
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--ann', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data_dir = args.data_dir or args.lang
    ann = pd.read_excel(args.ann) if args.ann else None

    if args.lang == 'it':
        norag = pd.read_csv(f'{data_dir}/norag_prompt_it.csv')
        wiki = pd.read_csv(f'{data_dir}/wiki_ita_prompt_it.csv')
        slang = pd.read_csv(f'{data_dir}/slang_ita_prompt_it.csv')
        lit = pd.read_csv(f'{data_dir}/literature_ita_prompt_it.csv')
        df = build_full_results_ita(norag, wiki, slang, lit, ann,
                                    truncated_ids=[248, 429, 438, 833, 1087])
    else:
        norag = pd.read_csv(f'{data_dir}/df_norag_en.csv')
        wiki = pd.read_csv(f'{data_dir}/df_wiki_en.csv')
        slang = pd.read_csv(f'{data_dir}/df_slang_en.csv')
        lit = pd.read_csv(f'{data_dir}/df_literature_en.csv')
        df = build_full_results_eng(norag, wiki, slang, lit, ann)

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Columns: {df.columns.tolist()}")
