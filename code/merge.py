import pandas as pd
from pathlib import Path


def load_ita(data_dir='ita'):
    wiki = pd.read_csv(f'{data_dir}/wiki_ita_prompt_it.csv')
    slang = pd.read_csv(f'{data_dir}/slang_ita_prompt_it.csv')
    lit = pd.read_csv(f'{data_dir}/literature_ita_prompt_it.csv')
    norag = pd.read_csv(f'{data_dir}/norag_prompt_it.csv')
    return wiki, slang, lit, norag


def load_eng(data_dir='eng'):
    wiki = pd.read_csv(f'{data_dir}/df_wiki_en.csv')
    slang = pd.read_csv(f'{data_dir}/df_slang_en.csv')
    lit = pd.read_csv(f'{data_dir}/df_literature_en.csv')
    norag = pd.read_csv(f'{data_dir}/df_norag_en.csv')
    return wiki, slang, lit, norag


def merge_ita(wiki, slang, lit, norag):
    # fix id mancante in slang
    if 1092 not in slang['id'].values:
        missing = wiki[wiki['id'] == 1092][
            ['id', 'text', 'Contesto', 'raw_RAG', 'raw_NORAG',
             'label_RAG', 'assumption_RAG', 'label_NORAG', 'assumption_NORAG']
        ].copy()
        missing['slang_compressed'] = float('nan')
        slang = pd.concat([slang, missing[slang.columns]], ignore_index=True)

    wiki_clean = wiki[['id', 'text', 'Contesto', 'retrieval_text_RAG', 'raw_RAG']].rename(columns={
        'retrieval_text_RAG': 'retrieval_wiki',
        'raw_RAG': 'raw_RAG_wiki',
    })
    slang_clean = slang[['id', 'slang_compressed', 'raw_RAG']].rename(columns={
        'slang_compressed': 'retrieval_slang',
        'raw_RAG': 'raw_RAG_slang',
    })
    lit_clean = lit[['id', 'lit_compressed', 'raw_output']].rename(columns={
        'lit_compressed': 'retrieval_literature',
        'raw_output': 'raw_RAG_literature',
    })
    norag_clean = norag[['id', 'raw_NORAG']]

    df = (wiki_clean
          .merge(slang_clean, on='id', how='left')
          .merge(lit_clean, on='id', how='left')
          .merge(norag_clean, on='id', how='left'))

    return df[['id', 'text', 'Contesto',
               'retrieval_wiki', 'raw_RAG_wiki',
               'retrieval_slang', 'raw_RAG_slang',
               'retrieval_literature', 'raw_RAG_literature',
               'raw_NORAG']]


def merge_eng(wiki, slang, lit, norag):
    wiki['id'] = wiki['id'].astype(int)

    wiki_clean = wiki[['id', 'text', 'retrieval_text_RAG', 'raw_RAG', 'raw_NORAG']].rename(columns={
        'retrieval_text_RAG': 'retrieval_wiki',
        'raw_RAG': 'raw_RAG_wiki',
    })
    slang_clean = slang[['id', 'slang_compressed', 'raw_RAG']].rename(columns={
        'slang_compressed': 'retrieval_slang',
        'raw_RAG': 'raw_RAG_slang',
    })
    lit_clean = lit[['id', 'lit_compressed', 'raw_output']].rename(columns={
        'lit_compressed': 'retrieval_literature',
        'raw_output': 'raw_RAG_literature',
    })

    df = (wiki_clean
          .merge(slang_clean, on='id', how='left')
          .merge(lit_clean, on='id', how='left'))

    return df[['id', 'text',
               'retrieval_wiki', 'raw_RAG_wiki',
               'retrieval_slang', 'raw_RAG_slang',
               'retrieval_literature', 'raw_RAG_literature',
               'raw_NORAG']]


def sample_100(df, lang, seed=42):
    ret_cols = ['retrieval_wiki', 'retrieval_slang', 'retrieval_literature']
    df = df.copy()
    df['_n_ret'] = df[ret_cols].notna().sum(axis=1)

    priority = []
    for n in [3, 2, 1]:
        sub = df[df['_n_ret'] == n]
        priority.append(sub)

    df_sorted = pd.concat(priority).drop(columns=['_n_ret'])
    return df_sorted.head(100).reset_index(drop=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', required=True, choices=['it', 'en'])
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()

    data_dir = args.data_dir or args.lang

    if args.lang == 'it':
        wiki, slang, lit, norag = load_ita(data_dir)
        df = merge_ita(wiki, slang, lit, norag)
    else:
        wiki, slang, lit, norag = load_eng(data_dir)
        df = merge_eng(wiki, slang, lit, norag)

    if args.sample:
        df = sample_100(df, args.lang)

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Columns: {df.columns.tolist()}")
