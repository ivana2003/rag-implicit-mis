import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

EMBED_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

WIKTIONARY_CATEGORIES = {
    'it': {
        'base_url': 'https://en.wiktionary.org/w/api.php',
        'categories': ['Italian_slang'],
        'lang_section': '== Italian ==',
        'relevant_tags': ['slang', 'vulgar', 'derogatory', 'offensive', 'colloquial'],
        'priority_tags': ['vulgar', 'derogatory', 'offensive'],
    },
    'en': {
        'base_url': 'https://en.wiktionary.org/w/api.php',
        'categories': ['English_slang', 'English_internet_slang', 'English_incel_slang'],
        'lang_section': '== English ==',
        'relevant_tags': ['slang', 'vulgar', 'derogatory', 'offensive', 'internet'],
        'priority_tags': ['vulgar', 'derogatory', 'offensive'],
    },
}


def safe_str(x):
    if x is None:
        return ''
    s = str(x).strip()
    return '' if s == 'nan' else s


def normalize(s):
    return re.sub(r'\s+', ' ', s.lower().strip())


def get_category_members(base_url, category, pause=0.5):
    members = []
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': f'Category:{category}',
        'cmlimit': 500,
        'format': 'json',
    }
    while True:
        r = requests.get(base_url, params=params, timeout=10)
        data = r.json()
        members += [m['title'] for m in data['query']['categorymembers']]
        if 'continue' not in data:
            break
        params['cmcontinue'] = data['continue']['cmcontinue']
        time.sleep(pause)
    return members


def get_wiktionary_definitions(base_url, title, lang_section, pause=0.3):
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'revisions',
        'rvprop': 'content',
        'format': 'json',
    }
    r = requests.get(base_url, params=params, timeout=10)
    pages = r.json()['query']['pages']
    page = next(iter(pages.values()))
    if 'revisions' not in page:
        return []

    wikitext = page['revisions'][0]['*']
    if lang_section not in wikitext:
        return []

    section = wikitext[wikitext.index(lang_section):]
    next_lang = re.search(r'\n== [A-Z][a-z]+ ==', section[5:])
    if next_lang:
        section = section[:next_lang.start() + 5]

    defs = re.findall(r'#\s*(.+)', section)
    cleaned = []
    for d in defs:
        d = re.sub(r'\[\[([^\]|]+\|)?([^\]]+)\]\]', r'\2', d)
        d = re.sub(r'\{\{[^}]+\}\}', '', d)
        d = re.sub(r"'''?", '', d)
        d = d.strip()
        if len(d) > 5:
            cleaned.append(d)
    return cleaned


def pick_best_definition(definitions, priority_tags):
    for tag in priority_tags:
        for d in definitions:
            if tag.lower() in d.lower():
                return d
    return definitions[0] if definitions else ''


def build_slang_kb(base_url, categories, lang_section, relevant_tags, cache_path, pause=0.5):
    if Path(cache_path).exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    cache = {}
    for cat in categories:
        members = get_category_members(base_url, cat, pause)
        for title in tqdm(members, desc=f'Building KB ({cat})'):
            if title in cache:
                continue
            defs = get_wiktionary_definitions(base_url, title, lang_section, pause)
            relevant = [d for d in defs if any(t in d.lower() for t in relevant_tags)]
            if relevant:
                cache[title] = relevant
            time.sleep(pause)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    return cache


def extract_tokens(text, stop_words):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 2]


def build_retrieval(text, context, slang_cache, slang_set, stop_words,
                    embedder, priority_tags, cos_threshold, has_context):
    combined = (safe_str(text) + ' ' + safe_str(context)).strip() if has_context else safe_str(text)
    tokens = extract_tokens(combined, stop_words)

    candidates = {}
    for token in tokens:
        if token in slang_set:
            candidates[token] = slang_cache[token]
        for word in slang_set:
            if token in word or word in token:
                candidates[word] = slang_cache[word]

    if not candidates:
        return []

    query_emb = embedder.encode([combined])
    hits = []
    for word, defs in candidates.items():
        best_def = pick_best_definition(defs, priority_tags)
        pair = f"{word}: {best_def}"
        pair_emb = embedder.encode([pair])
        sim = cosine_similarity(query_emb, pair_emb)[0][0]
        if sim >= cos_threshold:
            hits.append({'title': word, 'summary': best_def, 'cos_ctx': float(sim)})

    hits.sort(key=lambda x: x['cos_ctx'], reverse=True)
    return hits


def compress_retrieval(text, context, retrieval_text, embedder, threshold=0.3, has_context=True):
    if not retrieval_text:
        return ''
    combined = (safe_str(text) + ' ' + safe_str(context)).strip() if has_context else safe_str(text)
    query_emb = embedder.encode([combined])
    sentences = [s.strip() for s in retrieval_text.split('|') if len(s.strip()) > 5]
    if not sentences:
        return retrieval_text
    sent_emb = embedder.encode(sentences)
    sims = cosine_similarity(query_emb, sent_emb)[0]
    filtered = [s for s, sim in zip(sentences, sims) if sim >= threshold]
    return ' | '.join(filtered) if filtered else ''


def run_pipeline(input_path, output_path, lang, cache_path,
                 cos_threshold=0.3, compress_threshold=0.3, pause=0.5):
    df = pd.read_csv(input_path)
    kb_config = WIKTIONARY_CATEGORIES[lang]
    has_context = lang == 'it'

    nlp = spacy.load('it_core_news_lg' if lang == 'it' else 'en_core_web_lg',
                     disable=['parser', 'lemmatizer', 'ner'])
    stop_words = set(w.lower() for w in nlp.Defaults.stop_words)
    embedder = SentenceTransformer(EMBED_MODEL)

    slang_cache = build_slang_kb(
        kb_config['base_url'], kb_config['categories'],
        kb_config['lang_section'], kb_config['relevant_tags'],
        cache_path, pause
    )
    slang_set = set(w.lower() for w in slang_cache.keys())

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        text = safe_str(row.get('text', ''))
        context = safe_str(row.get('Contesto', '')) if has_context else ''

        hits = build_retrieval(
            text, context, slang_cache, slang_set, stop_words,
            embedder, kb_config['priority_tags'], cos_threshold, has_context
        )

        retrieval_text = '\n\n'.join([f"### {h['title']}\n{h['summary']}" for h in hits])
        if len(retrieval_text) > 4500:
            retrieval_text = retrieval_text[:4500] + '...'

        compressed = compress_retrieval(
            text, context, retrieval_text, embedder, compress_threshold, has_context
        )

        results.append({
            'id': row.get('id'),
            'text': text,
            'Contesto': context,
            'slang_compressed': compressed if compressed else None,
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} rows to {output_path}")
    print(f"Rows with slang: {df_out['slang_compressed'].notna().sum()}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lang', required=True, choices=['it', 'en'])
    parser.add_argument('--cache', required=True)
    parser.add_argument('--cos-threshold', type=float, default=0.3)
    parser.add_argument('--compress-threshold', type=float, default=0.3)
    args = parser.parse_args()

    run_pipeline(
        args.input, args.output, args.lang, args.cache,
        args.cos_threshold, args.compress_threshold
    )
