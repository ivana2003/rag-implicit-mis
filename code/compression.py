import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

NOISE_SECTIONS = [
    '== Note ==', '== Bibliografia ==', '== Collegamenti esterni ==',
    '== Voci correlate ==', '== Altri progetti ==', '== Pagine correlate ==',
    '=== Note ===', '=== Bibliografia ===', '== Classifiche ==',
]


def split_passages(text):
    if pd.isna(text):
        return []
    return [p.strip() for p in text.split('###') if p.strip()]


def clean_passage(passage):
    for section in NOISE_SECTIONS:
        if section in passage:
            passage = passage[:passage.index(section)]
    return passage.strip()


def split_sentences(passage):
    passage = re.sub(r'==+[^=]+=+\n?', '', passage)
    passage = re.sub(r'\n+', ' ', passage).strip()
    sentences = re.split(r'(?<=[.!?])\s+', passage)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def compress_retrieval(text, context, passages, threshold=0.3):
    if not passages:
        return None

    query = (text + ' ' + context).strip()
    query_emb = model.encode([query])

    result = []
    for passage in passages:
        passage = clean_passage(passage)
        sentences = split_sentences(passage)
        if not sentences:
            continue

        sent_emb = model.encode(sentences)
        sims = cosine_similarity(query_emb, sent_emb)[0]

        for sent, sim in zip(sentences, sims):
            if sim >= threshold:
                result.append(sent)

    return ' '.join(result) if result else None


def compress_dataframe(df, text_col='text', context_col='Contesto',
                       retrieval_col='retrieval_text_RAG',
                       output_col='retrieval_compressed',
                       threshold=0.3):
    df = df.copy()
    df['_passages'] = df[retrieval_col].apply(split_passages)

    tqdm.pandas(desc="Compressing")
    df[output_col] = df.progress_apply(
        lambda row: compress_retrieval(
            row[text_col],
            row.get(context_col, ''),
            row['_passages'],
            threshold=threshold
        ),
        axis=1
    )
    df.drop(columns=['_passages'], inplace=True)
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--text-col', default='text')
    parser.add_argument('--context-col', default='Contesto')
    parser.add_argument('--retrieval-col', default='retrieval_text_RAG')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--sep', default=',')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)
    print(f"Loaded {len(df)} rows")

    df = compress_dataframe(
        df,
        text_col=args.text_col,
        context_col=args.context_col,
        retrieval_col=args.retrieval_col,
        threshold=args.threshold
    )

    print(f"Rows with retrieval:   {df[args.retrieval_col].notna().sum()}")
    print(f"Rows with compressed:  {df['retrieval_compressed'].notna().sum()}")
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
