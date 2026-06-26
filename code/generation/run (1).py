import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def safe_str(x):
    # Returns empty string for None or NaN values
    return "" if (x is None or str(x).strip() == "nan") else str(x)


def build_prompt_fewshot_norag(text, contesto):
    # Few-shot prompt without retrieval: 3 examples + target text
    # The model must identify the implied assumption and classify as misogynous (SÌ/NO)
    prompt = f"""Quali sono gli assunti impliciti del testo? Sulla base dei presupposti impliciti, il testo è misogino? Rispondere solo con SÌ o NO. Restituisci SOLO JSON valido. Seguire l'esempio.
Testo: non dovremmo abbassare i nostri standard solo per assumere più donne.
Presupposto: le donne sono meno qualificate degli uomini.
Risposta: SÌ.
Testo: le donne che parlano di calcio non si possono sentire.
Presupposto: le donne non capiscono niente di calcio, è una cosa da maschi.
Risposta: SÌ.
Testo: Dopo aver visto (per sbaglio) 5 minuti di Uomini e Donne ho deciso che odierò le donne.
Presupposto: Le donne che vanno alla trasmissione Uomini e Donne sono stupide.
Risposta: SÌ.
Testo: {text}
Contesto: {contesto}
Output JSON:
{{
  "testo": "{text}",
  "assunto_implicito": "scrivi qui",
  "risposta": "SÌ | NO"
}}"""
    return prompt


def build_prompt_fewshot_rag(text, contesto, retrieved):
    # Few-shot prompt with retrieval: same as no-RAG but includes retrieved knowledge
    # The retrieved context comes from one of the three knowledge bases (wiki, slang, literature)
    prompt = f"""Quali sono gli assunti impliciti del testo? Sulla base dei presupposti impliciti, il testo è misogino? Rispondere solo con SÌ o NO. Restituisci SOLO JSON valido. Considera anche la conoscenza fornita se presente. Seguire l'esempio.
Testo: non dovremmo abbassare i nostri standard solo per assumere più donne.
Presupposto: le donne sono meno qualificate degli uomini.
Risposta: SÌ.
Testo: le donne che parlano di calcio non si possono sentire.
Presupposto: le donne non capiscono niente di calcio, è una cosa da maschi.
Risposta: SÌ.
Testo: Dopo aver visto (per sbaglio) 5 minuti di Uomini e Donne ho deciso che odierò le donne.
Presupposto: Le donne che vanno alla trasmissione Uomini e Donne sono stupide.
Risposta: SÌ.
Testo: {text}
Contesto: {contesto}
Conoscenza: {retrieved}
Output JSON:
{{
  "testo": "{text}",
  "assunto_implicito": "scrivi qui",
  "risposta": "SÌ | NO"
}}"""
    return prompt


def build_prompt_zeroshot_norag(text, contesto):
    # Zero-shot prompt without retrieval: no examples, just the instruction
    prompt = f"""Quali sono gli assunti impliciti del testo? Sulla base dei presupposti impliciti, il testo è misogino? Rispondere solo con SÌ o NO. Restituisci SOLO JSON valido.
Testo: {text}
Contesto: {contesto}
Output JSON:
{{
  "testo": "{text}",
  "assunto_implicito": "scrivi qui",
  "risposta": "SÌ | NO"
}}"""
    return prompt


def build_prompt_zeroshot_rag(text, contesto, retrieved):
    # Zero-shot prompt with retrieval: instruction + retrieved knowledge, no examples
    prompt = f"""Quali sono gli assunti impliciti del testo? Sulla base dei presupposti impliciti, il testo è misogino? Rispondere solo con SÌ o NO. Restituisci SOLO JSON valido. Considera anche la conoscenza fornita se presente.
Testo: {text}
Contesto: {contesto}
Conoscenza: {retrieved}
Output JSON:
{{
  "testo": "{text}",
  "assunto_implicito": "scrivi qui",
  "risposta": "SÌ | NO"
}}"""
    return prompt


def call_model(model, tokenizer, prompt, temperature, max_tokens):
    # Applies chat template, runs inference, and returns the generated text
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens (skip the input)
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def load_model(model_id, dtype):
    # Loads tokenizer and model with automatic device mapping (GPU if available)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch_dtype, trust_remote_code=True
    )
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(
        description="Run LLM inference for implicit misogyny detection. "
                    "For each row, generates both a RAG response (using retrieved knowledge) "
                    "and a no-RAG baseline response. Results are written row-by-row to support "
                    "resuming interrupted jobs via --batch-start."
    )
    ap.add_argument("--input-csv", required=True, help="Input CSV with text, context and retrieval columns")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--mode", required=True, choices=["fewshot", "zeroshot"], help="Prompting strategy")
    ap.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"], help="Model precision")
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
    ap.add_argument("--batch-start", type=int, default=0, help="Row index to start from (for resuming)")
    ap.add_argument("--batch-end", type=int, default=None, help="Row index to end at (exclusive)")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv, sep=",")
    print(f"Columns: {df.columns.tolist()}, Rows: {len(df)}")

    model, tokenizer = load_model(args.model_id, args.dtype)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = args.batch_start
    end = args.batch_end if args.batch_end else len(df)

    # Append mode if resuming from a mid-run checkpoint, else write fresh
    mode = "a" if args.batch_start > 0 else "w"
    cols = ["id", "text", "Contesto", "retrieval_compressed", "raw_RAG", "raw_NORAG"]
    with out_path.open(mode, newline="", encoding="utf-8") as f:
        if args.batch_start == 0:
            csv.writer(f).writerow(cols)

    for idx in tqdm(range(start, end), desc="Processing"):
        row = df.iloc[idx]
        rid = row["id"]
        text = safe_str(row["text"])
        contesto = safe_str(row.get("Contesto", ""))
        retrieved = safe_str(row.get("retrieval_compressed", ""))

        # Build prompts for both RAG and no-RAG conditions
        if args.mode == "fewshot":
            prompt_rag = build_prompt_fewshot_rag(text, contesto, retrieved)
            prompt_no = build_prompt_fewshot_norag(text, contesto)
        else:
            prompt_rag = build_prompt_zeroshot_rag(text, contesto, retrieved)
            prompt_no = build_prompt_zeroshot_norag(text, contesto)

        raw_rag = call_model(model, tokenizer, prompt_rag, args.temperature, args.max_new_tokens)
        raw_no = call_model(model, tokenizer, prompt_no, args.temperature, args.max_new_tokens)

        # Write each row immediately so partial results are saved if the job is interrupted
        with out_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([rid, text, contesto, retrieved, raw_rag, raw_no])

    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
