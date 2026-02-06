from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_DIR = "clinical_nlp_model"
MAX_LENGTH = 256


def load_artifacts(model_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model, max_length: int = MAX_LENGTH, top_k: int = 2) -> dict:
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
    top_probs = top_probs[0].tolist()
    top_ids = top_ids[0].tolist()

    id2label = getattr(model.config, "id2label", None)
    if not id2label or not isinstance(id2label, dict) or len(id2label) == 0:
        id2label = {0: "BACKGROUND", 1: "OBJECTIVE", 2: "METHODS", 3: "RESULTS", 4: "CONCLUSIONS"}

    top_labels = [id2label.get(i, str(i)) for i in top_ids]

    pred_label = top_labels[0]
    confidence = top_probs[0]

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "top_k": list(zip(top_labels, top_probs)),
    }


def main():
    tokenizer, model = load_artifacts()

    examples = [
        "Cardiovascular disease remains a leading cause of mortality worldwide.",
        "We conducted a randomized controlled trial with 120 patients and measured outcomes at 12 weeks.",
        "The treatment group showed a statistically significant reduction in systolic blood pressure.",
        "In conclusion, the intervention improved clinical outcomes with minimal adverse events."
    ]

    print("\n=== Running example predictions ===\n")
    for i, text in enumerate(examples, start=1):
        result = predict(text, tokenizer, model, top_k=2)
        print(f"Example {i}: {text}")
        print(f"  Prediction: {result['prediction']}  |  Confidence: {result['confidence']:.3f}")
        print(f"  Top-2: {', '.join([f'{lbl} ({p:.3f})' for lbl, p in result['top_k']])}")
        print()

    print("=== Interactive mode ===")
    print("Paste a biomedical sentence (or type 'q' to quit).\n")

    while True:
        user_text = input("> ").strip()
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        if not user_text:
            continue

        result = predict(user_text, tokenizer, model, top_k=3)
        print(f"Prediction: {result['prediction']}  |  Confidence: {result['confidence']:.3f}")
        print("Top-3:")
        for lbl, p in result["top_k"]:
            print(f"  - {lbl}: {p:.3f}")
        print()


if __name__ == "__main__":
    main()
