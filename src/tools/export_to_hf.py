import torch, os, argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", default="models/bert_sentiment")
    p.add_argument("--base", default="bert-base-uncased")
    p.add_argument("--num_labels", type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=args.num_labels)

    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    tok.save_pretrained(args.outdir)
    model.save_pretrained(args.outdir)
    print(f"Saved HF model to {args.outdir}")

if __name__ == "__main__":
    main()
