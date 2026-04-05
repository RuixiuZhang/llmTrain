import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="llm100m", help="HF模型目录")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="输入提示词")
    args = parser.parse_args()

    device = get_device()
    print(f"正在从 {args.model_path} 加载模型至 {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

    print("\nAI> ", end="")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
    print("\n")


if __name__ == "__main__":
    main()