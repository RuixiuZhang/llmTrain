"""
recurrent_transformer_gpt2.py

Recurrent-attention building blocks for the adaptive-recurrent (ACT /
Universal-Transformer) language model. act_model_v2.py imports the
config, RecurrentAttention, FeedForward, RoPE/apply_rotary helpers, and
device setup from this module. The standalone trainer / data pipeline /
generation code at the bottom is the original Transformer-XL training
script and is now legacy.

Target hardware: single RTX PRO 6000 Blackwell GPU on Linux.
Uses the GPT-2 tokenizer + Transformer-XL style recurrent KV memory.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import csv
import glob
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# CUDA global optimizations
# ─────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
class Config:
    VOCAB_SIZE    = 50257
    EMBED_DIM     = 512
    HEADS         = 8
    LAYERS        = 6
    FF_DIM        = 2048
    SEG_LEN       = 128
    MEM_LEN       = 256
    SEGS_PER_SEQ  = 4
    MAX_LENGTH    = SEG_LEN * SEGS_PER_SEQ
    DROPOUT       = 0.1

    BATCH_SIZE    = 6          # conservative setting for limited VRAM
    GRAD_ACCUM    = 8          # effective batch=48
    MAX_STEPS     = 72000
    LR            = 5e-5       # more stable with recurrent + GPT-2 vocab
    MIN_LR        = 5e-6
    WARMUP_STEPS  = 2000
    WEIGHT_DECAY  = 0.1
    GRAD_CLIP     = 0.5
    GRAD_SKIP     = 100.0      # skip the update when gnorm exceeds this value

    LOG_INTERVAL  = 100
    SAVE_INTERVAL = 500
    EVAL_INTERVAL = 500
    EVAL_ITERS    = 30

    NUM_WORKERS   = 0          # multiprocessing dataloader disabled, set to 0
    PIN_MEMORY    = False
    DTYPE         = torch.bfloat16
    USE_COMPILE   = False      # torch.compile disabled
    MEM_FRACTION  = 0.85

    TEMPERATURE   = 0.8
    TOP_K         = 40
    TOP_P         = 0.92
    GENERATE_LEN  = 256

    DATA_PATH     = "data/raw/input.txt"
    PARQUET_GLOB  = "data/raw/train-*.parquet"
    TOKENS_PATH   = "data/processed/gpt2_tokens.pt"
    TOKENIZER_NAME = "gpt2"
    TOKENIZER_DIR  = "data/tokenizer/gpt2"
    HF_CACHE_DIR   = "data/hf_cache"
    MAX_TOKENIZE   = 4096
    MODEL_PATH    = "results/recurrent_transformer_final.pt"
    CKPT_DIR      = "results/checkpoints"
    SAMPLE_DIR    = "results/samples"
    METRICS_PATH  = "results/metrics.csv"
    LATEST_CKPT   = "results/checkpoints/latest.pt"
    BEST_CKPT     = "results/checkpoints/best.pt"
    RESUME        = True

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = torch.cuda.mem_get_info()[0] / 1e9
            print(f"✅ GPU : {name}")
            print(f"   total VRAM: {mem:.1f} GB  |  available: {free:.1f} GB")
            print(f"   CUDA : {torch.version.cuda}")
            return torch.device("cuda")
        print("⚠️  CUDA not detected, using CPU")
        return torch.device("cpu")


cfg    = Config()
DEVICE = cfg.get_device()
# NOTE: the legacy standalone trainer created results/ and data/ directories here on
# import. Removed for the published repo so importing these building blocks for
# inference does not create stray directories in the working tree.

if DEVICE.type == "cuda":
    if cfg.DTYPE == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        cfg.DTYPE = torch.float16
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(cfg.MEM_FRACTION)
else:
    cfg.DTYPE = torch.float32


# ─────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin, mem_len):
    q_cos = cos[mem_len:][None, None, :, :]
    q_sin = sin[mem_len:][None, None, :, :]
    k_cos = cos[None, None, :, :]
    k_sin = sin[None, None, :, :]
    return q * q_cos + rotate_half(q) * q_sin, k * k_cos + rotate_half(k) * k_sin


# ─────────────────────────────────────────────
# Recurrent Flash Attention
# ─────────────────────────────────────────────
class RecurrentAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = cfg.EMBED_DIM // cfg.HEADS
        self.n_heads  = cfg.HEADS
        self.qkv  = nn.Linear(cfg.EMBED_DIM, 3 * cfg.EMBED_DIM, bias=False)
        self.proj = nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.qknorm = int(os.environ.get("RT_QKNORM", "0"))     # GATED: RMSNorm q/k before RoPE
        if self.qknorm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, mem_kv=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if self.qknorm:                                         # normalize q and NEW k before RoPE & mem cat
            q = self.q_norm(q)
            k = self.k_norm(k)                                  # mem k was already normalized when written

        # Fixed-size (MEM_LEN) memory so torch.compile sees ONE static shape every
        # segment. Unfilled slots are masked out, so this is numerically identical
        # to a growing memory: RoPE depends only on relative q-k distance, and a
        # constant position shift + masked padding leaves attention scores unchanged.
        M = cfg.MEM_LEN
        if mem_kv is None:
            mk = k.new_zeros(B, self.n_heads, M, self.head_dim)
            mv = v.new_zeros(B, self.n_heads, M, self.head_dim)
            mvalid = torch.zeros(M, dtype=torch.bool, device=x.device)
        else:
            mk, mv, mvalid = mem_kv

        raw_k  = torch.cat([mk, k], dim=2)          # pre-RoPE keys, (B, H, M+T, hd)
        v_full = torch.cat([mv, v], dim=2)
        total_len = M + T
        cos, sin = self.rope(total_len, x.device)
        q, k_rot = apply_rotary(q, raw_k, cos, sin, M)

        # query row i sees: filled memory slots (mvalid) + current tokens j <= i
        col_ok = torch.cat([mvalid, torch.ones(T, dtype=torch.bool, device=x.device)])
        allow  = torch.ones(T, total_len, dtype=torch.bool, device=x.device)
        allow[:, M:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        allow &= col_ok[None, :]
        attn_mask = torch.zeros(T, total_len, dtype=q.dtype, device=x.device)
        attn_mask.masked_fill_(~allow, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k_rot, v_full,
            attn_mask = attn_mask,
            dropout_p = cfg.DROPOUT if self.training else 0.0,
        )
        out = self.proj(out.transpose(1, 2).reshape(B, T, C))
        new_valid = torch.cat([mvalid, torch.ones(T, dtype=torch.bool, device=x.device)])
        new_mem = (
            raw_k[:, :, -M:, :].detach(),
            v_full[:, :, -M:, :].detach(),
            new_valid[-M:].detach(),
        )
        return out, new_mem


# ─────────────────────────────────────────────
# SwiGLU FFN
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused = int(os.environ.get("RT_FUSED_FFN", "0"))   # fuse gate(w1)+up(w3) into one D->2FF GEMM
        if self.fused:
            self.w13 = nn.Linear(cfg.EMBED_DIM, 2 * cfg.FF_DIM, bias=False)
        else:
            self.w1 = nn.Linear(cfg.EMBED_DIM, cfg.FF_DIM, bias=False)
            self.w3 = nn.Linear(cfg.EMBED_DIM, cfg.FF_DIM, bias=False)
        self.w2   = nn.Linear(cfg.FF_DIM,    cfg.EMBED_DIM, bias=False)
        self.drop = nn.Dropout(cfg.DROPOUT)

    def forward(self, x):
        if self.fused:
            g, u = self.w13(x).chunk(2, dim=-1)                 # one matmul, then split gate/up
            return self.drop(self.w2(F.silu(g) * u))
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.EMBED_DIM)
        self.norm2 = nn.LayerNorm(cfg.EMBED_DIM)
        self.attn  = RecurrentAttention()
        self.ff    = FeedForward()

    def forward(self, x, mem_kv=None):
        attn, new_mem = self.attn(self.norm1(x), mem_kv)
        x = x + attn
        x = x + self.ff(self.norm2(x))
        return x, new_mem


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class RecurrentGPT2LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.VOCAB_SIZE, cfg.EMBED_DIM)
        self.drop    = nn.Dropout(cfg.DROPOUT)
        self.blocks  = nn.ModuleList([TransformerBlock() for _ in range(cfg.LAYERS)])
        self.norm    = nn.LayerNorm(cfg.EMBED_DIM)
        self.head    = nn.Linear(cfg.EMBED_DIM, cfg.VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def init_mems(self):
        return [None] * cfg.LAYERS

    def forward(self, idx, targets=None, mems=None):
        if mems is None:
            mems = self.init_mems()

        x = self.drop(self.tok_emb(idx))
        new_mems = []
        for block, mem_kv in zip(self.blocks, mems):
            x, new_mem = block(x, mem_kv)
            new_mems.append(new_mem)

        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, cfg.VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=-1
            )
        return logits, loss, new_mems

    def count_params(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"📊 params: {n/1e6:.2f}M  |  BF16 VRAM estimate: {n*2/1e9:.2f} GB")


# ─────────────────────────────────────────────
# Data preprocessing
# ─────────────────────────────────────────────
def load_tokenizer():
    from transformers import AutoTokenizer

    local_files = [
        os.path.join(cfg.TOKENIZER_DIR, "tokenizer_config.json"),
        os.path.join(cfg.TOKENIZER_DIR, "vocab.json"),
        os.path.join(cfg.TOKENIZER_DIR, "merges.txt"),
    ]
    local_ready = all(os.path.exists(path) for path in local_files)
    source = cfg.TOKENIZER_DIR if local_ready else cfg.TOKENIZER_NAME

    tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=cfg.HF_CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    if not local_ready:
        tokenizer.save_pretrained(cfg.TOKENIZER_DIR)
        print(f"✅ GPT-2 tokenizer saved to working directory → {cfg.TOKENIZER_DIR}")
    else:
        print(f"📂 GPT-2 tokenizer loaded from working directory → {cfg.TOKENIZER_DIR}")

    return tokenizer


def build_tokens(tokenizer):
    if os.path.exists(cfg.TOKENS_PATH):
        return

    samples = []
    block = cfg.MAX_LENGTH + 1
    parquet_files = sorted(glob.glob(cfg.PARQUET_GLOB))

    if parquet_files:
        from datasets import load_dataset
        from tqdm import tqdm

        print(f"📦 reading parquet data: {len(parquet_files)} files")
        for path in parquet_files:
            print(f"   {path}")

        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
        print(f"   samples: {len(dataset):,}")

        buffer = []
        for item in tqdm(dataset, desc="GPT-2 Tokenizing"):
            text = item.get("text", "")
            if not text or not text.strip():
                continue

            ids = tokenizer(
                text,
                truncation=True,
                max_length=cfg.MAX_TOKENIZE,
                add_special_tokens=False,
            )["input_ids"]
            ids.append(tokenizer.eos_token_id)
            buffer.extend(ids)

            while len(buffer) >= block:
                samples.append(buffer[:block])
                buffer = buffer[cfg.MAX_LENGTH:]

    else:
        print(f"📖 reading raw text: {cfg.DATA_PATH}")
        with open(cfg.DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        ids.append(tokenizer.eos_token_id)

        for i in range(0, len(ids) - block + 1, cfg.MAX_LENGTH):
            samples.append(ids[i:i + block])

    if len(samples) == 0:
        raise ValueError("too little data, cannot slice out training samples")

    tokens = torch.tensor(samples, dtype=torch.long)
    torch.save(tokens, cfg.TOKENS_PATH)
    print(f"✅ tokens saved → {cfg.TOKENS_PATH}")
    print(f"   shape: {tuple(tokens.shape)}  |  memory: {tokens.nbytes/1e9:.2f} GB")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class PreTokenizedDataset(Dataset):
    def __init__(self, path=cfg.TOKENS_PATH, split="train"):
        print(f"📂 loading: {path} ({split})")
        tokens = torch.load(path, map_location="cpu", weights_only=True)
        cut = int(0.9 * len(tokens))
        self.tokens = tokens[:cut] if split == "train" else tokens[cut:]
        print(f"   samples: {len(self.tokens):,}  |  "
              f"seq len: {self.tokens.shape[1]}  |  "
              f"memory: {self.tokens.nbytes/1e9:.2f} GB")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        s = self.tokens[i]
        return s[:-1], s[1:]


# ─────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────
def get_lr(step, total_steps):
    if step < cfg.WARMUP_STEPS:
        return cfg.LR * (step + 1) / max(1, cfg.WARMUP_STEPS)
    p = (step - cfg.WARMUP_STEPS) / max(1, total_steps - cfg.WARMUP_STEPS)
    p = min(max(p, 0.0), 1.0)
    return cfg.MIN_LR + (cfg.LR - cfg.MIN_LR) * 0.5 * (1 + math.cos(math.pi * p))


# ─────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────
def save_ckpt(model, optimizer, step, loss, path):
    torch.save({
        "step"      : step,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "loss"      : loss,
    }, path)
    print(f"  💾 → {path}")


def load_ckpt(model, optimizer):
    if not cfg.RESUME or not os.path.exists(cfg.LATEST_CKPT):
        return 0, float("inf")

    print(f"\n📂 loading checkpoint: {cfg.LATEST_CKPT}")
    ckpt = torch.load(cfg.LATEST_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", float("inf"))
    print(f"   ✅ resuming training from step {step}  |  loss {loss:.4f}")
    return step, loss


# ─────────────────────────────────────────────
# Eval
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, autocast):
    model.eval()
    losses = []
    iterator = iter(dataloader)
    for _ in range(cfg.EVAL_ITERS):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            x, y = next(iterator)

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        mems = model.init_mems()
        with autocast:
            for start in range(0, cfg.MAX_LENGTH, cfg.SEG_LEN):
                xs = x[:, start:start + cfg.SEG_LEN]
                ys = y[:, start:start + cfg.SEG_LEN]
                _, loss, mems = model(xs, ys, mems)
                losses.append(loss.item())

    model.train()
    return sum(losses) / max(1, len(losses))


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.95),
        fused=(DEVICE.type == "cuda")
    )
    autocast = torch.amp.autocast(
        device_type=DEVICE.type,
        dtype=cfg.DTYPE,
        enabled=(DEVICE.type == "cuda" and cfg.DTYPE != torch.float32)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and cfg.DTYPE == torch.float16))

    gstep, best_loss = load_ckpt(model, optimizer)
    total_steps = cfg.MAX_STEPS
    iterator = iter(train_loader)

    if not os.path.exists(cfg.METRICS_PATH):
        with open(cfg.METRICS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "loss", "val_loss", "lr", "gnorm", "tok_s"])

    print(f"\n🚀 training started")
    print(f"   total steps: {total_steps:,}  |  start step: {gstep:,}")
    print(f"   effective batch: {cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
    print(f"   GPT-2 tokenizer + BF16/FP16 + recurrent KV memory\n")

    total_loss = 0.0
    tok_count  = 0
    t_log      = time.time()

    try:
        while gstep < total_steps:
            optimizer.zero_grad(set_to_none=True)
            step_loss_sum = 0.0
            step_loss_cnt = 0

            for _ in range(cfg.GRAD_ACCUM):
                try:
                    x, y = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    x, y = next(iterator)

                x = x.to(DEVICE)
                y = y.to(DEVICE)
                mems = model.init_mems()

                for start in range(0, cfg.MAX_LENGTH, cfg.SEG_LEN):
                    xs = x[:, start:start + cfg.SEG_LEN]
                    ys = y[:, start:start + cfg.SEG_LEN]

                    with autocast:
                        _, raw_loss, mems = model(xs, ys, mems)
                        loss = raw_loss / (cfg.GRAD_ACCUM * cfg.SEGS_PER_SEQ)

                    scaler.scale(loss).backward()
                    step_loss_sum += raw_loss.item()
                    step_loss_cnt += 1
                    tok_count += xs.numel()

            lr = get_lr(gstep, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            gnorm = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)

            if (not math.isfinite(gnorm)) or gnorm > cfg.GRAD_SKIP:
                print(f"  Step {gstep+1:6d} | skip update: gnorm {gnorm:.2f} > {cfg.GRAD_SKIP:.1f}")
                optimizer.zero_grad(set_to_none=True)
            else:
                scaler.step(optimizer)
            scaler.update()
            gstep += 1

            step_loss = step_loss_sum / max(1, step_loss_cnt)
            total_loss += step_loss

            if gstep % cfg.LOG_INTERVAL == 0:
                dt    = max(time.time() - t_log, 1e-6)
                avg   = total_loss / cfg.LOG_INTERVAL
                tok_s = tok_count / dt
                free  = torch.cuda.mem_get_info()[0] / 1e9 if DEVICE.type == "cuda" else 0.0
                print(f"  Step {gstep:6d}/{total_steps} | Loss {avg:.4f} | PPL {math.exp(min(avg, 20)):6.1f} | "
                      f"LR {lr:.2e} | gnorm {gnorm:.2f} | {tok_s/1000:.1f}k tok/s | free VRAM {free:.1f}GB")
                total_loss = 0.0
                tok_count  = 0
                t_log      = time.time()

            if gstep % cfg.EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_loader, autocast)
                print(f"  Eval {gstep:6d} | Val Loss {val_loss:.4f} | Val PPL {math.exp(min(val_loss, 20)):.1f}")
                with open(cfg.METRICS_PATH, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([gstep, step_loss, val_loss, lr, round(gnorm, 4), round(tok_s, 2)])
                save_ckpt(model, optimizer, gstep, val_loss, cfg.LATEST_CKPT)
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_ckpt(model, optimizer, gstep, val_loss, cfg.BEST_CKPT)

            if gstep % cfg.SAVE_INTERVAL == 0:
                save_ckpt(model, optimizer, gstep, step_loss, cfg.LATEST_CKPT)
                save_ckpt(model, optimizer, gstep, step_loss, os.path.join(cfg.CKPT_DIR, f"ckpt_step{gstep}.pt"))

    except KeyboardInterrupt:
        print("\n⚠️  interrupt detected, saving latest checkpoint then exiting")
        save_ckpt(model, optimizer, gstep, step_loss if "step_loss" in locals() else best_loss, cfg.LATEST_CKPT)
        raise

    torch.save(model.state_dict(), cfg.MODEL_PATH)
    print(f"✅ saved → {cfg.MODEL_PATH}")


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=None):
    max_new = max_new or cfg.GENERATE_LEN
    model.eval()
    x = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)
    mems = model.init_mems()

    with torch.amp.autocast(
        device_type=DEVICE.type,
        dtype=cfg.DTYPE,
        enabled=(DEVICE.type == "cuda" and cfg.DTYPE != torch.float32)
    ):
        if x.size(1) > 1:
            for start in range(0, x.size(1) - 1, cfg.SEG_LEN):
                chunk = x[:, start:min(start + cfg.SEG_LEN, x.size(1) - 1)]
                if chunk.numel() > 0:
                    _, _, mems = model(chunk, mems=mems)

        cur = x[:, -1:]
        for _ in range(max_new):
            logits, _, mems = model(cur, mems=mems)
            logits = logits[:, -1, :] / cfg.TEMPERATURE

            if cfg.TOP_K > 0:
                logits[logits < torch.topk(logits, cfg.TOP_K).values[:, -1, None]] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            sp[torch.cumsum(sp, -1) - sp > cfg.TOP_P] = 0
            sp    /= sp.sum(-1, keepdim=True)
            cur    = si.gather(-1, torch.multinomial(sp, 1))
            x      = torch.cat([x, cur], dim=-1)

    return tokenizer.decode(x[0].tolist())


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    tokenizer = load_tokenizer()
    build_tokens(tokenizer)

    model = RecurrentGPT2LM().to(DEVICE)
    model.count_params()

    train_set = PreTokenizedDataset(split="train")
    val_set   = PreTokenizedDataset(split="val")
    train_loader = DataLoader(
        train_set,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = True,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = cfg.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = cfg.PIN_MEMORY,
    )

    train(model, train_loader, val_loader)

    prompt = "The future of artificial intelligence"
    print(f"\n🎯 {prompt}\n" + "─"*50)
    text = generate(model, tokenizer, prompt)
    out_path = os.path.join(cfg.SAMPLE_DIR, "generated.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n✅ generation saved → {out_path}")


if __name__ == "__main__":
    main()
