import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.set_float32_matmul_precision("high")

class CFG:
    vocab   = 50257
    dim     = 768
    heads   = 12
    layers  = 10
    ff      = 2048
    seq     = 512
    batch   = 64
    acc     = 4
    lr      = 5e-5
    warmup  = 1000
    wd      = 0.01
    clip    = 0.5
    steps   = 5000
    workers = 4
    tokens  = "train_tokens.pt"
    model   = "model.pt"
    dtype   = torch.bfloat16

cfg    = CFG()
device = "cuda"

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t     = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    cos = cos[None, None, :, :]   # [1, 1, T, head_dim]
    sin = sin[None, None, :, :]
    q   = q * cos + rotate_half(q) * sin
    k   = k * cos + rotate_half(k) * sin
    return q, k

# Flash Attention via SDPA
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert cfg.dim % cfg.heads == 0, "dim 必须能被 heads 整除"
        self.head_dim = cfg.dim // cfg.heads
        self.n_heads  = cfg.heads
        self.qkv      = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj     = nn.Linear(cfg.dim, cfg.dim,     bias=False)
        self.rope     = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv     = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # 各 [B, T, H, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v)) # → [B, H, T, head_dim]

        cos, sin = self.rope(T, x.device)
        q, k     = apply_rotary(q, k, cos, sin)

        # PyTorch ≥ 2.0 自动调用 Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = 0.1 if self.training else 0.0,
            is_causal = True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

# FeedForward（SwiGLU）
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ff, bias=False)   # gate
        self.w2 = nn.Linear(cfg.ff, cfg.dim, bias=False)   # down
        self.w3 = nn.Linear(cfg.dim, cfg.ff, bias=False)   # up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# Transformer Block（Pre-Norm）
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.dim)
        self.ln2  = nn.LayerNorm(cfg.dim)
        self.attn = SelfAttention()
        self.ff   = FeedForward()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok    = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([Block() for _ in range(cfg.layers)])
        self.norm   = nn.LayerNorm(cfg.dim)
        self.head   = nn.Linear(cfg.dim, cfg.vocab, bias=False)

        #lm_head 与 embedding 共享参数
        self.head.weight = self.tok.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.01)
        for name, p in self.named_parameters():
            if "tok" in name or "head" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x      = self.tok(idx)
        for block in self.blocks:
            x  = block(x)
        logits = self.head(self.norm(x))

        loss = None
        if targets is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab),
                shift_targets.view(-1),
            )
        return logits, loss

    def count_params(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        print(f"参数量: {n / 1e6:.2f}M")
        return n

# 数据集
# train_tokens.pt 期望形状：[N, seq+1]，每行为一条 token 序列
class TokenDataset(Dataset):
    def __init__(self):
        print("正在加载 token 文件...")
        # 修复：weights_only=True 消除新版 PyTorch 的安全警告
        self.tokens = torch.load(cfg.tokens, weights_only=True)
        assert self.tokens.ndim == 2, (
            f"期望形状 [N, seq+1]，实际得到 {self.tokens.shape}"
        )
        # 数据验证：检查token范围
        min_id = self.tokens.min().item()
        max_id = self.tokens.max().item()
        if min_id < 0 or max_id >= cfg.vocab:
            print(f"警告：token ID 超出范围！最小: {min_id}, 最大: {max_id}, vocab: {cfg.vocab}")
            # 截断超出范围的token
            self.tokens = torch.clamp(self.tokens, 0, cfg.vocab - 1)
            print("已将超出范围的token截断到有效范围。")
        else:
            print("token ID 范围检查通过。")
        print(f"数据集: {self.tokens.shape}  "
              f"（{len(self.tokens):,} 条序列，序列长 {self.tokens.shape[1]}）")

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int):
        s = self.tokens[i]
        return s[:-1], s[1:]   # input: [seq], target: [seq]（自回归偏移1）


# 学习率：线性 Warmup + 余弦衰减
def get_lr(step: int) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# 训练主循环
def train():
    # ── 数据 ──
    dataset = TokenDataset()
    loader  = DataLoader(
        dataset,
        batch_size         = cfg.batch,
        shuffle            = True,
        num_workers        = cfg.workers,
        pin_memory         = True,
        drop_last          = True,   # [OPT] 丢弃不完整 batch，保证梯度累积对齐
        persistent_workers = True,   # [OPT] 复用 worker 进程，减少进程创建开销
        prefetch_factor    = 4,      # [OPT] 预取 4 个 batch，隐藏数据 IO 延迟
    )

    # ── 模型 ──
    model = MiniLLM().to(device)
    model.count_params()             # 修复：compile 之前调用，否则方法不存在

    # 注意：mode="reduce-overhead" 会启用 CUDAGraphs，在梯度累积场景下
    # 可能会触发 "tensor overwritten by subsequent run" 错误。
    # 修复方案 A（推荐）：改用 "default" 模式，稳定兼容梯度累积。
    # 修复方案 B：保留 reduce-overhead，但每次 forward 前调用
    #             torch.compiler.cudagraph_mark_step_begin()
    model = torch.compile(model, mode="default")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.lr,
        betas        = (0.9, 0.95),
        weight_decay = cfg.wd,
        fused        = True,         # [OPT] 启用 fused AdamW kernel
    )

    autocast = torch.amp.autocast(device_type="cuda", dtype=cfg.dtype)

    # ── 训练状态 ──
    step        = 0      # 优化器更新次数
    batch_idx   = 0      # 修复：独立 micro-batch 计数器，解决梯度累积死锁
    t0          = time.perf_counter()
    tokens_seen = 0

    print("=" * 60)
    print("开始训练")
    print(f"  有效 batch size : {cfg.batch * cfg.acc}")
    print(f"  目标 steps      : {cfg.steps:,}")
    print(f"  设备            : {device}")
    print("=" * 60)

    optimizer.zero_grad(set_to_none=True)

    while step < cfg.steps:
        for x, y in loader:
            #non_blocking=True 异步数据传输，与 GPU 计算流水线并行
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # 告知 CUDAGraphs 新的一步开始，防止张量被覆盖
            # 若使用 mode="default" 则此行无副作用，保留亦可
            torch.compiler.cudagraph_mark_step_begin()

            # ── 前向 + 反向（每个 micro-batch）──
            with autocast:
                _, loss = model(x, y)
                loss_scaled = loss / cfg.acc  # 梯度归一化

            loss_scaled.backward()
            batch_idx   += 1
            tokens_seen += x.numel()

            # ── 每累积 acc 个 micro-batch 执行一次参数更新 ──
            if batch_idx % cfg.acc != 0:
                continue

            lr = get_lr(step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # [OPT]

            # ── 日志（每 10 步打印）──
            if step % 10 == 0:
                elapsed   = time.perf_counter() - t0
                tok_per_s = tokens_seen / elapsed if elapsed > 0 else 0
                print(
                    f"step {step:6d} | "
                    f"loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | "
                    f"tok/s {tok_per_s:>10,.0f}"
                )

            # ── Checkpoint（每 100 步保存）──
            if step % 100 == 0 and step > 0:
                torch.cuda.synchronize()  # 确保 GPU 计算完毕再写磁盘
                torch.save(model._orig_mod.state_dict(), cfg.model)
                print(f"  → checkpoint 已保存至 {cfg.model}")

            step += 1
            if step >= cfg.steps:
                break

    torch.cuda.synchronize()
    torch.save(model._orig_mod.state_dict(), cfg.model)
    total_time = time.perf_counter() - t0
    print("=" * 60)
    print(f"训练完成！总耗时: {total_time / 60:.1f} 分钟")
    print(f"模型已保存至  : {cfg.model}")
    print("=" * 60)


if __name__ == "__main__":
    train()