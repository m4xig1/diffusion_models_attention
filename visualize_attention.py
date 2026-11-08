"""Visualize JiT-B/16 self-attention with and without register tokens.

Uses the same EMA checkpoint, noise, class, and timestep for both passes.
The image is generated once with register guidance, then used as context for
the paired attention measurements.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model_jit import JiT_models


def to_image(tensor):
    pixels = (tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def heatmap(values, size, vmax):
    x = np.clip(values / max(vmax, 1e-12), 0, 1)
    # Dark purple -> red -> yellow; a fixed scale is shared across paired panels.
    stops = np.array([[16, 8, 43], [91, 21, 110], [208, 43, 62], [251, 153, 44], [252, 247, 163]])
    pos = x * (len(stops) - 1)
    low = np.minimum(pos.astype(int), len(stops) - 2)
    rgb = stops[low] * (1 - (pos - low))[..., None] + stops[low + 1] * (pos - low)[..., None]
    return Image.fromarray(rgb.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_ema1"]
    prefix = "net."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model = JiT_models["JiT-B/16"](input_size=256, in_context_start=4, in_context_len=32)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    return model.eval().to(device)


@torch.no_grad()
def generate(model, label, seed, steps):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn((1, 3, 256, 256), generator=generator, device="cuda")
    labels = torch.tensor([label], device="cuda")
    for i in range(steps):
        t = i / steps
        tt = torch.tensor([t], device="cuda")
        with_regs = model(z, tt, labels, use_registers=True)
        without_regs = model(z, tt, labels, use_registers=False)
        velocity = (without_regs + 3.5 * (with_regs - without_regs) - z) / max(1 - t, 0.05)
        if not 0.03 < t < 0.9:
            velocity = (with_regs - z) / max(1 - t, 0.05)
        z = z + velocity / steps
    return z


def capture(model, x, t, label, use_registers, layers):
    maps = {}
    handles = []
    register_count = model.in_context_len if use_registers else 0
    patch_count = (model.input_size // model.patch_size) ** 2
    query = register_count + (patch_count // 2 + (model.input_size // model.patch_size) // 2)

    def hook_for(index):
        def hook(module, args, kwargs):
            hidden = args[0]
            batch, length, width = hidden.shape
            qkv = module.qkv(hidden).reshape(batch, length, 3, module.num_heads, width // module.num_heads).permute(2, 0, 3, 1, 4)
            q = kwargs["rope"](module.q_norm(qkv[0]))
            k = kwargs["rope"](module.k_norm(qkv[1]))
            score = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attention = score.float().softmax(dim=-1).mean(dim=1)[0]
            maps[index] = {
                "pixel_keys": attention[query, register_count:].detach().cpu().numpy().reshape(16, 16),
                "register_share": attention[register_count:, :register_count].sum(-1).detach().cpu().numpy().reshape(16, 16) if register_count else None,
                "query_register_share": float(attention[query, :register_count].sum()) if register_count else 0.0,
                "row_sum": float(attention[query].sum()),
            }
        return hook

    for index in layers:
        handles.append(model.blocks[index].attn.register_forward_pre_hook(hook_for(index), with_kwargs=True))
    try:
        with torch.no_grad():
            model(x, torch.tensor([t], device=x.device), torch.tensor([label], device=x.device), use_registers=use_registers)
    finally:
        for handle in handles:
            handle.remove()
    return maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("attention_output"))
    parser.add_argument("--class-id", type=int, default=207, help="ImageNet class ID (207: golden retriever)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--timestep", type=float, default=0.5)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 7, 11])
    args = parser.parse_args()
    assert torch.cuda.is_available(), "A CUDA GPU is required by the upstream RoPE implementation"
    assert 0 <= args.class_id < 1000 and 0 < args.timestep < 1
    assert all(0 <= layer < 12 for layer in args.layers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, "cuda")
    sample = generate(model, args.class_id, args.seed, args.steps)
    to_image(sample[0]).save(args.output_dir / "sample.png")

    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    noise = torch.randn(sample.shape, generator=generator, device="cuda")
    z = args.timestep * sample + (1 - args.timestep) * noise
    to_image(z[0]).save(args.output_dir / "noisy_input.png")
    paired = {mode: capture(model, z, args.timestep, args.class_id, mode == "with_registers", args.layers)
              for mode in ("without_registers", "with_registers")}
    meta = {"checkpoint": str(args.checkpoint), "class_id": args.class_id, "seed": args.seed,
            "steps": args.steps, "timestep": args.timestep, "layers": args.layers,
            "query_patch": [8, 8], "heads": "mean of 12", "register_count": 32,
            "query_register_share": {mode: {str(i): paired[mode][i]["query_register_share"] for i in args.layers} for mode in paired},
            "row_sum": {mode: {str(i): paired[mode][i]["row_sum"] for i in args.layers} for mode in paired}}
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    for mode in paired:
        for layer in args.layers:
            np.save(args.output_dir / f"{mode}_layer{layer}_pixel_keys.npy", paired[mode][layer]["pixel_keys"])
            if mode == "with_registers":
                np.save(args.output_dir / f"{mode}_layer{layer}_register_share.npy", paired[mode][layer]["register_share"])

    cell = 256
    board = Image.new("RGB", (cell * 4, 28 + len(args.layers) * (cell + 28)), "white")
    draw = ImageDraw.Draw(board)
    titles = ["Generated image", "No registers: pixel keys", "With registers: pixel keys", "With registers: register share"]
    for col, title in enumerate(titles):
        draw.text((col * cell + 5, 5), title, fill="black")
    for row, layer in enumerate(args.layers):
        top = 28 + row * (cell + 28)
        board.paste(to_image(sample[0]), (0, top))
        a = paired["without_registers"][layer]["pixel_keys"]
        b = paired["with_registers"][layer]["pixel_keys"]
        scale = max(float(a.max()), float(b.max()))
        board.paste(heatmap(a, cell, scale), (cell, top))
        board.paste(heatmap(b, cell, scale), (cell * 2, top))
        board.paste(heatmap(paired["with_registers"][layer]["register_share"], cell, 1), (cell * 3, top))
        draw.text((5, top + 5), f"Layer {layer} / t={args.timestep}", fill="white", stroke_width=1, stroke_fill="black")
    board.save(args.output_dir / "comparison.png")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
