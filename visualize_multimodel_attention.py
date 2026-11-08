"""Plot register-free attention through denoising for JiT-B/L/H-16."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model_jit import JiT_models
from visualize_attention import heatmap, to_image


MODELS = (
    ("JiT-B/16", "b16", 4),
    ("JiT-L/16", "l16", 8),
    ("JiT-H/16", "h16", 10),
)


def load_model(name, start, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    key = "model_ema1" if "model_ema1" in checkpoint else "model"
    state = checkpoint[key]
    state = {k.removeprefix("net."): v for k, v in state.items()}
    model = JiT_models[name](input_size=256, in_context_start=start, in_context_len=32)
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    return model.eval().cuda()


def selected_layers(depth):
    return [round((depth - 1) * fraction) for fraction in (0.25, 0.5, 1.0)]


def forward_with_maps(model, z, timestep, label, layers):
    maps = {}
    handles = []
    query = 8 * 16 + 8

    def hook_for(index):
        def hook(module, args, kwargs):
            hidden = args[0]
            batch, tokens, width = hidden.shape
            assert tokens == 256, "register-free mode must contain only 16x16 image patches"
            qkv = module.qkv(hidden).reshape(batch, tokens, 3, module.num_heads, width // module.num_heads).permute(2, 0, 3, 1, 4)
            q = kwargs["rope"](module.q_norm(qkv[0]))
            k = kwargs["rope"](module.k_norm(qkv[1]))
            probability = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).float().softmax(dim=-1)
            mean_row = probability[0, :, query, :].mean(dim=0)
            mean_received = probability[0].mean(dim=(0, 1))
            maps[index] = {
                "center_query": mean_row.cpu().numpy().reshape(16, 16),
                "mean_received": mean_received.cpu().numpy().reshape(16, 16),
            }
        return hook

    for index in layers:
        handles.append(model.blocks[index].attn.register_forward_pre_hook(hook_for(index), with_kwargs=True))
    try:
        prediction = model(z, torch.tensor([timestep], device=z.device), label, use_registers=False)
    finally:
        for handle in handles:
            handle.remove()
    return prediction, maps


@torch.inference_mode()
def run_model(model, seed, class_id, steps, captures, output_dir, slug):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn((1, 3, 256, 256), generator=generator, device="cuda")
    label = torch.tensor([class_id], device="cuda")
    layers = selected_layers(len(model.blocks))
    records = {}
    for step in range(steps):
        t = step / steps
        if step in captures:
            stage = to_image(z[0])
            prediction, maps = forward_with_maps(model, z, t, label, layers)
            records[step] = {"stage": stage, "maps": maps}
            stage.save(output_dir / f"{slug}_step{step}_state.png")
            for layer, views in maps.items():
                for view, values in views.items():
                    np.save(output_dir / f"{slug}_step{step}_layer{layer}_{view}.npy", values)
        else:
            prediction = model(z, torch.tensor([t], device="cuda"), label, use_registers=False)
        z = z + (prediction - z) / max(1 - t, 0.05) / steps
    to_image(z[0]).save(output_dir / f"{slug}_final.png")
    return layers, records


def draw_figure(results, steps, captures, output_dir, view):
    cell = 192
    header = 42
    row_height = cell + 28
    footer = 54
    width = cell * 4
    height = header + row_height * len(MODELS) * len(captures) + footer
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 12) if Path(font_path).exists() else ImageFont.load_default()
    titles = ("Denoising state", "Early layer", "Middle layer", "Final layer")
    for col, title in enumerate(titles):
        draw.text((col * cell + 8, 10), title, fill="black", font=font)
    vmax = max(float(views[view].max()) for result in results.values() for record in result["records"].values() for views in record["maps"].values())
    for model_index, (name, slug, _) in enumerate(MODELS):
        layers = results[slug]["layers"]
        for time_index, step in enumerate(captures):
            row = model_index * len(captures) + time_index
            top = header + row * row_height
            draw.rectangle((0, top, width, top + row_height - 1), fill="#f3f3f3")
            draw.text((8, top + 4), f"{name}  |  step {step}/{steps}  |  t={step/steps:.2f}  |  layers {layers}", fill="black", font=small)
            record = results[slug]["records"][step]
            figure.paste(record["stage"].resize((cell, cell)), (0, top + 28))
            for column, layer in enumerate(layers, start=1):
                figure.paste(heatmap(record["maps"][layer][view], cell, vmax), (column * cell, top + 28))
    bar_y = height - 35
    for x in range(25, width - 95):
        color = heatmap(np.array([[(x - 25) / (width - 120)]]), 1, 1).getpixel((0, 0))
        draw.line((x, bar_y, x, bar_y + 12), fill=color)
    draw.text((25, bar_y + 15), "0", fill="black", font=small)
    draw.text((width - 88, bar_y + 15), f"{vmax:.3f} probability", fill="black", font=small)
    path = output_dir / f"three_models_{view}_no_registers.png"
    figure.save(path)
    return path, vmax


def main():
    parser = argparse.ArgumentParser()
    for _, slug, _ in MODELS:
        parser.add_argument(f"--{slug}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("attention_output/multimodel"))
    parser.add_argument("--class-id", type=int, default=207)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--capture-steps", type=int, nargs="+", default=[10, 25, 40])
    args = parser.parse_args()
    assert torch.cuda.is_available()
    assert 0 <= args.class_id < 1000 and args.steps > 0
    assert len(set(args.capture_steps)) == len(args.capture_steps)
    assert all(0 <= step < args.steps for step in args.capture_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, slug, start in MODELS:
        model = load_model(name, start, getattr(args, slug))
        layers, records = run_model(model, args.seed, args.class_id, args.steps, args.capture_steps, args.output_dir, slug)
        results[slug] = {"layers": layers, "records": records}
        del model
        torch.cuda.empty_cache()
        print(f"Completed {name}: layers {layers}", flush=True)
    plots = {}
    scales = {}
    for view in ("center_query", "mean_received"):
        path, vmax = draw_figure(results, args.steps, args.capture_steps, args.output_dir, view)
        plots[view] = str(path)
        scales[view] = vmax
    metadata = {
        "models": {slug: {"name": name, "checkpoint": str(getattr(args, slug)), "layers": results[slug]["layers"]} for name, slug, _ in MODELS},
        "registers": False,
        "class_id": args.class_id,
        "seed": args.seed,
        "sampling": "50-step Euler flow ODE with no guidance" if args.steps == 50 else f"{args.steps}-step Euler flow ODE with no guidance",
        "steps": args.steps,
        "capture_steps": args.capture_steps,
        "timesteps": [step / args.steps for step in args.capture_steps],
        "query_patch": [8, 8],
        "aggregation": {"center_query": "attention from center query, mean over heads", "mean_received": "attention to each key, mean over heads and image queries"},
        "color_scale_max": scales,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {plots}", flush=True)


if __name__ == "__main__":
    main()
