"""Plot class-conditioned register queries attending to image and register keys."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from visualize_attention import heatmap, to_image
from visualize_multimodel_attention import MODELS, load_model


def register_layers(model):
    start = model.in_context_start
    last = len(model.blocks) - 1
    return [start, (start + last) // 2, last]


def capture_register_attention(model, z, t, label, layers):
    maps = {}
    handles = []
    registers = model.in_context_len

    def hook_for(index):
        def hook(module, args, kwargs):
            hidden = args[0]
            batch, tokens, width = hidden.shape
            assert tokens == registers + 256
            qkv = module.qkv(hidden).reshape(batch, tokens, 3, module.num_heads, width // module.num_heads).permute(2, 0, 3, 1, 4)
            q = kwargs["rope"](module.q_norm(qkv[0]))
            k = kwargs["rope"](module.k_norm(qkv[1]))
            attention = ((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).float().softmax(dim=-1)
            register_rows = attention[0, :, :registers, :].mean(dim=0)
            pixel = register_rows[:, registers:].cpu().numpy().reshape(registers, 16, 16)
            register = register_rows[:, :registers].cpu().numpy()
            maps[index] = {
                "register0_to_image": pixel[0],
                "mean_register_to_image": pixel.mean(axis=0),
                "register_to_register": register,
                "all_register_to_image": pixel,
                "register0_image_mass": float(pixel[0].sum()),
                "mean_image_mass": float(pixel.mean(axis=0).sum()),
            }
        return hook

    for index in layers:
        handles.append(model.blocks[index].attn.register_forward_pre_hook(hook_for(index), with_kwargs=True))
    try:
        model(z, torch.tensor([t], device=z.device), label, use_registers=True)
    finally:
        for handle in handles:
            handle.remove()
    return maps


@torch.inference_mode()
def run_model(model, slug, seed, class_id, steps, captures, output_dir):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    z = torch.randn((1, 3, 256, 256), generator=generator, device="cuda")
    label = torch.tensor([class_id], device="cuda")
    layers = register_layers(model)
    records = {}
    for step in range(steps):
        t = step / steps
        if step in captures:
            stage = to_image(z[0])
            stage.save(output_dir / f"{slug}_step{step}_state.png")
            maps = capture_register_attention(model, z, t, label, layers)
            records[step] = {"stage": stage, "maps": maps}
            for layer, views in maps.items():
                np.save(output_dir / f"{slug}_step{step}_layer{layer}_all_register_to_image.npy", views["all_register_to_image"])
                np.save(output_dir / f"{slug}_step{step}_layer{layer}_register_to_register.npy", views["register_to_register"])
        prediction = model(z, torch.tensor([t], device=z.device), label, use_registers=False)
        z = z + (prediction - z) / max(1 - t, 0.05) / steps
    return layers, records


def draw_figure(results, steps, captures, output_dir, view):
    cell, header, row_height, footer = 192, 42, 220, 54
    width = cell * 4
    height = header + row_height * len(MODELS) * len(captures) + footer
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 12) if Path(font_path).exists() else ImageFont.load_default()
    for col, title in enumerate(("Denoising state", "First register layer", "Middle register layer", "Final layer")):
        draw.text((col * cell + 6, 10), title, fill="black", font=font)
    all_values = np.concatenate([
        views[view].reshape(-1)
        for result in results.values()
        for record in result["records"].values()
        for views in record["maps"].values()
    ])
    vmax = float(np.quantile(all_values, 0.99))
    for model_index, (name, slug, _) in enumerate(MODELS):
        layers = results[slug]["layers"]
        for time_index, step in enumerate(captures):
            row = model_index * len(captures) + time_index
            top = header + row * row_height
            draw.rectangle((0, top, width, top + row_height - 1), fill="#f3f3f3")
            draw.text((8, top + 4), f"{name}  |  step {step}/{steps}  |  t={step/steps:.2f}  |  layers {layers}", fill="black", font=small)
            record = results[slug]["records"][step]
            figure.paste(record["stage"].resize((cell, cell)), (0, top + 28))
            for col, layer in enumerate(layers, 1):
                views = record["maps"][layer]
                figure.paste(heatmap(views[view], cell, vmax), (col * cell, top + 28))
                if view != "register_to_register":
                    mass = views["register0_image_mass"] if view == "register0_to_image" else views["mean_image_mass"]
                    draw.text((col * cell + 5, top + 33), f"to pixels {mass:.0%}", fill="white", stroke_width=1, stroke_fill="black", font=small)
    bar_y = height - 35
    for x in range(25, width - 95):
        color = heatmap(np.array([[(x - 25) / (width - 120)]]), 1, 1).getpixel((0, 0))
        draw.line((x, bar_y, x, bar_y + 12), fill=color)
    draw.text((25, bar_y + 15), "0", fill="black", font=small)
    draw.text((width - 170, bar_y + 15), f">={vmax:.3f} (99th pct)", fill="black", font=small)
    path = output_dir / f"three_models_{view}.png"
    figure.save(path)
    return path, vmax


def main():
    parser = argparse.ArgumentParser()
    for _, slug, _ in MODELS:
        parser.add_argument(f"--{slug}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("attention_output/register_queries"))
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
        layers, records = run_model(model, slug, args.seed, args.class_id, args.steps, args.capture_steps, args.output_dir)
        results[slug] = {"layers": layers, "records": records}
        del model
        torch.cuda.empty_cache()
        print(f"Completed {name}: register layers {layers}", flush=True)
    plots, scales = {}, {}
    for view in ("register0_to_image", "mean_register_to_image", "register_to_register"):
        path, vmax = draw_figure(results, args.steps, args.capture_steps, args.output_dir, view)
        plots[view] = str(path)
        scales[view] = vmax
    image_mass = {}
    for _, slug, _ in MODELS:
        image_mass[slug] = {}
        for step, record in results[slug]["records"].items():
            image_mass[slug][str(step)] = {}
            for layer in results[slug]["layers"]:
                views = record["maps"][layer]
                image_mass[slug][str(step)][str(layer)] = {
                    "register0": views["register0_image_mass"],
                    "mean_register": views["mean_image_mass"],
                }
    metadata = {
        "model_layers": {slug: results[slug]["layers"] for _, slug, _ in MODELS},
        "checkpoint_paths": {slug: str(getattr(args, slug)) for _, slug, _ in MODELS},
        "class_id": args.class_id,
        "seed": args.seed,
        "steps": args.steps,
        "capture_steps": args.capture_steps,
        "register_count": 32,
        "trajectory": f"register-free {args.steps}-step Euler flow ODE, same as previous plots",
        "register_queries": "class-conditioned register tokens; register 0 is the first one",
        "attention": "mean over heads after Q/K normalization and RoPE",
        "image_key_probability_mass": image_mass,
        "color_scale_max": scales,
        "color_scale": "linear, shared within each plot, clipped above the 99th percentile; raw arrays are unchanged",
        "plots": plots,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {plots}", flush=True)


if __name__ == "__main__":
    main()
