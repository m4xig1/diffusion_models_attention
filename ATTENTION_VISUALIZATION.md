# JiT register attention visualization

This experiment uses the paper's JiT-B/16 training checkpoint in both inference modes. The upstream `JiT.forward` method switches register tokens on or off with `use_registers`. The same checkpoint, class label, noisy input, and timestep are used in both passes. The image is generated once with the paper's RG scale of 3.5 using 20 Euler steps. This is a controlled register ablation of one model, rather than a comparison against a separately trained register-free JiT checkpoint.

Download `training_b16/checkpoint-last.pth` from [the authors' checkpoint collection](https://huggingface.co/yresearch/register-guidance/tree/main/training_b16) to `checkpoints/jit_b16_register_guidance.pth`. The checkpoint used for the included result has SHA-256 `779291c237b02784d4028452af670e80b546c2dac35060e67de400c03fb104c8`.

Build and run on a CUDA machine:

```bash
docker build -f Dockerfile.attention -t jit-attention .
mkdir -p attention_output
docker run --rm --gpus 'device=0' -v "$PWD/checkpoints:/checkpoints:ro" -v "$PWD/attention_output:/workspace/attention_output" jit-attention python visualize_attention.py --checkpoint /checkpoints/jit_b16_register_guidance.pth
```

`comparison.png` shows the generated image, attention from the center image query to image keys in each mode, and the fraction of each image query's attention assigned to register keys. Rows are layers 4, 7, and 11 at diffusion time 0.5. The pixel-key panels in each row share a color scale; register share uses 0 to 1. Mean attention probabilities are taken across all 12 heads after applying the model's Q/K normalization and RoPE. Raw 16x16 arrays and metadata are also saved.

This is self-attention over image patches and optional register tokens. JiT has no text cross-attention, so these maps cannot be interpreted as prompt-token relevance.

The included result uses ImageNet class 207 (golden retriever), seed 7, and the paper's `training_b16` checkpoint. The center patch assigns 0% of attention to registers in the disabled mode and 1.9%, 19.9%, and 2.7% in the enabled mode at layers 4, 7, and 11. These values describe this one image and timestep; they are not dataset averages.

## Three models without register tokens

The follow-up experiment uses the authors' `finetuning_b16`, `finetuning_l16`, and `finetuning_h16` checkpoints. Download each `checkpoint-last.pth` into `checkpoints/` as `jit_b16_finetuning.pth`, `jit_l16_finetuning.pth`, and `jit_h16_finetuning.pth`. These weights were fine-tuned with register dropout; all inference calls here set `use_registers=False`. This compares three model sizes in register-free inference mode, not three separately trained register-free checkpoints.

```bash
mkdir -p attention_output/multimodel
docker run --rm --gpus 'device=0' -v "$PWD/checkpoints:/checkpoints:ro" -v "$PWD/attention_output:/workspace/attention_output" jit-attention python visualize_multimodel_attention.py --b16 /checkpoints/jit_b16_finetuning.pth --l16 /checkpoints/jit_l16_finetuning.pth --h16 /checkpoints/jit_h16_finetuning.pth
```

`multimodel/three_models_center_query_no_registers.png` plots attention from the center image patch. `multimodel/three_models_mean_received_no_registers.png` averages attention received by each image patch over all queries. Both show the denoising state at steps 10, 25, and 40 of 50 Euler steps. Each model uses the same ImageNet class 207 and initial noise seed 7. Columns select early, middle, and final layers: B/16 `[3, 6, 11]`, L/16 `[6, 12, 23]`, H/16 `[8, 16, 31]`. Each plot uses one absolute color scale across its models, steps, and layers. Underlying 16×16 probability arrays, intermediate states, final samples, and metadata are saved in the same directory.

## Register queries looking at other tokens

JiT has no single CLS token. Its 32 class-conditioned in-context tokens are the paper's registers. To inspect their query rows, run the following with the same three checkpoints:

```bash
mkdir -p attention_output/register_queries
docker run --rm --gpus 'device=0' -v "$PWD/checkpoints:/checkpoints:ro" -v "$PWD/attention_output:/workspace/attention_output" jit-attention python visualize_register_queries.py --b16 /checkpoints/jit_b16_finetuning.pth --l16 /checkpoints/jit_l16_finetuning.pth --h16 /checkpoints/jit_h16_finetuning.pth
```

The script recreates the same register-free denoising trajectories, then evaluates register attention at the saved states. `register_queries/three_models_register0_to_image.png` shows register 0's attention to all 256 image patches; `three_models_mean_register_to_image.png` averages that map over all 32 register queries; `three_models_register_to_register.png` shows the 32×32 register-query to register-key attention matrix, averaged over heads. These are raw attention probabilities. The image heatmaps sum to the fraction of attention directed at image patches, which is written on each panel; the rest goes to register keys. Each plot shares one linear color scale and saturates the top 1% of values so broad patterns remain visible; the saved probability arrays have no clipping. Layer choices start where registers are inserted: B/16 `[4, 7, 11]`, L/16 `[8, 15, 23]`, H/16 `[10, 20, 31]`.
