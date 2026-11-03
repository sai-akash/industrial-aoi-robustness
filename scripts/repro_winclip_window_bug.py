"""
repro_winclip_window_bug.py

Minimal reproducer for the WinCLIP window-embedding bug in anomalib. No dataset needed: one random image.

anomalib's WinClipModel._get_window_embeddings permutes the window tokens to sequence-first (NLD -> LND)
before calling the OpenCLIP vision transformer, but the installed OpenCLIP transformer is batch-first. Attention
then runs across windows instead of across the tokens of one window, and every window gets the same embedding.

Expected output with the affected versions:
    original: window embeddings are identical (std across windows ~0)
    patched:  window embeddings differ, and a window covering the full grid reproduces the global embedding

Usage:
    python scripts/repro_winclip_window_bug.py
"""

from importlib.metadata import version

import torch
import torch.nn.functional as F
from anomalib.models.image.winclip.torch_model import WinClipModel


def window_embeddings_batch_first(self, feature_map: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """The fix: same as anomalib's method, without the NLD -> LND permutes and using the model's own pooling."""
    batch_size, n_masks = feature_map.shape[0], masks.shape[1]
    class_index = torch.zeros(1, n_masks, dtype=int, device=feature_map.device)
    masks = torch.cat((class_index, masks + 1)).T
    masked = torch.cat([torch.index_select(feature_map, 1, mask) for mask in masks])
    visual = self.clip.visual
    masked = visual.transformer(visual.ln_pre(visual.patch_dropout(masked)))
    pooled, _ = visual._pool(masked)  # noqa: SLF001
    if visual.proj is not None:
        pooled = pooled @ visual.proj
    return pooled.reshape((n_masks, batch_size, -1)).permute(1, 0, 2)


def window_spread(embeddings: torch.Tensor) -> float:
    """Largest per-dimension std across the windows of one image; 0 means all windows are identical."""
    return embeddings[0].std(dim=0).max().item()


@torch.inference_mode()
def main():
    print(f"anomalib {version('anomalib')} | open_clip_torch {version('open_clip_torch')} | torch {version('torch')}")
    torch.manual_seed(0)
    model = WinClipModel()
    model.setup("screw", None)
    model.eval()
    image = torch.rand(1, 3, 240, 240)

    captured = {}
    hook = model.clip.visual.patch_dropout.register_forward_hook(lambda _m, inputs, _o: captured.update(tokens=inputs[0].detach()))
    global_embedding, _ = model.clip.encode_image(image)
    hook.remove()
    tokens = captured["tokens"]

    original = WinClipModel._get_window_embeddings  # noqa: SLF001
    for name, method in [("original", original), ("patched", window_embeddings_batch_first)]:
        spreads = [window_spread(method(model, tokens, masks)) for masks in model.masks]
        full_grid = torch.arange(tokens.shape[1] - 1)[:, None]
        cosine = F.cosine_similarity(method(model, tokens, full_grid)[0], global_embedding).item()
        n_windows = [masks.shape[1] for masks in model.masks]
        print(f"{name:>8}: windows per scale {n_windows}, max std across windows {[f'{s:.6f}' for s in spreads]}, full-grid window vs. global cosine {cosine:.4f}")


if __name__ == "__main__":
    main()
