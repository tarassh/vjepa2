import unittest
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import RoPEAreaAttention, RoPEAttention
from src.models.vision_transformer import VisionTransformer


def _make_video_mask(batch_size, total_tokens, visible_tokens):
    return torch.stack([torch.sort(torch.randperm(total_tokens)[:visible_tokens])[0] for _ in range(batch_size)])


def _make_test_vit(attention_pattern=None, use_area_attention=False, area_attention_layers=None):
    return VisionTransformer(
        img_size=64,
        patch_size=16,
        num_frames=4,
        tubelet_size=2,
        embed_dim=192,
        depth=4,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        use_sdpa=False,
        use_rope=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        attention_pattern=attention_pattern,
        use_area_attention=use_area_attention,
        area_attention_layers=area_attention_layers,
        area_spatial_splits=2,
        area_temporal_splits=2,
    )


class TestAreaAttention(unittest.TestCase):
    def test_area_attention_loads_rope_weights(self):
        rope_attn = RoPEAttention(dim=192, num_heads=3, qkv_bias=True, grid_size=8, use_sdpa=False)
        area_attn = RoPEAreaAttention(dim=192, num_heads=3, qkv_bias=True, grid_size=8, use_sdpa=False)

        msg = area_attn.load_state_dict(rope_attn.state_dict(), strict=False)

        self.assertEqual(msg.missing_keys, [])
        self.assertEqual(msg.unexpected_keys, [])
        for key, value in rope_attn.state_dict().items():
            torch.testing.assert_close(value, area_attn.state_dict()[key])

    def test_single_area_matches_full_attention(self):
        torch.manual_seed(0)

        rope_attn = RoPEAttention(dim=192, num_heads=3, qkv_bias=True, grid_size=8, use_sdpa=False)
        area_attn = RoPEAreaAttention(
            dim=192,
            num_heads=3,
            qkv_bias=True,
            grid_size=8,
            use_sdpa=False,
            spatial_splits=1,
            temporal_splits=1,
        )
        area_attn.load_state_dict(rope_attn.state_dict(), strict=False)

        x = torch.randn(1, 32, 192)
        rope_out = rope_attn(x, T=2, H_patches=4, W_patches=4)
        area_out = area_attn(x, T=2, H_patches=4, W_patches=4)

        torch.testing.assert_close(rope_out, area_out, atol=1e-6, rtol=1e-6)

    def test_attention_pattern_builds_hybrid_stack(self):
        model = _make_test_vit(attention_pattern=["area", "area", "global", "global"])

        self.assertEqual(model.attention_pattern, ["area", "area", "global", "global"])
        self.assertEqual(type(model.blocks[0].attn).__name__, "RoPEAreaAttention")
        self.assertEqual(type(model.blocks[1].attn).__name__, "RoPEAreaAttention")
        self.assertEqual(type(model.blocks[2].attn).__name__, "RoPEAttention")
        self.assertEqual(type(model.blocks[3].attn).__name__, "RoPEAttention")

    def test_legacy_area_range_builds_same_pattern(self):
        model = _make_test_vit(use_area_attention=True, area_attention_layers=[0, 2])

        self.assertEqual(model.attention_pattern, ["area", "area", "global", "global"])

    def test_baseline_state_dict_loads_strictly_into_hybrid(self):
        baseline = _make_test_vit()

        for hybrid in (
            _make_test_vit(attention_pattern=["area", "area", "global", "global"]),
            _make_test_vit(use_area_attention=True, area_attention_layers=[0, 2]),
        ):
            msg = hybrid.load_state_dict(baseline.state_dict(), strict=True)
            self.assertEqual(msg.missing_keys, [])
            self.assertEqual(msg.unexpected_keys, [])

    def test_hybrid_vit_forward_with_sparse_mask(self):
        torch.manual_seed(0)

        model = _make_test_vit(attention_pattern=["area", "area", "area", "global"])

        x = torch.randn(2, 3, 4, 64, 64)
        total_tokens = (4 // 2) * (64 // 16) * (64 // 16)
        visible_tokens = total_tokens // 2
        mask = _make_video_mask(batch_size=2, total_tokens=total_tokens, visible_tokens=visible_tokens)

        out = model(x, masks=mask)

        self.assertEqual(out.shape, (2, visible_tokens, 192))


if __name__ == "__main__":
    unittest.main()
