from dlmake import Config
import unittest
import torch


class AttentionTest(unittest.TestCase):
    
    def test_attention_forward_mha(self):
        from dlmake import MultiHeadAttention
        config = Config.from_yaml("tests/configs/attention_mha.yaml")
        attn = MultiHeadAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mha_rope(self):
        from dlmake import MultiHeadAttention
        config = Config.from_yaml("tests/configs/attention_mha_rope.yaml")
        attn = MultiHeadAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mha_qknorm(self):
        from dlmake import MultiHeadAttention
        config = Config.from_yaml("tests/configs/attention_mha_qknorm.yaml")
        attn = MultiHeadAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mha_flash(self):
        from dlmake import MultiHeadAttention
        config = Config.from_yaml("tests/configs/attention_mha_flash.yaml")
        attn = MultiHeadAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mqa(self):
        from dlmake import MultiQueryAttention
        config = Config.from_yaml("tests/configs/attention_mqa.yaml")
        attn = MultiQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mqa_rope(self):
        from dlmake import MultiQueryAttention
        config = Config.from_yaml("tests/configs/attention_mqa_rope.yaml")
        attn = MultiQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mqa_qknorm(self):
        from dlmake import MultiQueryAttention
        config = Config.from_yaml("tests/configs/attention_mqa_qknorm.yaml")
        attn = MultiQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mqa_flash(self):
        from dlmake import MultiQueryAttention
        config = Config.from_yaml("tests/configs/attention_mqa_flash.yaml")
        attn = MultiQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_gqa(self):
        from dlmake import GroupedQueryAttention
        config = Config.from_yaml("tests/configs/attention_gqa.yaml")
        attn = GroupedQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_gqa_rope(self):
        from dlmake import GroupedQueryAttention
        config = Config.from_yaml("tests/configs/attention_gqa_rope.yaml")
        attn = GroupedQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_gqa_qknorm(self):
        from dlmake import GroupedQueryAttention
        config = Config.from_yaml("tests/configs/attention_gqa_qknorm.yaml")
        attn = GroupedQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_gqa_flash(self):
        from dlmake import GroupedQueryAttention
        config = Config.from_yaml("tests/configs/attention_gqa_flash.yaml")
        attn = GroupedQueryAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)

    def test_attention_forward_mla(self):
        from dlmake import MultiHeadLatentAttention
        config = Config.from_yaml("tests/configs/attention_mla.yaml")
        attn = MultiHeadLatentAttention(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        attn(x)
