from dlmake import Config, Transformer, DecoderBlock
import unittest
import torch


class MockModel(Transformer):
    def __init__(self, config: Config):
        super(MockModel, self).__init__(config)

        self.decoders = [
            DecoderBlock(config=config, layer=i, ff=None, shallow_ff=None)
            for i in range(config.nlayers)
        ]

    def forward(self, *args, start_pos: int = 0):
        x = args[0]
        return x


class FeedForwardTest(unittest.TestCase):
    
    def test_ff_gpt(self):
        from dlmake import GPTFeedForward

        config = Config.from_yaml("tests/configs/ff_gpt.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, GPTFeedForward)
        

    def test_ff_llama(self):
        from dlmake import LlamaFeedForward
        config = Config.from_yaml("tests/configs/ff_llama.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, LlamaFeedForward)


    def test_ff_moe_non_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_moe_no_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)

    def test_ff_moe_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_moe_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, True)
        self.assertEqual(
            model.decoders[0].mlp.expert_w1.size(-1),  # type: ignore
            config.feed_forward.moe.latent_dim
        )

    def test_ff_shallow_gpt_llama(self):
        from dlmake import GPTFeedForward, LlamaFeedForward
        config = Config.from_yaml("tests/configs/ff_shallow_gpt_llama.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, LlamaFeedForward)
        self.assertIsInstance(model.decoders[1].mlp, GPTFeedForward)

    def test_ff_shallow_gpt_moe_non_latent(self):
        from dlmake import GPTFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_gpt_moe_no_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, False)
        self.assertIsInstance(model.decoders[1].mlp, GPTFeedForward)

    def test_ff_shallow_gpt_moe_latent(self):
        from dlmake import GPTFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_gpt_moe_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, True)
        self.assertEqual(
            model.decoders[0].mlp.expert_w1.size(-1),  # type: ignore
            config.feed_forward.shallow_moe.latent_dim
        )
        self.assertIsInstance(model.decoders[1].mlp, GPTFeedForward)

    def test_ff_shallow_llama_gpt(self):
        from dlmake import GPTFeedForward, LlamaFeedForward
        config = Config.from_yaml("tests/configs/ff_shallow_llama_gpt.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, GPTFeedForward)
        self.assertIsInstance(model.decoders[1].mlp, LlamaFeedForward)

    def test_ff_shallow_llama_moe_non_latent(self):
        from dlmake import LlamaFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_llama_moe_no_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, False)
        self.assertIsInstance(model.decoders[1].mlp, LlamaFeedForward)

    def test_ff_shallow_llama_moe_latent(self):
        from dlmake import LlamaFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_llama_moe_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, True)
        self.assertEqual(
            model.decoders[0].mlp.expert_w1.size(-1),  # type: ignore
            config.feed_forward.shallow_moe.latent_dim
        )
        self.assertIsInstance(model.decoders[1].mlp, LlamaFeedForward)
    
    def test_ff_shallow_moe_gpt(self):
        from dlmake import GPTFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_moe_gpt.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, GPTFeedForward)
        self.assertIsInstance(model.decoders[1].mlp, MoE)
        self.assertEqual(model.decoders[1].mlp.use_latent, False)

    def test_ff_shallow_moe_llama(self):
        from dlmake import LlamaFeedForward, MoE
        config = Config.from_yaml("tests/configs/ff_shallow_moe_llama.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, LlamaFeedForward)
        self.assertIsInstance(model.decoders[1].mlp, MoE)
        self.assertEqual(model.decoders[1].mlp.use_latent, False)

    def test_ff_shallow_moe_moe_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_shallow_moe_moe_latent.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, True)
        self.assertEqual(
            model.decoders[0].mlp.expert_w1.size(-1),  # type: ignore
            config.feed_forward.shallow_moe.latent_dim
        )
        self.assertIsInstance(model.decoders[1].mlp, MoE)
        self.assertEqual(model.decoders[1].mlp.use_latent, False)

    def test_ff_shallow_moe_latent_moe(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_shallow_moe_latent_moe.yaml")
        model = MockModel(config)
        self.assertIsInstance(model.decoders[0].mlp, MoE)
        self.assertEqual(model.decoders[0].mlp.use_latent, False)
        self.assertIsInstance(model.decoders[1].mlp, MoE)
        self.assertEqual(model.decoders[1].mlp.use_latent, True)
        self.assertEqual(
            model.decoders[1].mlp.expert_w1.size(-1),  # type: ignore
            config.feed_forward.moe.latent_dim
        )

    def test_ff_forward_gpt(self):
        from dlmake import GPTFeedForward 
        config = Config.from_yaml("tests/configs/ff_forward_gpt.yaml")
        ff = GPTFeedForward(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_llama(self):
        from dlmake import LlamaFeedForward
        config = Config.from_yaml("tests/configs/ff_forward_llama.yaml")
        ff = LlamaFeedForward(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_non_latent_no_shared(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_no_latent_no_shared.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_non_latent_shared_non_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_no_latent_shared_no_latent.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_non_latent_shared_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_no_latent_shared_latent.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_latent_no_shared(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_latent_no_shared.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_latent_shared_non_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_latent_shared_no_latent.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

    def test_ff_forward_moe_latent_shared_latent(self):
        from dlmake import MoE
        config = Config.from_yaml("tests/configs/ff_forward_moe_latent_shared_latent.yaml")
        ff = MoE(config)
        B = config.batch_size
        C = config.ctx_size
        E = config.embed_dim
        x = torch.rand(B, C, E, dtype=config.dtype)
        ff(x)

