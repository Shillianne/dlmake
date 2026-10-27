from dlmake import Config, BaseTransformer
import unittest
import torch


class FullPassTest(unittest.TestCase):
    
    def test_full_pass_training(self):
        config = Config.from_yaml("tests/configs/complete_model.yaml")
        model = BaseTransformer(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        x = torch.randint(0, 1024, (config.batch_size, config.ctx_size), dtype=torch.int32).to(device)
        model(x)
    
    def test_full_pass_multitoken_inference(self):
        config = Config.from_yaml("tests/configs/complete_model.yaml")
        model = BaseTransformer(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        x = torch.randint(0, 1024, (1, config.ctx_size), dtype=torch.int32).to(device)
        model(x)

    def test_full_pass_singletoken_inference(self):
        config = Config.from_yaml("tests/configs/complete_model.yaml")
        model = BaseTransformer(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        x = torch.randint(0, 1024, (1, 1), dtype=torch.int32).to(device)
        model(x)
