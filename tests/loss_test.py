from dlmake import Config, BaseTransformer, OuroLoss
import unittest
import torch


class LossTest(unittest.TestCase):

    def test_ouro_loss(self):
        config = Config.from_yaml("tests/configs/complete_model.yaml")
        model = BaseTransformer(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        data = torch.randint(0, 1024, (4, 129), dtype=torch.long).to(device)
        x = data[:, :-1]
        loss_fn = OuroLoss()
        target = data[:, 1:].contiguous()
        y = model(x)
        loss = loss_fn(y, target, model.unnormalized)
        self.assertFalse((loss == float("-inf")).item() or (loss == float("inf")).item(), "The loss value cannot be infinite")
