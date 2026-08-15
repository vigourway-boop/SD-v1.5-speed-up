import unittest

import torch

import config


class GeneratorSeedTest(unittest.TestCase):
    def test_explicit_seed_is_reproducible(self):
        first = torch.rand(16, generator=config.make_generator(12345), device=config.DEVICE)
        second = torch.rand(16, generator=config.make_generator(12345), device=config.DEVICE)
        self.assertTrue(torch.equal(first, second))

    def test_different_explicit_seeds_change_noise(self):
        first = torch.rand(16, generator=config.make_generator(12345), device=config.DEVICE)
        second = torch.rand(16, generator=config.make_generator(54321), device=config.DEVICE)
        self.assertFalse(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
