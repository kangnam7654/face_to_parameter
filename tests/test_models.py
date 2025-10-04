import unittest
import torch
from models.imitator import Imitator

class TestImitatorModel(unittest.TestCase):
    def test_forward_pass(self):
        """
        Tests the forward pass of the Imitator model to ensure correct output shape.
        """
        batch_size = 2
        latent_dim = 960
        expected_output_shape = (batch_size, 3, 512, 512)

        # Initialize the model
        model = Imitator(latent_dim=latent_dim)

        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, latent_dim)

        # Perform the forward pass
        output = model(dummy_input)

        # Assert that the output shape is as expected
        self.assertEqual(output.shape, expected_output_shape)

if __name__ == '__main__':
    unittest.main()
