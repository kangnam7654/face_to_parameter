import io
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path

import torch

import infer_legacy
from models.imitator import Imitator
from models.legacy_imitator import (
    LEGACY_LATENT_DIM,
    LegacyCheckpointError,
    LegacyImitator,
    load_legacy_imitator,
    validate_legacy_state_dict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_CHECKPOINT = REPO_ROOT / "weigths" / "imitator.pt"

EXPECTED_KEYS = [
    "deconv1.conv.weight",
    "deconv1.bn.weight",
    "deconv1.bn.bias",
    "deconv1.bn.running_mean",
    "deconv1.bn.running_var",
    "deconv1.bn.num_batches_tracked",
    "deconv2.conv.weight",
    "deconv3.conv.weight",
    "deconv4.conv.weight",
    "deconv5.conv.weight",
    "deconv6.conv.weight",
    "deconv7.conv.weight",
]


class _NotAStateDict:
    """Arbitrary picklable class; must be refused by weights_only loading."""

    def __init__(self):
        self.payload = 1


def _fresh_state_dict():
    torch.manual_seed(0)
    return OrderedDict(LegacyImitator().state_dict())


class LegacySchemaTest(unittest.TestCase):
    def test_state_dict_keys_and_first_layer_shape(self):
        sd = LegacyImitator().state_dict()
        self.assertEqual(sorted(sd.keys()), sorted(EXPECTED_KEYS))
        self.assertEqual(tuple(sd["deconv1.conv.weight"].shape), (37, 512, 4, 4))
        self.assertEqual(tuple(sd["deconv7.conv.weight"].shape), (3, 64, 3, 3))

    def test_newer_imitator_state_dict_is_rejected(self):
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(Imitator(latent_dim=37).state_dict())
        self.assertIn("unexpected keys", str(ctx.exception))

    def test_missing_key_is_rejected(self):
        sd = _fresh_state_dict()
        del sd["deconv4.conv.weight"]
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(sd)
        self.assertIn("missing keys", str(ctx.exception))

    def test_extra_key_is_rejected(self):
        sd = _fresh_state_dict()
        sd["deconv8.conv.weight"] = torch.zeros(1)
        with self.assertRaises(LegacyCheckpointError):
            validate_legacy_state_dict(sd)

    def test_shape_mismatch_is_rejected(self):
        sd = _fresh_state_dict()
        sd["deconv1.conv.weight"] = torch.zeros(960, 512, 4, 4)
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(sd)
        self.assertIn("shape mismatch", str(ctx.exception))

    def test_non_finite_weight_is_rejected(self):
        sd = _fresh_state_dict()
        sd["deconv2.conv.weight"][0, 0, 0, 0] = float("nan")
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(sd)
        self.assertIn("NaN or Inf", str(ctx.exception))

    def test_float_dtype_mismatch_is_rejected(self):
        sd = _fresh_state_dict()
        sd["deconv2.conv.weight"] = sd["deconv2.conv.weight"].half()
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(sd)
        self.assertIn("expected torch.float32", str(ctx.exception))

    def test_counter_dtype_mismatch_is_rejected(self):
        sd = _fresh_state_dict()
        sd["deconv1.bn.num_batches_tracked"] = sd["deconv1.bn.num_batches_tracked"].int()
        with self.assertRaises(LegacyCheckpointError) as ctx:
            validate_legacy_state_dict(sd)
        self.assertIn("expected torch.int64", str(ctx.exception))

    def test_non_mapping_is_rejected(self):
        with self.assertRaises(LegacyCheckpointError):
            validate_legacy_state_dict([torch.zeros(1)])


class LegacyLoadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_strict_load(self):
        sd = _fresh_state_dict()
        path = self.tmp / "legacy.pt"
        torch.save(sd, path)
        model = load_legacy_imitator(path)
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, sd[key]), key)

    def test_missing_file_is_reported(self):
        with self.assertRaises(LegacyCheckpointError):
            load_legacy_imitator(self.tmp / "does_not_exist.pt")

    def test_arbitrary_pickle_is_refused(self):
        path = self.tmp / "object.pt"
        torch.save(_NotAStateDict(), path)
        with self.assertRaises(LegacyCheckpointError) as ctx:
            load_legacy_imitator(path)
        self.assertIn("weights_only=True", str(ctx.exception))

    def test_incompatible_checkpoint_is_not_partially_loaded(self):
        sd = _fresh_state_dict()
        sd["deconv3.conv.weight"] = torch.zeros(512, 512, 5, 5)
        path = self.tmp / "bad.pt"
        torch.save(sd, path)
        with self.assertRaises(LegacyCheckpointError):
            load_legacy_imitator(path)


class LegacyForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = LegacyImitator().eval()

    def test_output_shape_and_range(self):
        with torch.no_grad():
            out = self.model(torch.rand(2, LEGACY_LATENT_DIM))
        self.assertEqual(tuple(out.shape), (2, 3, 256, 256))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertLessEqual(float(out.abs().max()), 1.0)

    def test_wrong_input_dim_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 960))
        with self.assertRaises(ValueError):
            self.model(torch.rand(LEGACY_LATENT_DIM))

    def test_synthetic_vector_is_deterministic(self):
        a = infer_legacy.make_synthetic_vector(7)
        b = infer_legacy.make_synthetic_vector(7)
        c = infer_legacy.make_synthetic_vector(8)
        self.assertEqual(tuple(a.shape), (1, LEGACY_LATENT_DIM))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_generate_is_deterministic(self):
        a = infer_legacy.generate(self.model, 3)
        b = infer_legacy.generate(self.model, 3)
        self.assertTrue(torch.equal(a, b))
        arr = infer_legacy.to_rgb_uint8(a)
        self.assertEqual(arr.shape, (256, 256, 3))
        self.assertEqual(str(arr.dtype), "uint8")

    def test_invalid_seed_raises(self):
        for bad in (-1, 2**32, 1.5, True, "3"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                infer_legacy.make_synthetic_vector(bad)

    def test_invalid_output_tensor_is_rejected(self):
        with self.assertRaises(ValueError):
            infer_legacy.validate_output(torch.zeros(1, 3, 128, 128))
        bad = torch.zeros(1, 3, 256, 256)
        bad[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            infer_legacy.validate_output(bad)
        with self.assertRaises(ValueError):
            infer_legacy.validate_output(torch.full((1, 3, 256, 256), 2.0))


class CliArgumentTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _expect_usage_error(self, argv):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                infer_legacy.main(argv)
        self.assertEqual(ctx.exception.code, 2)

    def test_output_is_required(self):
        self._expect_usage_error(["--seed", "0"])

    def test_seed_is_required(self):
        self._expect_usage_error(["--output", str(self.tmp / "x.png")])

    def test_negative_seed_rejected(self):
        self._expect_usage_error(["--seed", "-1", "--output", str(self.tmp / "x.png")])

    def test_non_png_output_rejected(self):
        self._expect_usage_error(["--seed", "0", "--output", str(self.tmp / "x.jpg")])

    def test_missing_output_parent_rejected(self):
        self._expect_usage_error(
            ["--seed", "0", "--output", str(self.tmp / "nope" / "x.png")]
        )

    def test_existing_output_requires_overwrite(self):
        target = self.tmp / "x.png"
        target.write_bytes(b"")
        self._expect_usage_error(["--seed", "0", "--output", str(target)])

    def test_incompatible_checkpoint_returns_error_and_writes_nothing(self):
        bad = self.tmp / "bad.pt"
        torch.save({"deconv1.conv.weight": torch.zeros(1)}, bad)
        target = self.tmp / "x.png"
        with redirect_stderr(io.StringIO()) as err:
            code = infer_legacy.main(
                ["--checkpoint", str(bad), "--seed", "0", "--output", str(target)]
            )
        self.assertEqual(code, 1)
        self.assertFalse(target.exists())
        self.assertIn("incompatible", err.getvalue())


@unittest.skipUnless(TRACKED_CHECKPOINT.is_file(), "tracked legacy checkpoint not present")
class TrackedCheckpointTest(unittest.TestCase):
    """Strict compatibility of the real committed checkpoint (load only)."""

    def test_tracked_checkpoint_loads_strictly(self):
        sd = torch.load(str(TRACKED_CHECKPOINT), map_location="cpu", weights_only=True)
        self.assertEqual(sorted(sd.keys()), sorted(EXPECTED_KEYS))
        validate_legacy_state_dict(sd)
        model = load_legacy_imitator(TRACKED_CHECKPOINT)
        self.assertIsInstance(model, LegacyImitator)

    def test_tracked_checkpoint_rejected_by_newer_imitator(self):
        sd = torch.load(str(TRACKED_CHECKPOINT), map_location="cpu", weights_only=True)
        with self.assertRaises(RuntimeError):
            Imitator(latent_dim=37).load_state_dict(sd, strict=True)


if __name__ == "__main__":
    unittest.main()
