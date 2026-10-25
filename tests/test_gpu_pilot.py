"""CPU gates cover leakage/forward/sampling; no CPU optimizer or backward calls."""

import numpy as np
import pytest

from tidetail.pilot_data import safe_file, train_end_with_embargo, windows_from_train


def test_full_event_purge_at_embargo_boundary():
    assert train_end_with_embargo(1000, [(890, 910)]) == 890
    assert train_end_with_embargo(1000, [(800, 850)]) == 904
    assert train_end_with_embargo(1000, [(904, 910)]) == 904


def test_training_normalization_and_window_containment():
    train = np.arange(200, dtype=float)
    x, y, mean, std = windows_from_train(train)
    assert x.shape == (105, 72) and y.shape == (105, 24)
    assert mean == train.mean() and std == train.std()
    np.testing.assert_allclose(y[-1, -1], (199 - mean) / std, rtol=1e-6)
    # Perturb unused rows before preparing exactly the same frozen train block.
    full = np.r_[train, np.ones(100) * 1e9]
    x2, y2, m2, s2 = windows_from_train(full[:200])
    np.testing.assert_array_equal(x, x2)
    np.testing.assert_array_equal(y, y2)
    assert (mean, std) == (m2, s2)


def test_nonfinite_refused():
    with pytest.raises(ValueError):
        windows_from_train(np.r_[np.arange(200), np.nan])


def test_symlink_refused(tmp_path):
    (tmp_path / "source").write_text("test")
    (tmp_path / "link").symlink_to(tmp_path / "source")
    with pytest.raises(ValueError):
        safe_file(tmp_path, "link")


def test_temporal_model_and_sampler_forward_only():
    torch = pytest.importorskip("torch")
    from tidetail.torch_diffusion import Diffusion, TemporalDenoiser

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = TemporalDenoiser().eval()
    diffusion = Diffusion(steps=8)
    with torch.no_grad():
        history = torch.randn(2, 72)
        noisy = torch.randn(2, 1, 24)
        t = torch.tensor([1, 4])
        out = model(noisy, t, history)
        assert out.shape == noisy.shape and torch.isfinite(out).all()
        assert not torch.allclose(out, model(noisy, t, history + 1))
        assert not torch.allclose(out, model(noisy, t + 1, history))
        sample = diffusion.sample(model, history, seed=99)
        assert sample.shape == (2, 1, 24) and torch.isfinite(sample).all()
        torch.testing.assert_close(sample, diffusion.sample(model, history, seed=99))
        assert not torch.allclose(sample, diffusion.sample(model, history + 1, seed=99))
        assert (diffusion.bar[1:] < diffusion.bar[:-1]).all()
        assert diffusion.bar[-1] < 0.001
        with pytest.raises(ValueError):
            model(noisy, t, history[:, :10])


def test_gpu_guard_rejects_conflicting_process(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "gpu_pilot_cli", Path(__file__).parents[1] / "scripts/train_gpu_pilot.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    uuid = "6be030f8-625e-c4e7-a2fb-1309a673664b"
    assert cli.canonical_uuid(uuid) == cli.canonical_uuid("GPU-" + uuid)
    assert cli.canonical_uuid(uuid) != cli.canonical_uuid("9cd80b3c-6b00-5a65-ae55-e79b351594d5")
    with pytest.raises(ValueError):
        cli.canonical_uuid("invalid")
    gpu = "4, GPU-example, NVIDIA GeForce RTX 3090, 24576, 18, 0, 27\n"
    monkeypatch.setattr(cli, "query", lambda args: gpu if "--query-gpu" in args[0] else "")
    assert cli.gpu_status(4)["physical_index"] == 4
    monkeypatch.setattr(
        cli,
        "query",
        lambda args: gpu if "--query-gpu" in args[0] else "GPU-example, 1234, python, 512\n",
    )
    with pytest.raises(RuntimeError, match="resource conflict"):
        cli.gpu_status(4)
    assert cli.gpu_status(4, own_pid=1234)["used_mib"] == 18
