import torch

from uwb_tracking.models import PaperResidualCNN, UncertaintyFusionNet
from uwb_tracking.training import student_t_nll


def test_paper_model_shape_and_official_channels():
    model = PaperResidualCNN(input_length=128).eval()
    out = model(torch.rand(4, 1, 128, 2))
    assert out["mean_index"].shape == (4,)
    assert model.stem[0].conv.out_channels == 8
    assert model.stem[4].conv.out_channels == 16
    assert model.residual[0].main[0].conv.out_channels == 32
    assert model.residual[1].main[0].conv.out_channels == 64
    assert model.residual[2].main[0].conv.out_channels == 128


def test_paper_spatial_shape_matches_matlab_graph():
    model = PaperResidualCNN(input_length=500).eval()
    x = torch.rand(2, 1, 500, 2)
    with torch.no_grad():
        stem = model.stem(x - model.input_mean)
        residual = model.residual(stem)
    assert stem.shape == (2, 16, 50, 1)
    assert residual.shape == (2, 128, 7, 1)


def test_proposed_model_outputs_positive_scale_and_gates():
    model = UncertaintyFusionNet().eval()
    out = model(torch.rand(4, 6, 128))
    assert out["mean_fraction"].shape == (4,)
    assert torch.all(out["scale_fraction"] > 0)
    assert torch.allclose(out["gate_cir"] + out["gate_var"], torch.ones(4), atol=1e-5)


def test_student_t_loss_prefers_correct_mean():
    target = torch.tensor([0.5])
    scale = torch.tensor([0.1])
    good = student_t_nll(target, torch.tensor([0.5]), scale).item()
    bad = student_t_nll(target, torch.tensor([0.9]), scale).item()
    assert good < bad
