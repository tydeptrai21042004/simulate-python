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


def test_lite_model_is_physically_small_and_keeps_uncertainty_interface():
    from uwb_tracking.models import LiteArchitecture, LiteUncertaintyFusionNet

    model = LiteUncertaintyFusionNet(arch=LiteArchitecture((8, 12, 16), 12, 24)).eval()
    out = model(torch.rand(6, 6, 176))
    assert sum(p.numel() for p in model.parameters()) == 5167
    assert out["mean_fraction"].shape == (6,)
    assert torch.all(out["scale_fraction"] > 0)
    assert torch.allclose(out["gate_cir"] + out["gate_var"], torch.ones(6), atol=1e-5)


def test_structured_ticket_rewinds_selected_channels_and_shrinks_model():
    import copy
    from uwb_tracking.models import (
        LiteArchitecture,
        LiteUncertaintyFusionNet,
        build_rewound_structured_ticket,
    )

    torch.manual_seed(7)
    source_arch = LiteArchitecture((10, 14, 18), 12, 24)
    target_arch = LiteArchitecture((6, 10, 12), 12, 24)
    supernet = LiteUncertaintyFusionNet(arch=source_arch)
    initial = copy.deepcopy(supernet.state_dict())
    with torch.no_grad():
        # Make ranking deterministic and different across channels.
        for idx in range(supernet.cir_encoder.conv1.weight.shape[0]):
            supernet.cir_encoder.conv1.weight[idx].fill_(float(idx + 1))
    ticket, selection = build_rewound_structured_ticket(supernet, initial, target_arch)
    assert ticket.cir_encoder.conv1.out_channels == 6
    assert ticket.cir_encoder.conv3.out_channels == 12
    assert sum(p.numel() for p in ticket.parameters()) < sum(p.numel() for p in supernet.parameters())
    expected = initial["cir_encoder.conv1.weight"][selection.cir_c1]
    assert torch.allclose(ticket.cir_encoder.conv1.weight, expected)
    out = ticket(torch.rand(3, 6, 176))
    assert out["mean_fraction"].shape == (3,)
