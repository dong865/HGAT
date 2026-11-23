import numpy as np
import torch

from . import v3_demand_distillation
from .coalition_value import generate_candidate_assignments
from .v3_demand_distillation import (
    NormalizedDemandHead,
    ScaleAwareCoalitionValueHead,
    build_demand_geometry,
    build_scale_aware_candidate_features,
    capacities_from_demand,
    demand_target_from_values,
    fit_demand_student,
    fit_scale_aware_teacher,
)


def test_scale_aware_features_distinguish_two_of_four_from_three_of_six():
    hidden = 2
    task_embeddings = torch.zeros(2, hidden)
    task_positions = np.array([[0.0, 0.0], [5.0, 0.0]])
    f4 = build_scale_aware_candidate_features(
        torch.zeros(4, hidden), task_embeddings,
        np.array([[-1, 0], [1, 0], [0, 1], [5, 0]], dtype=float),
        task_positions, np.array([0, 0, 1, 1]), capture_size=3,
    )
    f6 = build_scale_aware_candidate_features(
        torch.zeros(6, hidden), task_embeddings,
        np.array([[-1, 0], [1, 0], [0, 1], [4, 0], [5, 1], [6, 0]], dtype=float),
        task_positions, np.array([0, 0, 0, 1, 1, 1]), capture_size=3,
    )

    # First scale-aware suffix value is the absolute coalition size.
    assert f4[0, -6].item() == 2.0
    assert f6[0, -6].item() == 3.0
    # Resource sufficiency also differs: 4/(2*3) vs 6/(2*3).
    assert f4[0, -1].item() < f6[0, -1].item()


def test_teacher_three_plus_one_values_create_asymmetric_demand_target():
    candidates = np.stack(generate_candidate_assignments(4, 2))
    values = torch.full((len(candidates),), -10.0)
    index = np.flatnonzero(np.all(candidates == [0, 0, 0, 1], axis=1))[0]
    values[index] = 10.0

    target = demand_target_from_values(values, candidates, 2, temperature=0.05)

    assert target[0] > 0.999
    assert target[1] < 0.001


def test_best_mode_target_removes_candidate_multiplicity_bias():
    candidates = np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0],
    ])
    # 3+1 has the larger mean value despite 2+2 having more candidates.
    values = torch.tensor([5.0, 5.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

    assert hasattr(v3_demand_distillation, "best_mode_demand_target")
    target, mode = v3_demand_distillation.best_mode_demand_target(
        values, candidates, num_agents=4, num_tasks=2,
    )

    assert mode == (3, 1)
    assert torch.allclose(target, torch.tensor([0.75, 0.25]))


def test_normalized_demand_head_and_capacity_obey_resource_contract():
    head = NormalizedDemandHead(hidden_dim=4)
    embeddings = torch.randn(3, 4)
    mask = torch.tensor([True, True, False])

    demand = head(embeddings, mask, num_agents=4, capture_size=3)
    capacity = capacities_from_demand(demand, mask, num_agents=4)

    assert torch.allclose(demand[:2].sum(), torch.tensor(1.0), atol=1e-6)
    assert demand[2].item() == 0.0
    assert torch.all(capacity[:2] >= 1.0)
    assert capacity[2].item() == 0.0
    assert torch.allclose(capacity.sum(), torch.tensor(4.0), atol=1e-6)


def test_demand_geometry_is_member_permutation_invariant():
    pursuers = np.array([[-1, 0], [1, 0], [0, 1], [5, 0]], dtype=float)
    tasks = np.array([[0, 0], [5, 0]], dtype=float)

    first = build_demand_geometry(pursuers, tasks, map_size=8.0)
    second = build_demand_geometry(pursuers[[2, 0, 3, 1]], tasks, map_size=8.0)

    assert first.shape == (2, 6)
    assert torch.allclose(first, second)


def test_teacher_and_student_fit_functions_update_only_supplied_models():
    torch.manual_seed(4)
    teacher = ScaleAwareCoalitionValueHead(hidden_dim=2, width=8)
    teacher_features = torch.randn(4, 3, 2 * 2 + 7 + 6)
    teacher_labels = torch.tensor([0.0, 1.0, 2.0, 3.0])
    teacher_before = [p.detach().clone() for p in teacher.parameters()]

    teacher_metrics = fit_scale_aware_teacher(
        teacher, teacher_features, teacher_labels, epochs=5, lr=1e-2,
    )

    assert np.isfinite(teacher_metrics["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(
        teacher_before, teacher.parameters()
    ))

    student = NormalizedDemandHead(hidden_dim=2, width=8)
    task_embeddings = torch.randn(4, 2, 2)
    targets = torch.tensor([[0.99, 0.01], [0.01, 0.99],
                            [0.95, 0.05], [0.05, 0.95]])
    student_before = [p.detach().clone() for p in student.parameters()]

    student_metrics = fit_demand_student(
        student, task_embeddings, targets, num_agents=4,
        epochs=10, lr=1e-2,
    )

    assert np.isfinite(student_metrics["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(
        student_before, student.parameters()
    ))
