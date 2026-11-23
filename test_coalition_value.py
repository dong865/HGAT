import numpy as np
import torch


def test_candidate_features_are_invariant_to_member_order():
    from .coalition_value import build_candidate_features

    pursuer_embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    task_embeddings = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    pursuer_positions = np.array([
        [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [4.0, 0.0], [6.0, 0.0], [5.0, 1.0],
    ])
    task_positions = np.array([[0.0, 0.0], [5.0, 0.0]])
    assignment = np.array([0, 0, 0, 1, 1, 1])

    first = build_candidate_features(
        pursuer_embeddings, task_embeddings, pursuer_positions,
        task_positions, assignment,
    )
    permutation = np.array([2, 0, 1, 5, 3, 4])
    second = build_candidate_features(
        pursuer_embeddings[permutation], task_embeddings,
        pursuer_positions[permutation], task_positions,
        assignment[permutation],
    )
    torch.testing.assert_close(first, second)


def test_proxy_prefers_one_capture_capable_coalition_to_two_pairs():
    from .coalition_value import geometric_proxy_value

    pursuers = np.array([
        [0.7, 0.0], [-0.35, 0.61], [-0.35, -0.61], [5.7, 0.0],
    ])
    tasks = np.array([[0.0, 0.0], [5.0, 0.0]])
    three_one = geometric_proxy_value(
        pursuers, tasks, np.array([0, 0, 0, 1]), capture_size=3,
    )
    two_two = geometric_proxy_value(
        pursuers, tasks, np.array([0, 0, 1, 1]), capture_size=3,
    )
    assert three_one > two_two


def test_nonempty_candidate_generator_covers_requested_count_patterns():
    from .coalition_value import generate_candidate_assignments

    candidates = generate_candidate_assignments(4, 2)
    counts = {
        tuple(np.bincount(candidate, minlength=2))
        for candidate in candidates
    }
    assert counts == {(1, 3), (2, 2), (3, 1)}
    assert len(candidates) == 14


def test_pairwise_ranking_loss_rewards_correct_order():
    from .coalition_value import pairwise_ranking_loss

    labels = torch.tensor([3.0, 2.0, 1.0])
    correct = pairwise_ranking_loss(
        torch.tensor([3.0, 2.0, 1.0]), labels, margin=0.2,
    )
    reversed_order = pairwise_ranking_loss(
        torch.tensor([1.0, 2.0, 3.0]), labels, margin=0.2,
    )
    assert correct < reversed_order


def test_fit_value_head_learns_toy_candidate_ranking():
    from .coalition_value import CoalitionValueHead
    from .train_coalition_value import fit_value_head, ranking_accuracy

    torch.manual_seed(4)
    hidden_dim = 2
    # Two scenarios, three candidates each.  Feature zero is deliberately
    # monotonic with the label so a tiny head must be able to learn the order.
    features = torch.zeros(2, 3, 2, 2 * hidden_dim + 7)
    features[:, :, :, 0] = torch.tensor([0.0, 1.0, 2.0])[None, :, None]
    labels = torch.tensor([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
    model = CoalitionValueHead(hidden_dim=hidden_dim, width=16)

    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    fit_value_head(model, features, labels, epochs=120, lr=2e-2)
    after = ranking_accuracy(model, features, labels)

    assert after >= 0.99
    assert any(
        not torch.equal(before, after_parameter)
        for before, after_parameter in zip(before_parameters, model.parameters())
    )
