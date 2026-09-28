"""Codabench entry point for the self-perturbation method."""

from selfperturb_infer import predict_robustness


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    """Predict one native bool per problem, in the order received."""
    return predict_robustness(model_id, problems)
