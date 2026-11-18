"""
analyzer/ml/labels.py — Phase 8B Label Generation

Label definition:
    high_risk = 1  if any pr_reviews.state == 'CHANGES_REQUESTED'
    high_risk = 0  otherwise

This is a REVIEW-FRICTION / CHANGE-RISK PROXY.
It is NOT deployment-failure ground truth.
It is NOT CI failure ground truth.
AIDev v4 contains no CI/CD outcome data.

LEAKAGE PREVENTION:
    Review fields (state, body, user, submitted_at) are used ONLY to generate
    the label. They must NEVER appear in the feature vector passed to the model.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Leakage field registry
# ---------------------------------------------------------------------------

LEAKAGE_FIELDS: List[str] = [
    "merged_at",
    "closed_at",
    "state",
    "pr_reviews.state",
    "pr_reviews.submitted_at",
    "pr_reviews.user",
    "pr_reviews.body",
    "review_state",
    "review_submitted_at",
    "review_user",
    "review_body",
    "pr_state",
]

# Review state that maps to high_risk = 1
_HIGH_RISK_REVIEW_STATE: str = "CHANGES_REQUESTED"


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def assign_high_risk_label(review_states: Iterable[Optional[str]]) -> int:
    """
    Assign a binary risk label from a collection of PR review states.

    Args:
        review_states: Iterable of review state strings
                       (e.g. 'APPROVED', 'CHANGES_REQUESTED', 'COMMENTED', …).
                       None / empty strings are ignored.

    Returns:
        1 if any review has state == 'CHANGES_REQUESTED', else 0.

    This is a REVIEW-FRICTION proxy, not deployment-failure ground truth.
    """
    for state in review_states:
        if state and state.upper() == _HIGH_RISK_REVIEW_STATE:
            return 1
    return 0


def label_from_review_rows(review_rows: List[Dict[str, Any]]) -> int:
    """
    Assign label from a list of review row dicts (each with a 'state' key).

    Args:
        review_rows: List of dicts with at least a 'state' key.

    Returns:
        1 if any row has state == 'CHANGES_REQUESTED', else 0.
    """
    states = [row.get("state") for row in review_rows]
    return assign_high_risk_label(states)


# ---------------------------------------------------------------------------
# Leakage prevention
# ---------------------------------------------------------------------------


def validate_no_label_leakage(feature_dict: Dict[str, Any]) -> List[str]:
    """
    Validate that a feature dictionary contains no leakage fields.

    Args:
        feature_dict: The dictionary of feature name → value to inspect.

    Returns:
        List of leakage field names found in the dictionary (empty = no leakage).
    """
    found: List[str] = []
    feature_keys_lower = {k.lower() for k in feature_dict}
    for lf in LEAKAGE_FIELDS:
        if lf.lower() in feature_keys_lower:
            found.append(lf)
        # Also check for partial substring matches to catch aliased fields
        for key in feature_dict:
            if lf.lower() in key.lower() and lf not in found:
                found.append(lf)
                break
    return found


def assert_no_label_leakage(feature_dict: Dict[str, Any]) -> None:
    """
    Raise ValueError if any leakage field is present in feature_dict.

    Args:
        feature_dict: Feature dictionary to check.

    Raises:
        ValueError: If any leakage field is detected.
    """
    leaked = validate_no_label_leakage(feature_dict)
    if leaked:
        raise ValueError(
            f"LEAKAGE DETECTED: the following fields must NEVER be model inputs: "
            f"{leaked}\n"
            "Review information is used only for label construction."
        )


# ---------------------------------------------------------------------------
# Class balance reporting
# ---------------------------------------------------------------------------


def compute_class_balance(labels: List[int]) -> Dict[str, Any]:
    """
    Compute class balance statistics for a label vector.

    Args:
        labels: List of 0/1 integer labels.

    Returns:
        Dict with counts, fractions, and suggested class weight ratio.
    """
    n = len(labels)
    if n == 0:
        return {"n": 0, "n_positive": 0, "n_negative": 0, "pos_fraction": 0.0,
                "neg_fraction": 0.0, "imbalance_ratio": 0.0, "suggested_weight_ratio": 1.0}

    n_pos = sum(labels)
    n_neg = n - n_pos
    pos_frac = n_pos / n
    neg_frac = n_neg / n
    # Suggested weight for positive class = n_neg / n_pos (handles imbalance)
    suggested_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return {
        "n": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pos_fraction": round(pos_frac, 4),
        "neg_fraction": round(neg_frac, 4),
        "imbalance_ratio": round(n_neg / max(n_pos, 1), 4),
        "suggested_weight_ratio": round(suggested_weight, 4),
    }
