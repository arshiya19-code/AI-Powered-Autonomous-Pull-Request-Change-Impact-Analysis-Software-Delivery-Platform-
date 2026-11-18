import numpy as np
import pandas as pd

from scripts.rehabilitate_apachejit_dataset import FEATURES, add_history, fit_transform


def test_history_uses_strictly_earlier_project_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "project": ["p", "p", "p", "q"],
            "author_date": [100, 100, 200, 50],
            "buggy": [1, 0, 1, 1],
        }
    )

    result = add_history(frame)

    assert result.loc[:1, "prior_project_commits"].tolist() == [0, 0]
    assert result.loc[2, "prior_project_commits"] == 2
    assert result.loc[2, "prior_project_positive_count"] == 1
    assert result.loc[2, "prior_project_negative_count"] == 1
    assert result.loc[3, "prior_project_commits"] == 0


def test_winsor_bounds_are_fitted_from_training_only() -> None:
    train = pd.DataFrame({name: [0.0, 1.0, 2.0, 3.0] for name in FEATURES})
    unseen = pd.DataFrame({name: [-100.0, 100.0] for name in FEATURES})

    transform = fit_transform("winsor_1_99", train, FEATURES)
    output = transform.apply(unseen)

    assert np.allclose(output.iloc[0].to_numpy(), transform.lower.to_numpy())
    assert np.allclose(output.iloc[1].to_numpy(), transform.upper.to_numpy())
