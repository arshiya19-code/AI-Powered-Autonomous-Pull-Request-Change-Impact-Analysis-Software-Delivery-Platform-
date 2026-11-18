from tests.test_phase7c_candidate_audit import candidate
from dataset.phase7e_selection import select_candidates
from dataset.phase7e_splits import external_network_split


def test_selection_is_deterministic_and_never_balances() -> None:
    rows = [candidate(str(i), network=f"n{i}") for i in range(100)]
    left = select_candidates(rows, requested_rows=100)
    right = select_candidates(rows, requested_rows=100)
    assert [x.sample.sample_id for x in left.selected] == [x.sample.sample_id for x in right.selected]
    assert left.repository_cap == 2


def test_external_split_has_no_network_overlap() -> None:
    split = external_network_split([candidate("1", network="a"), candidate("2", network="b")], test_networks={"b"})
    assert [x.sample.sample_id for x in split.train] == ["1"]
    assert [x.sample.sample_id for x in split.test] == ["2"]
