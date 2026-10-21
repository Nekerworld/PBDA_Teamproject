"""The policy tuning gap tool reconstructs the committed v7 selection from the validation files."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'report_tools'))
import policy_tuning_gap


def test_policy_tuning_gap_finds_the_repeat_selected_configurations():
    gap = policy_tuning_gap.fold_gap(HERE / 'audit' / 'results_v7' / 'F4_unseen')
    for family in ('GNN', 'ALS', 'BPR'):
        assert gap['repeat']['families'][family]['selected_rank'] == 1
        assert gap['novel']['families'][family]['best'] >= gap['novel']['families'][family]['selected']
    assert gap['novel']['families']['GNN']['configs'] == 36
