# results

Model outputs and result tables.

- `deepseek*_seed<n>.json`: every suggestion of every method, per project and seed (the raw model outputs).
- `summary.csv`, `per_project.csv`, `stat_tests.csv`, `budget_tests.csv`, `reliability.csv`: written by `evaluate.py`.
- `judge/`: functional recall (`judge_summary.csv`, `judge_tests.csv`, `judge_per_project.csv`) and the cached judgments, written by `judge.py`.
- `judge_useful/`: usefulness (`useful_summary.csv`, `useful_tests.csv`, `useful_per_project.csv`) and the cached judgments, written by `judge_useful.py`.
- `threshold_check.csv`, `redundancy_check.csv`: samples for manual checks.

Run names: `deepseek` = Experiment 2 (zero-shot) at budget N; `_k2` = budget 2N; `_agent` = Critic Agent; `_fs` = Experiment 1 (few-shot).
