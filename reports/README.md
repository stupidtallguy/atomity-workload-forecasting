# Reports and supporting evidence

Start with the [six-page final report](final_report.pdf). The [editable source](final_report.md)
contains the same first-person report by Salar Rahnama.

| Folder | Contents | Suggested starting point |
| --- | --- | --- |
| `analysis/` | Data checks, training targets, segment summaries, exploratory charts | [Data audit](analysis/data_audit.md) |
| `validation/` | Chronological baseline and CPU model comparisons, validation predictions and metrics | [Model comparison](validation/model_comparison.md) |
| `policy/` | Pre-holdout decision record, empirical headroom diagnostics and chronological checks | [Frozen method and policy](policy/method_freeze.md) |
| `forecast/` | Saved prediction-generation provenance and separate operational plan | [Prediction generation](forecast/prediction_generation.md) |
| `evaluation/` | Final holdout scores, actual-versus-predicted rows, policy outcomes and chart | [Holdout evaluation](evaluation/holdout_evaluation.md) |
| `figures/` | The two report-sized figures embedded in the final report | Referenced by the editable source |
| `verification/` | Submission and clean-environment verification records | [Submission checks](verification/submission_check.json) and [clean reproduction](verification/clean_environment.json) |

The required submission predictions are at [`../predictions.csv`](../predictions.csv).
The checked pre-holdout configuration is at [`../config/frozen_method.json`](../config/frozen_method.json).

The detailed CSVs are retained for inspection and reproducibility. They contain
small aggregate/validation outputs, not copies of the raw VM dataset. Prediction-generation
provenance records the state before scoring; the separate evaluation record
documents the later examination of holdout labels. Folder organization does
not change those historical records, their hashes, or the frozen forecasts.

Use the commands in the [project README](../README.md) to reproduce this layout.
