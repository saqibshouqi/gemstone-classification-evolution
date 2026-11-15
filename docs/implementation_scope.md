# Request scope and attached documents

## User requests

The first request was to create a durable project structure and complete Person
1's Data and Scoreboard work. A later request asks specifically to implement
the G2 anchor. This G2 change implements the 2022 method package and its
integration with the existing shared data and scoreboard contracts.

## Implementation design document

The design document supplies the research protocol and team contracts: the
2015/2022/2025 comparison, Chow's fixed 68-class data split, shared prediction
format, duplicate audit, metrics, and package boundaries. These are project
requirements used to shape the code; they are not separate messages directing
the assistant. This G2 change is limited to its anchor: the two Otsu mask
pipelines, the first RGB histogram plus LBP feature representation, classifier
grids, SMOTE within CV folds, and shared prediction output. The remaining
feature ablations are stretch work for a later step.

The G2 reproduction compares both the published per-class segmentation rule
and the label-free selector. The published rule depends on the known class;
those results are therefore marked as oracle diagnostics, reported separately,
and never exposed through inference. The label-free results are the leakage-free
operational comparison. Because this dataset contains raw images rather than
the paper's saved masked images, the published masks are reconstructed from
the documented Otsu recipe.

The core E3 G2 feature ablation is histogram-only, LBP-only, and their
concatenation. The full 33 feature-method grid is explicitly stretch work and
is not part of the required G2 package.

## Coursework brief

The coursework brief describes the assessed work: Task 5 demonstrates a
historical and a recent implementation, the report is limited to 20 A4 pages,
and the group presents a viva while students are assessed individually. The
brief also allows full GenAI use and asks for a record of prompts and use.
Those assessment requirements are documented here but do not expand this
request into implementing all three model arms or writing the report. The
current AI-use record is in docs/ai_usage_log.md.
