# Local split manifests

`chow68_train_selected.txt` and `chow68_test_selected.txt` contain one
`Class Name/filename.jpg` path per line from the selected folders in the
Chow and Reyes-Aldasoro reference repository. They identify files only; image
data stays in the local Kaggle download and is excluded from version control.

The manifests are the bridge between the supplied 87-class Kaggle folder and
the paper's cleaned 68-class, 2,042-train / 284-test benchmark. Do not replace
them with a random split. Rebuild the canonical CSV only when intentionally
updating the data source, then record that change in `docs/experiment_protocol.md`.
