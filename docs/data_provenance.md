# Dataset provenance and limits

The local source is the Kaggle **Gemstones Images** dataset (87 class folders).
The benchmark subset follows Chow and Reyes-Aldasoro's 68-class cleaning and
fixed train/test selection. The expected benchmark has 2,042 training and 284
test images. The exact class list and source filenames are recorded in the
split artifacts and manifests.

The paper removed Garnet (named `Garnet Red` in the supplied raw folders)
because it overlaps several other labels, removed Moonstone because its images
mix several colours, and excluded classes with
fewer than 24 usable training images. Poorly segmented training images were
also discarded. The raw Kaggle folder alone does not encode all of those
decisions; therefore, class filtering alone is insufficient to reconstruct the
published training set.

For this local download, the 68 selected raw class folders contain 2,249 train
images. The reference `train_selected` manifest retains 2,042; the other 207
paths are recorded in `splits/excluded_training_images.csv`. The exact
per-image exclusion reason is not published, so the CSV records only that each
file is absent from the authors' selected training manifest.

The design document reports no licence on the Kaggle or source repository
pages it reviewed. Keep source images local for coursework, cite the dataset,
paper, and repository, and do not redistribute images or derived image
archives. Re-check the licence before any future sharing.

References:

- Chow, B.H.Y. and Reyes-Aldasoro, C.C. (2022), “Automatic Gemstone
  Classification Using Computer Vision”, *Minerals*, 12(1), 60.
  <https://doi.org/10.3390/min12010060>
- Reference implementation: <https://github.com/hybchow/gems>
- Dataset: <https://www.kaggle.com/datasets/lsind18/gemstones-images>

## Local audit record

The initial pHash/visual-descriptor pass covered all 2,326 selected images. At
the provisional thresholds (pHash distance at most 4 and embedding cosine at
least 0.99), it produced 51 candidate pairs, of which 22 passed both signals.
Two high-confidence pairs cross the train/test boundary; their test images are
listed in `results/audits/test_duplicates_to_exclude.csv`. The evaluator reports
both the full test score and a score excluding those flagged test images. The
candidate CSV and summary JSON are the numeric audit record. The review contact
sheet stays local because it contains dataset image pixels. The thresholds are
still marked for team review in the summary JSON; the evaluator's no-duplicate
score therefore follows the recorded provisional high-confidence rule.
