# Team integration contracts

Each package owns one arm directory (`src/g1`, `src/g2`, or `src/g3`). Common
data and scoring code lives in `src/gemstone_trends`; methods should depend on
those shared contracts instead of implementing their own split or metrics.

- Split: `splits/chow68.csv`; columns `path,class,split,fold,dup_group`.
- Class order: `splits/classes.txt`, fixed for every probability vector.
- Masks: `masks/<mode>/<split>/<class>/<stem>.png`, values 0/255 and original
  image dimensions.
- Predictions: `results/preds/<arm>_<model>_<mode>_s<seed>.csv` as described in
  `experiment_protocol.md`.
- Model adapter: each arm exposes `load(path)` and `predict(image)`, returning
  up to five `(class_name, probability)` pairs.

Do not edit another arm's folder without coordinating with its owner. Shared
contract changes belong in a reviewed change and must keep backward
compatibility where practical.
