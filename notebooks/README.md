# Notebook policy

Use notebooks for exploratory work and visualization. Put reusable loaders,
metrics, and model code in `src/`; notebooks should import that code and use the
canonical split. Never use the test split to choose preprocessing or tune
hyperparameters.
