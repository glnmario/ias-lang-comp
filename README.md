# Incremental Alternative Sampling (Private repo)

This repository contains the code for the experiments reported in:

**Incremental Alternative Sampling as a Window into the Temporal
and Representational Resolution of Prediction in Language Comprehension**. (June 2024). Mario Giulianelli, Sarenne Wallbridge, Ryan Cotterell, Raquel Fernández. *Preprint submitted to Journal of Memory and Language*.


## Instructions
**[0]** Install `requirements.txt`.

**[1]** Obtain estimates of surprisal and incremental information value. Below are the commands to run for the Aligned dataset; change the paths for Natural Stories.
```
python src/compute_surprisal.py \
    --dataset data/corpora/aligned/texts.csv \
    --output ddata/estimates/aligned/surprisal/gpt2-small_surprisal.csv \
    --aggregate_by_word \
    --return_tokens \
    --model_name_or_path gpt2 \
    --device cuda

python src/compute_incremental_information_value.py \
    --dataset data/corpora/aligned/texts.csv \
    --output data/estimates/aligned/iv_k50/gpt2-small_iv_n1.csv \
    --model_name_or_path gpt2 \
    --return_tokens \
    --layers "[0,1,2,3,4,5,6,7,8,9,10,11,12]" \
    --seq_len 1 \
    --n_sets 1 \
    --n_samples_per_set 50 \
    --forecast_horizons "[1,2,3,4,5,6,7,8,9,10]" \
    --seed 0 \
    --device cuda
```

**[2]** Preprocess corpora for the analysis: run `analysis/preprocess_aligned.ipynb` and `analysis/preprocess_naturalstories.ipynb`. This will create four output files:
```
analysis/preprocessed_corpora/aligned_preprocessed.csv  
analysis/preprocessed_corpora/aligned_preprocessed_normalised.csv
analysis/preprocessed_corpora/naturalstories_preprocessed.csv
analysis/preprocessed_corpora/naturalstories_preprocessed_normalised.csv
```   

**[3]** Run statistical tests: `analysis/statistical_tests.ipynb`. This will create four output files:
```
analysis/results/aligned_ols_against_baseline.csv
analysis/results/aligned_ols_against_surprisal.csv
analysis/results/naturalstories_ols_against_baseline.csv
analysis/results/naturalstories_ols_against_surprisal.csv
```
**Note**: We provide these precomputed dataframes in `analysis/results`.

**[4]** Analyse and visualise results: `analysis/hypothesis_testing.ipynb`. Plots are saved by default in `analysis/figures`.