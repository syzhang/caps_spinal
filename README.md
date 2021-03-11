### dcc pipeline for spinal data

To use the pipeline, install requirements.txt in a new conda environment. 

Directory is organised as below:
```
.
+-- code (this repo)
+-- data
|   +-- allsubj_creamA_wc_rest
+-- output
```

`correlations.py` calculates dcc for individual input time series. Can be run on the cluster with `jobs.sh`.

`visualise.py` plots time binned connectivity matrices with dcc outputs from above.
