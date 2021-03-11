# submit jobs

# dcc
for f in ../data/allsubj_creamA_wc_rest/s*.txt;
    do 
    echo "calculating dcc from ${f}"; 
    fsl_sub -T 10 -R 32 python correlations.py ${f}
done;
