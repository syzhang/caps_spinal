# submit jobs

# dcc different conditions
# for f in /vols/Data/pain/SpinalCaps2018/suyi_collab/allsubj_creamA_wc_ramp/s*.txt;
# for f in /vols/Data/pain/SpinalCaps2018/suyi_collab/allsubj_creamA_wc_rest/s*.txt;
# for f in /vols/Data/pain/SpinalCaps2018/suyi_collab/allsubj_creamB_wc_ramp/s*.txt;
for f in /vols/Data/pain/SpinalCaps2018/suyi_collab/allsubj_creamB_wc_rest/s*.txt;
    do 
    echo "calculating dcc from ${f}"; 
    fsl_sub -T 10 -R 32 python correlations.py ${f}
done;