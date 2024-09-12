import os

dardel_login = "micpel@dardel.pdc.kth.se"
dardel_folder = "/cfs/klemming/projects/snic/hess_flow/RoughnessAmorphous/GmxRuns"

labels_n = ['06','08','10','12','14','16']
labels_a = ['02','04','06','08','10']

fn1 = "flow-Preparation-*.tar.gz"

for n in labels_n:
    for a in labels_a:

        tag = 'n'+n+'a'+a
        fn2 = 'flow-prep-'+tag+'.tar.gz'
        cmd = 'scp '+dardel_login+':'+dardel_folder+'/'+tag+'/Preparation/'+fn1+' ./'+fn2
        print(cmd)
        os.system(cmd)
