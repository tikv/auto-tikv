from controller import read_metric, read_knob, set_knob, knob_set, init_knobs, load_workload, run_workload, calc_metric, restart_db
from gpmodel import configuration_recommendation
from datamodel import GPDataSet
from settings import tikv_ip, tikv_port, target_knob_set, target_metric_name, wl_metrics, wltype, loadtype
import numpy as np
import time

if __name__ == '__main__':
    ds = GPDataSet()
    Round=200
    init_knobs()
    metric_list=wl_metrics[wltype]
    ds.initdataset(metric_list)
    num_knobs = len(target_knob_set)
    num_metrics = len(metric_list)


    KEY = str(time.time())
    while(Round>0):
        print("################## start a new Round ##################")
        rec = configuration_recommendation(ds)
        knob_cache = {}
        for x in rec.keys():
            set_knob(x, rec[x])
            knob_cache[x] = rec[x]

        print("Round: ", Round, rec)

        restart_db()
        lres = load_workload(loadtype)
        print(lres)
        if("_ERROR" in lres):
            print("load workload error")
            exit()

        new_knob_set = np.zeros([1, num_knobs])
        new_metric_before = np.zeros([1, num_metrics])
        new_metric_after = np.zeros([1, num_metrics])

        for i,x in enumerate(metric_list):
            new_metric_before[0][i] = read_metric(x)

        for i,x in enumerate(target_knob_set):
            new_knob_set[0][i] = read_knob(x, knob_cache)

        rres = run_workload(wltype)
        print(rres)
        if("_ERROR" in rres):
            print("run workload error")
            exit()

        for i,x in enumerate(metric_list):
            new_metric_after[0][i] = read_metric(x, rres)

        new_metric = calc_metric(new_metric_after, new_metric_before, metric_list)

        #print(new_metric,metric_list)

        ds.add_new_data(new_knob_set, new_metric)

        import pickle
        fp = "ds_"+KEY+"_"+str(Round)+"_.pkl"
        with open(fp, "wb") as f:
            pickle.dump(ds, f)

        ds.printdata()

        ds.merge_new_data()

        Round-=1

