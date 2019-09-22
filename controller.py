import sys
import os
from settings import tikv_ip, tikv_port, tikv_pd_ip, ycsb_port, ansibledir, deploydir
import psutil
import time
import numpy as np
from ruamel import yaml

#MEM_MAX = psutil.virtual_memory().total
MEM_MAX = 0.8*32*1024*1024*1024                 # memory size of tikv node, not current PC


#------------------knob controller------------------

# disable_auto_compactions
def set_disable_auto_compactions(ip, port, val):
    cmd="./tikv-ctl --host "+ip+":"+port+" modify-tikv-config -m kvdb -n default.disable_auto_compactions -v "+str(val)
    res=os.popen(cmd).read()                        # will return "success"
    return(res)

knob_set=\
    {
    "rocksdb.defaultcf.write-buffer-size":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 64,                           # if type==int, indicate min possible value
            "maxval": 1024,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 64                           # default value
        },
    "rocksdb.defaultcf.max-bytes-for-level-base":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 512,                          # if type==int, indicate min possible value
            "maxval": 4096,                         # if type==int, indicate max possible value
            "enumval": [],                          # if type==enum, list all valid values
            "type": "int",                          # int / enum
            "default": 512                            # default value
        },
    "rocksdb.defaultcf.target-file-size-base":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": [8,16,32,64,128],            # if type==enum, list all valid values
            "type": "enum",                         # int / enum
            "default": 8                            # default value
        },
    "rocksdb.defaultcf.disable-auto-compactions":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": ['false', 'true'],           # if type==enum, list all valid values
            "type": "bool",                         # int / enum
            "default": 0                            # default value
        },
    "rocksdb.defaultcf.block-size":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": [4, 8, 16, 32, 64],          # if type==enum, list all valid values
            "type": "enum",                         # int / enum
            "default": 0                            # default value
        },
    "rocksdb.defaultcf.bloom-filter-bits-per-key":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": [5,10,15,20],                # if type==enum, list all valid values
            "type": "enum",                         # int / enum
            "default": 0                            # default value
        },
    "rocksdb.writecf.bloom-filter-bits-per-key":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": [5,10,15,20],                # if type==enum, list all valid values
            "type": "enum",                         # int / enum
            "default": 0                            # default value
        },
    "rocksdb.writecf.optimize-filters-for-hits":
        {
            "changebyyml": True,
            "set_func": None,
            "minval": 0,                            # if type==int, indicate min possible value
            "maxval": 0,                            # if type==int, indicate max possible value
            "enumval": ['false', 'true'],           # if type==enum, list all valid values
            "type": "bool",                         # int / enum
            "default": 0                            # default value
        },
    }


#------------------metric controller------------------

def read_write_throughput(ip, port):
    return(0)           # DEPRECATED FUNCTION: throughput is instant and could be read from go-ycsb. No need to read in this function

def read_write_latency(ip, port):
    return(0)           # DEPRECATED FUNCTION: latency is instant and could be read from go-ycsb. No need to read in this function

def read_get_throughput(ip, port):
    return(0)           # DEPRECATED FUNCTION: throughput is instant and could be read from go-ycsb. No need to read in this function

def read_get_latency(ip, port):
    return(0)           # DEPRECATED FUNCTION: latency is instant and could be read from go-ycsb. No need to read in this function

def read_scan_throughput(ip, port):
    return(0)           # DEPRECATED FUNCTION: throughput is instant and could be read from go-ycsb. No need to read in this function

def read_scan_latency(ip, port):
    return(0)           # DEPRECATED FUNCTION: latency is instant and could be read from go-ycsb. No need to read in this function

def read_store_size(ip, port):
    cmd='./tikv-ctl --host '+ip+':'+port+' metrics'
    res=os.popen(cmd).read()
    reslist=res.split("\n")
    ans0 =0
    for rl in reslist:
        if ('tikv_engine_size_bytes{db="kv",type="default"}' in rl):
            ans0 = int(rl.split(' ')[1])
            break
    return(ans0)

def read_compaction_cpu(ip, port):
    cmd='./tikv-ctl --host '+ip+':'+port+' metrics'
    res=os.popen(cmd).read()
    reslist=res.split("\n")
    ans=0
    ans1=0
    for rl in reslist:
        if ('tikv_thread_cpu_seconds_total{name="rocksdb:low' in rl):
            ans1 = float(rl.split(' ')[1])
            ans+=ans1
    return(ans)

metric_set=\
    {"write_throughput":
         {
         "read_func": read_write_throughput,
         "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #incremental
         },
    "write_latency":
        {
         "read_func": read_write_latency,
         "lessisbetter": 1,                    # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #instant
        },
    "get_throughput":
        {
         "read_func": read_get_throughput,
         "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #incremental
        },
    "get_latency":
        {
         "read_func": read_get_latency,
         "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #instant
        },
    "scan_throughput":
        {
         "read_func": read_scan_throughput,
         "lessisbetter": 0,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #incremental
        },
    "scan_latency":
        {
         "read_func": read_scan_latency,
         "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #instant
        },
    "store_size":
        {
         "read_func": read_store_size,
         "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
         "calc": "ins",                       #instant
        },
    "compaction_cpu":
        {
         "read_func": read_compaction_cpu,
         "lessisbetter": 1,                   # whether less value of this metric is better(1: yes)
         "calc": "inc",                       #incremental
        },
    }


#------------------workload controller------------------

def run_workload(wl_type):
    #./go-ycsb run tikv -P ./workloads/smallpntlookup -p tikv.pd=192.168.1.130:2379
    cmd="./go-ycsb run tikv -P ./workloads/"+wl_type+" -p tikv.pd="+tikv_pd_ip+':'+ycsb_port+" --threads=512"
    print(cmd)
    res=os.popen(cmd).read()
    return(res)

def load_workload(wl_type):
    #./go-ycsb load tikv -P ./workloads/smallpntlookup -p tikv.pd=192.168.1.130:2379
    # cmd="./tikv-ctl --host "+tikv_ip+":"+tikv_port+" modify-tikv-config -m kvdb -n default.disable_auto_compactions -v 1"
    # tmp=os.popen(cmd).read()                        # will return "success"
    cmd="./go-ycsb load tikv -P ./workloads/"+wl_type+" -p tikv.pd="+tikv_pd_ip+':'+ycsb_port+" --threads=512"
    print(cmd)
    res=os.popen(cmd).read()
    # cmd="./tikv-ctl --host "+tikv_ip+":"+tikv_port+" compact -d kv --threads=512"
    # tmp=os.popen(cmd).read()                        # will return "success"
    return(res)


#------------------common functions------------------

def set_tikvyml(knob_sessname, knob_val):
    knob_sess=knob_sessname.split('.')[0:-1]
    knob_name=knob_sessname.split('.')[-1]
    ymldir=os.path.join(ansibledir,"conf","tikv_newcfg.yml")
    tmpdir=os.path.join(ansibledir,"conf","tikv.yml")
    tmpf=open(tmpdir)
    tmpcontent=yaml.load(tmpf, Loader=yaml.RoundTripLoader)
    if(knob_set[knob_sessname]['type']=='enum'):
        idx=knob_val
        knob_val=knob_set[knob_sessname]['enumval'][idx]
    if(knob_set[knob_sessname]['type']=='bool'):
        if(knob_val==0):
            knob_val=False
        else:
            knob_val=True
    if(knob_name=='block-size'):
        knob_val=str(knob_val)+"KB"
    if(knob_name=='write-buffer-size' or knob_name=='max-bytes-for-level-base' or knob_name=='target-file-size-base'):
        knob_val=str(knob_val)+"MB"
    if(knob_name in tmpcontent[knob_sess[0]][knob_sess[1]]):        # TODO: only support 2 level of knob_sess currently
        tmpcontent[knob_sess[0]][knob_sess[1]][knob_name]=knob_val
    else:
        return('failed')
    print("set_tikvyml:: ",knob_sessname, knob_sess, knob_name, knob_val)
    ymlf=open(ymldir, 'w')
    yaml.dump(tmpcontent, ymlf, Dumper=yaml.RoundTripDumper)
    os.popen("rm "+tmpdir+" && "+"mv "+ymldir+" "+tmpdir)
    time.sleep(0.5)
    return('success')

def set_knob(knob_name, knob_val):
    changebyyml=knob_set[knob_name]["changebyyml"]
    if(changebyyml):
        res=set_tikvyml(knob_name, knob_val)
    else:
        func=knob_set[knob_name]["set_func"]
        res=func(tikv_ip, tikv_port, knob_val)
    return res

def read_knob(knob_name, knob_cache):
    res=knob_cache[knob_name]
    return res

def read_metric(metric_name, rres=None):
    if(rres!=None):
        rl=rres.split('\n')
        rl.reverse()
        if(metric_name=="write_latency"):
            i=0
            while((not rl[i].startswith('UPDATE ')) and (not rl[i].startswith('INSERT '))):
                i+=1
            dat=rl[i][rl[i].find("Avg(us):") + 9:].split(",")[0]
            dat=int(dat)
            return(dat)
        elif(metric_name=="get_latency"):
            i=0
            while(not rl[i].startswith('READ ')):
                i+=1
            dat=rl[i][rl[i].find("Avg(us):") + 9:].split(",")[0]
            dat=int(dat)
            return(dat)
        elif(metric_name=="scan_latency"):
            i=0
            while(not rl[i].startswith('SCAN ')):
                i+=1
            dat=rl[i][rl[i].find("Avg(us):") + 9:].split(",")[0]
            dat=int(dat)
            return(dat)
        elif(metric_name=="write_throughput"):
            i=0
            while((not rl[i].startswith('UPDATE ')) and (not rl[i].startswith('INSERT '))):
                i+=1
            dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
            dat=float(dat)
            return(dat)
        elif(metric_name=="get_throughput"):
            i=0
            while(not rl[i].startswith('READ ')):
                i+=1
            dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
            dat=float(dat)
            return(dat)
        elif(metric_name=="scan_throughput"):
            i=0
            while(not rl[i].startswith('SCAN ')):
                i+=1
            dat=rl[i][rl[i].find("OPS:") + 5:].split(",")[0]
            dat=float(dat)
            return(dat)
    func=metric_set[metric_name]["read_func"]
    res=func(tikv_ip, tikv_port)
    return res

def init_knobs():
    # if there are knobs whose range is related to PC memory size, initialize them here
    pass

def calc_metric(metric_after, metric_before, metric_list):
    num_metrics = len(metric_list)
    new_metric = np.zeros([1, num_metrics])
    for i, x in enumerate(metric_list):
        if(metric_set[x]["calc"]=="inc"):
            new_metric[0][i]=metric_after[0][i]-metric_before[0][i]
        elif(metric_set[x]["calc"]=="ins"):
            new_metric[0][i]=metric_after[0][i]
    return(new_metric)

def restart_db():
    #cmd="cd /home/tidb/tidb-ansible/ && ansible-playbook unsafe_cleanup_data.yml"
    dircmd="cd "+ ansibledir + " && "
    clrcmd="ansible-playbook unsafe_cleanup_data.yml"
    depcmd="ansible-playbook deploy.yml"
    runcmd="ansible-playbook start.yml"
    ntpcmd="ansible-playbook -i hosts.ini deploy_ntp.yml -u tidb -b"   #need sleep 10s after ntpcmd
    print("-------------------------------------------------------")
    clrres = os.popen(dircmd+clrcmd).read()
    if("Congrats! All goes well" in clrres):
        print("unsafe_cleanup_data finished, res == "+clrres.split('\n')[-2])
    else:
        print(clrres)
        print("unsafe_cleanup_data failed")
        exit()
    print("-------------------------------------------------------")
    ntpres = os.popen(dircmd + ntpcmd).read()
    time.sleep(10)
    if ("Congrats! All goes well" in ntpres):
        print("set ntp finished, res == " + ntpres.split('\n')[-2])
    else:
        print(ntpres)
        print("set ntp failed")
        exit()
    print("-------------------------------------------------------")
    depres = os.popen(dircmd + depcmd).read()
    if ("Congrats! All goes well" in depres):
        print("deploy finished, res == "+depres.split('\n')[-2])
    else:
        print(depres)
        print("deploy failed")
        exit()
    print("-------------------------------------------------------")
    runres = os.popen(dircmd + runcmd).read()
    if ("Congrats! All goes well" in runres):
        print("start finished, res == "+runres.split('\n')[-2])
    else:
        print(runres)
        print("start failed")
        exit()
    print("-------------------------------------------------------")


