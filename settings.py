#info about tikv
tikv_ip="192.168.1.104"
tikv_port="20160"
tikv_pd_ip="192.168.1.104"
ycsb_port="2379"

# workloads and their related performance metrics
wl_metrics={
    "writeheavy":["write_throughput","write_latency","store_size","compaction_cpu"],        #UPDATE
    "pntlookup40": ["get_throughput","get_latency","store_size","compaction_cpu"],          #READ
    "pntlookup80": ["get_throughput","get_latency","store_size","compaction_cpu"],          #READ
    "longscan":  ["scan_throughput","scan_latency","store_size","compaction_cpu"],          #SCAN
    "shortscan": ["scan_throughput","scan_latency","store_size","compaction_cpu"],          #SCAN
    "smallpntlookup": ["get_throughput","get_latency","store_size","compaction_cpu"],       #READ
}
# workload to be load
loadtype = "shortscan"
# workload to be run
wltype = "shortscan"

# only 1 target metric to be optimized
target_metric_name="scan_latency"

# several knobs to be tuned
target_knob_set=['rocksdb.writecf.bloom-filter-bits-per-key',
                 'rocksdb.defaultcf.bloom-filter-bits-per-key',
                 'rocksdb.writecf.optimize-filters-for-hits',
                 'rocksdb.defaultcf.block-size',
                 'rocksdb.defaultcf.disable-auto-compactions']
#target_knob_set=['disable-auto-compactions', 'optimize-filters-for-hits', 'write-buffer-size', 'block-size', 'max-bytes-for-level-base']

ansibledir="/home/tidb/tidb-ansible/"
deploydir="/home/tidb/deploy/"
