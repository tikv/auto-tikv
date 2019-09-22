'''
Some code in this file is reused from OtterTune project under Apache License v2

Copyright (c) 2017-18, Carnegie Mellon University Database Group

@author: Bohan Zhang, Dana Van Aken

'''


from gpclass import *

def configuration_recommendation(target_data, runrec=None):
    print("running configuration recommendation...")
    if(target_data.num_previousamples<10 and runrec==None):                               #  give random recommendation on several rounds at first
        return gen_random_data(target_data)

    X_workload = target_data.new_knob_set
    X_columnlabels = target_data.knob_labels
    y_workload = target_data.new_metric_set
    y_columnlabels = target_data.metric_labels
    rowlabels_workload = target_data.new_rowlabels

    X_target = target_data.previous_knob_set
    y_target = target_data.previous_metric_set
    rowlabels_target = target_data.previous_rowlabels

    # Filter ys by current target objective metric
    target_objective = target_data.target_metric
    target_obj_idx = [i for i, cl in enumerate(y_columnlabels) if cl == target_objective]   #idx of target metric in y_columnlabels matrix

    lessisbetter = target_data.target_lessisbetter==1

    y_workload = y_workload[:, target_obj_idx]
    y_target = y_target[:, target_obj_idx]
    y_columnlabels = y_columnlabels[target_obj_idx]

    # Combine duplicate rows in the target/workload data (separately)
    #X_workload, y_workload, rowlabels_workload = combine_duplicate_rows(X_workload, y_workload, rowlabels_workload)
    #    print("remove duplicate: ", X_target, y_target, rowlabels_target)
    X_target, y_target, rowlabels_target = combine_duplicate_rows(X_target, y_target, rowlabels_target)
    #    print("remove duplicate: ", X_target, y_target, rowlabels_target)

    # Delete any rows that appear in both the workload data and the target
    # data from the workload data
    # dups_filter = np.ones(X_workload.shape[0], dtype=bool)
    # target_row_tups = [tuple(row) for row in X_target]
    # for i, row in enumerate(X_workload):
    #     if tuple(row) in target_row_tups:
    #         dups_filter[i] = False
    # X_workload = X_workload[dups_filter, :]
    # y_workload = y_workload[dups_filter, :]
    # rowlabels_workload = rowlabels_workload[dups_filter]

    # Combine target & workload Xs for preprocessing
    X_matrix = np.vstack([X_target, X_workload])

    # Dummy encode categorial variables
    categorical_info = dummy_encoder_helper(X_columnlabels)      #__INPUT__
    dummy_encoder = DummyEncoder(categorical_info['n_values'], categorical_info['categorical_features'], categorical_info['cat_columnlabels'], categorical_info['noncat_columnlabels'])
    X_matrix = dummy_encoder.fit_transform(X_matrix)

    # below two variables are needed for correctly determing max/min on dummies
    binary_index_set = set(categorical_info['binary_vars'])
    total_dummies = dummy_encoder.total_dummies()

    # Scale to N(0, 1)
    X_scaler = StandardScaler()
    X_scaler.fit(X_matrix)
    X_scaled = X_scaler.transform(X_matrix)
    #X_scaled = X_scaler.fit_transform(X_matrix)
    if y_target.shape[0] < 5:  # FIXME
        # FIXME (dva): if there are fewer than 5 target results so far
        # then scale the y values (metrics) using the workload's
        # y_scaler. I'm not sure if 5 is the right cutoff.
        y_target_scaler = None
        y_workload_scaler = StandardScaler()
        y_matrix = np.vstack([y_target, y_workload])
        y_scaled = y_workload_scaler.fit_transform(y_matrix)
    else:
        # FIXME (dva): otherwise try to compute a separate y_scaler for
        # the target and scale them separately.
        try:
            y_target_scaler = StandardScaler()
            y_workload_scaler = StandardScaler()
            y_target_scaled = y_target_scaler.fit_transform(y_target)
            y_workload_scaled = y_workload_scaler.fit_transform(y_workload)
            y_scaled = np.vstack([y_target_scaled, y_workload_scaled])
        except ValueError:
            y_target_scaler = None
            y_workload_scaler = StandardScaler()
            y_scaled = y_workload_scaler.fit_transform(y_target)

    # Set up constraint helper
    constraint_helper = ParamConstraintHelper(scaler=X_scaler, 
                                              encoder=dummy_encoder, 
                                              binary_vars=categorical_info['binary_vars'], 
                                              init_flip_prob=INIT_FLIP_PROB, 
                                              flip_prob_decay=FLIP_PROB_DECAY)    #__INPUT__

    # FIXME (dva): check if these are good values for the ridge
    # ridge = np.empty(X_scaled.shape[0])
    # ridge[:X_target.shape[0]] = 0.01
    # ridge[X_target.shape[0]:] = 0.1

    # FIXME: we should generate more samples and use a smarter sampling
    # technique
    num_samples = NUM_SAMPLES
    X_samples = np.empty((num_samples, X_scaled.shape[1]))
    X_min = np.empty(X_scaled.shape[1])
    X_max = np.empty(X_scaled.shape[1])

    X_mem = np.zeros([1, X_scaled.shape[1]])
    X_default = np.empty(X_scaled.shape[1])

    # Get default knob values
    for i, k_name in enumerate(X_columnlabels):
        X_default[i] = knob_set[k_name]['default']

    X_default_scaled = X_scaler.transform(X_default.reshape(1, X_default.shape[0]))[0]

    # Determine min/max for knob values
    for i in range(X_scaled.shape[1]):
        if i < total_dummies or i in binary_index_set:
            col_min = 0
            col_max = 1
        else:
            col_min = X_scaled[:, i].min()
            col_max = X_scaled[:, i].max()
            # Set min value to the default value
            # FIXME: support multiple methods can be selected by users
            #col_min = X_default_scaled[i]

        X_min[i] = col_min
        X_max[i] = col_max
        X_samples[:, i] = np.random.rand(num_samples) * (col_max - col_min) + col_min

    # Maximize the throughput, moreisbetter
    # Use gradient descent to minimize -throughput
    if not lessisbetter:
        y_scaled = -y_scaled

    q = queue.PriorityQueue()
    for x in range(0, y_scaled.shape[0]):
        q.put((y_scaled[x][0], x))

    i = 0
    while i < TOP_NUM_CONFIG:
        try:
            item = q.get_nowait()
            # Tensorflow get broken if we use the training data points as
            # starting points for GPRGD. We add a small bias for the
            # starting points. GPR_EPS default value is 0.001
            # if the starting point is X_max, we minus a small bias to
            # make sure it is within the range.
            dist = sum(np.square(X_max - X_scaled[item[1]]))
            if dist < 0.001:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] - abs(GPR_EPS)))
            else:
                X_samples = np.vstack((X_samples, X_scaled[item[1]] + abs(GPR_EPS)))
            i = i + 1
        except queue.Empty:
            break

    #X_samples=np.rint(X_samples)
    #X_samples=X_scaler.transform(X_samples)

    model = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
                  magnitude=DEFAULT_MAGNITUDE,
                  max_train_size=MAX_TRAIN_SIZE,
                  batch_size=BATCH_SIZE,
                  num_threads=NUM_THREADS,
                  learning_rate=DEFAULT_LEARNING_RATE,
                  epsilon=DEFAULT_EPSILON,
                  max_iter=MAX_ITER,
                  sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER,
                  mu_multiplier=DEFAULT_MU_MULTIPLIER)
    model.fit(X_scaled, y_scaled, X_min, X_max, ridge=DEFAULT_RIDGE)
    # print("constrains_min::::::::", X_min)
    # print("constrains_max::::::::", X_max)
    # print("train:::::::: ", X_scaled.shape, X_scaled, type(X_scaled[0][0]))
    # print("train:::::::: ", y_scaled.shape, y_scaled, type(y_scaled[0][0]))
    print("predict:::::::: ", X_samples.shape, X_scaler.inverse_transform(X_samples).astype(np.int16), type(X_samples[0][0]))
    res = model.predict(X_samples, constraint_helper=constraint_helper)

    best_config_idx = np.argmin(res.minl.ravel())
    best_config = res.minl_conf[best_config_idx, :]
    best_config = X_scaler.inverse_transform(best_config)
    print("rec:::::::", X_scaler.inverse_transform(res.minl_conf), res.minl)
    print('best_config==', best_config_idx, best_config)
    # Decode one-hot encoding into categorical knobs
    best_config = dummy_encoder.inverse_transform(best_config)

    # Although we have max/min limits in the GPRGD training session, it may
    # lose some precisions. e.g. 0.99..99 >= 1.0 may be True on the scaled data,
    # when we inversely transform the scaled data, the different becomes much larger
    # and cannot be ignored. Here we check the range on the original data
    # directly, and make sure the recommended config lies within the range
    X_min_inv = X_scaler.inverse_transform(X_min)
    X_max_inv = X_scaler.inverse_transform(X_max)
    best_config = np.minimum(best_config, X_max_inv)
    best_config = np.maximum(best_config, X_min_inv)

    best_config = np.rint(best_config)
    best_config = best_config.astype(np.int16)

    conf_map = {k: best_config[i] for i, k in enumerate(X_columnlabels)}
    print(conf_map)
    return conf_map


