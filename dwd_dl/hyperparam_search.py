from test_tube.hpc import SlurmCluster

# hyperparameters is a test-tube hyper params object
hyperparams = args.parse()

# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/home/lv71491/stefanoc/project/.out/',
    python_cmd='python'
)

# let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1

# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(train, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')