import os

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from test_tube import SlurmCluster, Experiment, HyperOptArgumentParser
from dask.distributed import Client

from dwd_dl import cfg
import dwd_dl.model as model
import dwd_dl.callbacks as callbacks
import dwd_dl.data_module as data_module
import dwd_dl as dl
from dwd_dl.cli import RadolanParser


def main(args):
    dl.cfg.initialize2(skip_download=True)
    exp = Experiment(
        name=args.test_tube_exp_name,
        save_dir=args.log_path,
        version=args.hpc_exp_number,
        autosave=False,
    )
    client = Client(processes=False)
    unet = model.UNetLitModel(**vars(args), timestamp_string=args.test_tube_exp_name)
    dm = data_module.RadolanDataModule(args.batch_size, args.workers, args.image_size)
    logger = TestTubeLogger(
        save_dir=args.log_path,
        name=args.test_tube_exp_name,
        create_git_tag=True,
        version=args.hpc_exp_number,
    )
    callbacks_list = callbacks.CallbacksList(args.test_tube_exp_name)

    if args.max_epochs is None:
        args.max_epochs = 100
    trainer = Trainer.from_argparse_args(args, logger=logger,
                                         flush_logs_every_n_steps=100, callbacks=list(callbacks_list))
    trainer.fit(unet, dm)
    checkpoint_path = cfg.CFG.create_checkpoint_path_with_name()
    trainer.save_checkpoint(checkpoint_path)
    dm.close()
    exp.close()
    client.close()


if __name__ == "__main__":
    dl.cfg.initialize2(skip_download=True)
    parser = HyperOptArgumentParser(conflict_handler='resolve')
    parser = RadolanParser.add_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = model.UNetLitModel.add_model_specific_args(parser)
    parser.add_argument('--test_tube_exp_name', default='my_test', type=str)
    parser.add_argument('--log_path', default=os.path.join(cfg.CFG.RADOLAN_ROOT, 'tt_logs'))
    parser.opt_list('--depth', options=[7, 6], tunable=True)
    args = parser.parse_args()

    cluster = SlurmCluster(
        hyperparam_optimizer=args,
        log_path='/home/fs71666/csaw2629/.out/download_%j.out',
        python_cmd='python',
        test_tube_exp_name=args.test_tube_exp_name
    )

    # let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
    cluster.notify_job_status(email='stefano.campostrini@studenti.unitn.it', on_done=True, on_fail=True)

    cluster.add_command('source activate py38')
    cluster.add_command('cd $HOME/dwd-dl-thesis/dwd_dl/')

    cluster.per_experiment_nb_cpus = 20
    cluster.per_experiment_nb_nodes = 10

    cluster.add_slurm_cmd(
        cmd='qos', value='mem_0096', comment='Quality of Service'
    )
    cluster.add_slurm_cmd(
        cmd='partition', value='mem_0096', comment='Partition'
    )

    cluster.job_time = '4:00:00'

    # run the models on the cluster
    cluster.optimize_parallel_cluster_cpu(main, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')
