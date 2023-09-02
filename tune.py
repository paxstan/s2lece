import torch
import logging
from functools import partial
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
# from train import TrainSleceNet
# from models.model import SleceNet
# from input_pipeline.dataloader import threaded_loader


class TuneS2leceNet:
    def __init__(self, config, dataloader, test_dataloader, run_paths, iscuda, device,
                 max_num_epochs=5, num_samples=10):
        self.config = config
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.run_paths = run_paths
        self.config["learning_rate"] = tune.loguniform(1e-4, 1e-1)
        # self.iters = tune.choice([i for i in range(1, 5)])
        self.iters = 5
        self.scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2,
        )
        self.iscuda = iscuda
        self.device = device
        self.max_num_epochs = max_num_epochs
        self.num_samples = num_samples
        ray.init(num_cpus=10, num_gpus=1)

    def __call__(self):
        tune_result = tune.run(
            partial(self.tune_s2lecenet,
                    device=self.device, iscuda=self.iscuda, iters=self.iters, run_paths=self.run_paths,
                    dataloader=self.dataloader, test_dataloader=self.test_dataloader),
            resources_per_trial={"cpu": 10, "gpu": 1},
            config=self.config,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
        )

        best_trial = tune_result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        print(f"Best trial final train loss: {best_trial.last_result['train loss']}")

        # Get a dataframe for analyzing trial results.
        df = tune_result.dataframe()
        df.to_csv("/home/paxstan/Documents/research_project/code/runs/tune_result.csv")

    @staticmethod
    def tune_s2lecenet(config, device, iscuda, iters, dataloader, test_dataloader, run_paths):
        net = SleceNet(config, device, iters=iters).to(device)
        loader = threaded_loader(dataloader, batch_size=4, iscuda=iscuda, threads=1)
        test_loader = threaded_loader(test_dataloader, batch_size=4, iscuda=iscuda, threads=1)
        train = TrainSleceNet(net=net, dataloader=loader, test_dataloader=test_loader, config=config,
                              run_paths=run_paths, is_cuda=iscuda)
        for result in train.train_slecenet():
            tune.report(loss=result["val loss"], train_loss=result["train loss"])
            # session.report(
            #     {"train loss": result["train loss"], "loss": result["val loss"]})


# config = {
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "iter": tune.choice([i for i in range(1, 11)])
# }

# Load the results of a Ray Tune experiment
analysis = tune.analysis.ExperimentAnalysis("/home/paxstan/ray_results/tune_s2lecenet_2023-07-08_23-28-39")

# Access various analysis functions and properties
# For example, you can retrieve the best trial:
best_trial = analysis.get_best_trial(metric="loss", mode="min")

# You can also access trial results, configurations, and other metrics:
trial_results = analysis.trial_dataframes
configurations = analysis.get_all_configs()
all_metrics = analysis.dataframe()

# Visualize the analysis results
all_metrics.to_csv("/home/paxstan/Documents/research_project/code/runs/tune_result.csv")
