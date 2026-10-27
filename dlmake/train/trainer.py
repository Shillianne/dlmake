from dlmake import Transformer, BaseDataLoader, Benchmark, Config
# import multiprocessing as mp
import torch.multiprocessing as mp
from .runner import Runner
from .worker import Worker
from .utils import Logger
from typing import Type
import subprocess
import requests
import torch
import io
import os


stats = [
    "loss",
    "acc",
    "test_loss",
    "test_acc",
    "val_loss",
    "val_acc",
    "norm",
    "tkps",
    "mfu",
    "vram_usage",
    "epoch",
    "eta",
]


class Trainer:
    def __init__(
        self,
        model: Type[Transformer],
        dataloader: Type[BaseDataLoader],
        config_path: str,
        datapath: str,
        savepath: str,
        benchmarks: list[Benchmark],
        device: str,
        **kwargs
    ):
        """
        Trainer class for pretraining models.

        Arguments:
            - model (Type[Transformer]): The class of the model to train. Model initialization is handled by the trainer.
            - dataloader (Type[BaseDataLoader]): The class of the dataloader to use. The trainer handles dividing the dataset in splits and shuffling them.
            - config_path (str): The path to configuration file.
            - datapath (str): The path to where the dataset to train on is stored.
            - savepath (str): The path to where the results and metadata of the training process is stored.
            - benchmarks (list[Benchmark]): A list of benchmarks to run the model through during training at specific intervals, specified in the configuration.
            - device (str): The device to use for training.
        """

        name, ext = os.path.splitext(config_path)
        if ext == ".json":
            config = Config.from_json(config_path)
        elif ext == ".yaml":
            config = Config.from_yaml(config_path)
        else:
            raise ValueError(f"Cannot load configurations in {ext} format.")

        if config.training.server.use_server:
            try:
                from .server import TrainingServer
            except ImportError as e:
                raise ImportError(
                    "Server support requires FastAPI to be installed.\n"
                    "Install the required libraries with: pip install 'dlmake[server]'."
                ) from e
            else:
                self.server = True
        else:
            self.server = False

        self.config = config
        self.name = name
        self.model_class = model
        self.datapath = datapath
        self.dataloader = dataloader
        self.savepath = savepath
        self.benchmarks = benchmarks
        self.device = device
        self.models = 1
        self.kwargs = kwargs
        self.testing = int(os.environ.get("DLMAKE_TESTING", -1)) != -1

        try:
            subprocess.check_output(["nvidia-smi"])
        except FileNotFoundError:
            use_gpu = False
        else:
            use_gpu = True

        if use_gpu:
            self.gpu_mib = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"]
            ).decode().strip().split(" ")[0])
        else:
            p1 = subprocess.Popen(["free", "-m"], stdout=subprocess.PIPE)
            p1.wait()
            p2 = subprocess.Popen(["awk", "{print $2}"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p2.wait()
            p3 = subprocess.Popen(["head", "-n", "2"], stdin=p2.stdout, stdout=subprocess.PIPE)
            p3.wait()
            self.gpu_mib = int(subprocess.check_output(["tail", "-n", "1"], stdin=p3.stdout).decode().strip())

        if not self.testing and not os.path.exists(savepath):
            os.mkdir(savepath)

    def train(self):
        save_name = f"{self.name}"
        if not self.testing:
            if not os.path.exists(os.path.join(self.savepath, save_name)):
                exists = False
                os.mkdir(os.path.join(self.savepath, save_name))
            else:
                exists = True
            logger = Logger(os.path.join(self.savepath, save_name, "log.log"), new=exists)
        else:
            logger = Logger("", new=True)
    
        with torch.device("meta"):
            model = self.model_class(self.config)
            size_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / (8 * 1024 * 1024)

            if size_mib > self.gpu_mib:
                logger.warn(f"The current model requires {size_mib:,} MiB of space for the parameters, but the current GPU only has {self.gpu_mib:,} MiB.")
                logger.warn("Skipping model...")
                return
            del model

        ctx = mp.get_context("spawn")
        worker = Worker(
            model_class=self.model_class,
            config=self.config,
            datapath=self.datapath,
            savepath=self.savepath,
            dataloader=self.dataloader,
            benchmarks=self.benchmarks,
            model_index=0,
            model_count=self.models,
            device=self.device,
            logger=logger,
        )
        main2work, work2main = ctx.Pipe(duplex=False)
        p1 = ctx.Process(target=worker.work, args=(work2main,))
        if self.server:
            runner = Runner(self.config)
            p2 = ctx.Process(target=runner.run_server)
            p2.start()
            logger.log("Waiting for server to start.")
            connected = False
            while not connected:
                try:
                    r = requests.get(
                        f"http://{self.config.training.server.host}:{self.config.training.server.port}/health",
                    )
                except:
                    continue
                else:
                    if r.status_code == 200:
                        logger.log("Detected server. Starting training...")
                        connected = True
            p3 = ctx.Process(target=runner.vram_usage, args=(stats,))
            p3.start()
        p1.start()
        while True:
            content: dict = main2work.recv()
            if self.server:
                if content == "end":
                    p3.terminate()  # type: ignore
                    p3.join()  # type: ignore
                    p3.close()  # type: ignore
                    p2.terminate()  # type: ignore
                    p2.join()  # type: ignore
                    p2.close()  # type: ignore
                    break
                else:
                    response = requests.post(
                        f"http://{self.config.training.server.host}:{self.config.training.server.port}/update",
                        json={k: (content[k] if k in content.keys() else None) for k in stats}
                    )
                    if self.config.use_looping and "loop_exit_probs" in content.keys():
                        buffer = io.BytesIO()    
                        torch.save(content["loop_exit_probs"], buffer)
                        buffer.seek(0)
                        r2 = requests.post(
                            f"http://{self.config.training.server.host}:{self.config.training.server.port}/loop",
                            data=buffer,
                            headers={"Content-Type": "application/octet-stream"}
                        )
                        status = r2.status_code != 200
                        content = r2.json()
                    else:
                        status = False
                        content = {}
                    if response.status_code != 200 or status:
                        logger.error("Trainer: Couldn't send train metadata to server.")
                        if status:
                            logger.error(f"Trainer: Message from server: {content['detail']}")
                        p1.terminate()
                        p1.join()
                        p1.close()
                        p3.terminate()  # type: ignore
                        p3.join()  # type: ignore
                        p3.close()  # type: ignore
                        p2.terminate()  # type: ignore
                        p2.join()  # type: ignore
                        p2.close()  # type: ignore
                        break
            else:
                if content == "end":
                    p1.join()
                    break
