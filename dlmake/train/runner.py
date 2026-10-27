from dlmake import Config
import subprocess
import requests
import time


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config
        try:
            subprocess.check_output(["nvidia-smi"])
        except FileNotFoundError:
            self.use_gpu = False
        else:
            self.use_gpu = True

    def run_server(self):
        from .server import TrainingServer
        server = TrainingServer(self.config)
        server.run()

    def vram_usage(self, stats: list[str]):
        while True:
            if self.use_gpu:
                used = int(subprocess.check_output([
                    "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"
                ]).decode().strip().split(" ")[0])
            else:
                p1 = subprocess.Popen(["free", "-m"], stdout=subprocess.PIPE)
                p2 = subprocess.Popen(["awk", "'{print $3}'"], stdin=p1.stdout, stdout=subprocess.PIPE)
                p3 = subprocess.Popen(["head", "-n", "2"], stdin=p2.stdout, stdout=subprocess.PIPE)
                used = int(subprocess.check_output(["tail", "-n", "1"], stdin=p3.stdout).decode().strip())

            requests.post(
                f"http://{self.config.training.server.host}:{self.config.training.server.port}/update",
                json={k: (used if k == "vram_used" else None) for k in stats}
            )
            time.sleep(self.config.training.server.vram_heartbeat)
