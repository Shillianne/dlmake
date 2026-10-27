from .server import TrainingServer
from dlmake import Config
import subprocess
import requests
import time


class Runner:
    def __init__(self, config: Config) -> None:
        self.config = config

    def run_server(self):
        server = TrainingServer(self.config)
        server.run()

    def vram_usage(self, stats: list[str]):
        while True:
            used = int(subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"
            ]).decode().strip().split(" ")[0])

            requests.post(
                f"http://{self.config.training.server.host}:{self.config.training.server.port}/update",
                json={k: (used if k == "vram_used" else None) for k in stats}
            )
            time.sleep(self.config.training.server.vram_heartbeat)
