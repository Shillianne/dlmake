from .controllers import stat_router, benchmark_router
from fastapi import FastAPI
from dlmake import Config
import uvicorn


class TrainingServer:
    def __init__(self, config: Config):
        self.host = config.training.server.host
        self.port = config.training.server.port

        self.app = FastAPI()

        @self.app.get("/health")
        def health():
            return {"status": "ok"}

        @self.app.get("/config/metrics")
        def metrics():
            return config.training.logging.metrics.__dict__

        self.app.include_router(stat_router)
        self.app.include_router(benchmark_router)

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
