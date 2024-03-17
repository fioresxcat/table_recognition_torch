from .base_trainer import BaseTrainer


class SLATrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)