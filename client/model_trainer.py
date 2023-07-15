from abc import ABC, abstractmethod
from fractal_learning.training import datamodule


class ModelTrainer(ABC):
    """Abstract base class for federated learning trainer.
       1. The goal of this abstract class is to be compatible to
       any deep learning frameworks such as PyTorch, TensorFlow, Keras, MXNET, etc.
       2. This class can be used in both server and client side
       3. This class is an operator which does not cache any states inside.
    """
    def __init__(self, args, model=None, device=None):
        self.model = model
        self.id = 0
        self.args = args
        self.device = device
        self.current_classes = None

        if args.centralized_fractal_pretrain_steps:
            self.pretrain_loader = datamodule.FractalClassDataModule().train_dataloader()


    def set_id(self, trainer_id):
        self.id = trainer_id

    @abstractmethod
    def update_output_dim(self, class_num):
        pass

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, device, args=None):
        pass

