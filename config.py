import torch.cuda


class BaseConfig:
    def __init__(self):
        self.result_dir = "result"
        self.experiment_name = "experiment_1"
        self.train_path = "../data/SegDataset/TrainDataset"
        self.use_bright = True
        self.save_frequency = 40
        self.in_channels = 3
        self.out_channels = 1
        self.mid_channels = 16
        self.batch_size = 4
        self.lr = 2e-4
        self.init_gain = 0.02
        self.n_epochs = 200
        self.decay_epoch = 150
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = 352
        self.init_gain = 0.02
        self.init_type = 'xavier'
        self.norm = "BATCH"
        self.interpolation_mode = "bilinear"
        self.test_epoch = 40
        self.lambda_identity = 0.5
        self.threshold = 0.5