import os
import torch


class Config:
    def __init__(self, lambda_h=65.0, lambda_c=15.0, lambda_r=12.0,
                 lambda_s=0.5, inn_blocks=16, experiment_name="IC-INN"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.LAMBDA_H = lambda_h
        self.LAMBDA_C = lambda_c
        self.LAMBDA_R = lambda_r
        self.LAMBDA_S = lambda_s
        self.LAMBDA_Z = 0.1
        self.LAMBDA_HF = 0.0
        self.INN_BLOCKS = inn_blocks
        self.KEY_DIM = 256
        self.EXP_NAME = experiment_name
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TRAIN_DATA_ROOT = '#'
        self.TEST_DATA_ROOT = '#'
        self.CHECKPOINT_DIR = './checkpoints'
        self.RESULT_DIR = './results'
        self.IMAGE_SIZE = 256
        self.BATCH_SIZE = 4
        self.NUM_WORKERS = 4
        self.TRAIN_LIMIT = 6000
        self.TEST_LIMIT = 100
        self.EPOCHS = 300
        self.START_EPOCH = 0
        self.SAVE_INTERVAL = 10
        self.LR = 1e-4
        self.ACCUM_STEPS = 3
        self.STAGE1_END = 80
        self.STAGE2_END = 180
        self.PRETRAIN_EPOCHS = 100
        self.USE_VC_KEYS = True
