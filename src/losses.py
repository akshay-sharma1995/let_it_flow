import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ssim
class ReconstructionLoss():
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def ReturnLoss(self, frame1, frame2):
        return self.criterion(frame1, frame2)

class SsimLoss():
    def __init__(self):
        super().__init__()
        self.criterion = pytorch_ssim.SSIM()
    
    def ReturnLoss(self, frame1, frame2):
        return -self.criterion(frame1, frame2)
