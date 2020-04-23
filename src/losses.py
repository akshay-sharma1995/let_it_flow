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
        
def flow_loss(frames1, frames2, frames2_pred, flow):
    spatial_loss = spatial_smoothing_loss()
    s_loss = spatial_loss(flow)
    recons_loss = torch.nn.MSELoss()

    loss = 0
    loss = recons_loss(frames2_pred, frames2)

    loss = (0.001)*loss + s_loss
    return loss


class spatial_smoothing_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(spatial_smoothing_loss, self).__init__()
        self.eps = 1e-6
    
    def forward(self, X): ## X is flow map
        u = X[:,0:1]
        v = X[:,1:2]
        # print("u",u.size())
        hf1 = torch.tensor([[[[0,0,0],[-1,2,-1],[0,0,0]]]]).to(X.device).float()
        hf2 = torch.tensor([[[[0,-1,0],[0,2,0],[0,-1,0]]]]).to(X.device).float()
        hf3 = torch.tensor([[[[-1,0,-1],[0,4,0],[-1,0,-1]]]]).to(X.device).float()
        # diff = torch.add(X, -Y)

        u_hloss = F.conv2d(u,hf1,padding=1,stride=1)
        # print("uhloss",type(u_hloss))
        u_vloss = F.conv2d(u,hf2,padding=1,stride=1)
        u_dloss = F.conv2d(u,hf3,padding=1,stride=1)

        v_hloss = F.conv2d(v,hf1,padding=1,stride=1)
        v_vloss = F.conv2d(v,hf2,padding=1,stride=1)
        v_dloss = F.conv2d(v,hf3,padding=1,stride=1)

        u_hloss = charbonier(u_hloss,self.eps)
        u_vloss = charbonier(u_vloss,self.eps)
        u_dloss = charbonier(u_dloss,self.eps)

        v_hloss = charbonier(v_hloss,self.eps)
        v_vloss = charbonier(v_vloss,self.eps)
        v_dloss = charbonier(v_dloss,self.eps)

        loss = u_hloss + u_vloss + u_dloss + v_hloss + v_vloss + v_dloss
        return loss 

def charbonier(x,eps):
    gamma = 0.45
    loss = x*x + eps*eps
    loss = torch.pow(loss,gamma)
    loss = torch.mean(loss)
    return loss

