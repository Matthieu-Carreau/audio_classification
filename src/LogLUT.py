import torch

class LogLUT:

    def __init__(self, ref_y=torch.linspace(0, 20, 15), 
                 eps=1e-5, extrapolate="linear", device=None):
        """
        - extrapolate: "constant" or "linear"
        """
        ref_x = torch.expm1(ref_y) * eps
        self.slopes = torch.diff(ref_y) / torch.diff(ref_x)
        self.offset = ref_y[:-1] - self.slopes * ref_x[:-1]
        
        self.inv_eps = torch.tensor([1 / eps]).to(device)
        self.inv_y_step = torch.tensor([1 / ref_y[1]]).to(device)

        self.ref_x = ref_x

        if extrapolate == "constant":
            # Add a last point with zero slope 
            self.slopes = torch.cat([self.slopes, torch.zeros(1)]) 
            self.offset = torch.cat([self.offset, ref_y[-1].unsqueeze(0)])

        self.slopes = self.slopes.to(device)
        self.offset = self.offset.to(device)
        self.n_segments = len(self.slopes)

    def log_lut(self, x): # 4'01"
        indices = (self.inv_y_step * torch.log1p(x * self.inv_eps)).long().clamp_(0, self.n_segments - 1)
        return self.slopes[indices] * x + self.offset[indices]