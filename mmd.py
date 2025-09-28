# mmd.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from models import TargetEncoder


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            if n_samples < 2:
                return 1.0
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        
        L2_distances = torch.cdist(X, X) ** 2
        
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)

class MMDLoss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        
        return XX - 2 * XY + YY


def train_domain_adaptation(source_dataset, target_dataset, num_epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
    
    target_encoder = TargetEncoder().to(device)
    mmd_loss = MMDLoss(kernel=RBF()).to(device)
    
    optimizer = optim.AdamW(target_encoder.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        target_encoder.train()
        total_loss = 0
        
        for source_batch, target_batch in zip(source_loader, target_loader):
            source_input, _ = source_batch
            target_input, _ = target_batch

            if source_input.dim() == 2:
                source_input = source_input.unsqueeze(1)
            if target_input.dim() == 2:
                target_input = target_input.unsqueeze(1)
            
            source_input = source_input.to(device)
            target_input = target_input.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                target_output = target_encoder(target_input)
                
                source_flat = source_input.view(source_input.size(0), -1)
                target_flat = target_output.view(target_output.size(0), -1)
                
                loss = mmd_loss(source_flat, target_flat)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(source_loader)
        print(f'epoch [{epoch+1}/{num_epochs}], average MMD Loss: {avg_loss:.8f}')
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(target_encoder.state_dict(), 'target_encoder_best.pt')
            print(f"best model saved with loss: {best_loss:.8f}")

    return target_encoder
