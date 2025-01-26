import torch
import torch.nn as nn


class KLDivergenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pij, qij):
        # Compute KLD(pij || qij)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute sum of all variational terms
        loss_kld = loss_kld.sum()

        return loss_kld


class JSDivergenceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pij, qij):
        # Compute the average distribution M
        M = 0.5 * (pij + qij)
        
        # Compute KL divergence terms
        kl_p_m = pij * (torch.log(pij) - torch.log(M))
        kl_q_m = qij * (torch.log(qij) - torch.log(M))
        
        # Compute sum of KL divergence terms
        kl_p_m = kl_p_m.sum()  # Sum over the appropriate dimension
        kl_q_m = kl_q_m.sum()  # Sum over the appropriate dimension
        
        # Compute JS divergence
        js_divergence = 0.5 * kl_p_m + 0.5 * kl_q_m

        return js_divergence


class HellingerDistanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pij, qij):
        # Compute the Hellinger distance
        sqrt_pij = torch.sqrt(pij)
        sqrt_qij = torch.sqrt(qij)
        
        # Hellinger distance calculation
        distance = torch.norm(sqrt_pij - sqrt_qij, p=2) / torch.sqrt(torch.tensor(2.0))

        return distance


class SquaredHellingerDistanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pij, qij):
        # Compute the Hellinger distance
        sqrt_pij = torch.sqrt(pij)
        sqrt_qij = torch.sqrt(qij)
        
        # Squared Hellinger distance calculation
        distance = torch.square(sqrt_pij - sqrt_qij) / 2.0
        return distance.sum()



