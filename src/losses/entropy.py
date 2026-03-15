"""
losses/entropy.py

Entropy Loss — polarizzazione della confidence mask verso valori binari.

Incentiva la confidence mask α(x,y) a prendere decisioni nette (α ∈ {0,1})
invece di rimanere centrata su 0.5 (media tra ramo globale e locale), che
produrrebbe aloni cromatici ai bordi dove le due griglie divergono.

Riferimento tesi: §5.1.7 e §6.5.8
Formula:
    L_entropy = -(λ_e / HW) Σ_{i,j} [α_ij·log(α_ij + ε) + (1-α_ij)·log(1-α_ij + ε)]

Con λ_e = 0.01 (costante in tutte le fasi del curriculum).

Proprietà:
    - L_entropy = 0       quando α ∈ {0,1} ovunque (mask binaria perfetta)
    - L_entropy = log(2)  quando α = 0.5 ovunque (massima incertezza)
    Il segno negativo fa sì che minimizzare L_entropy = massimizzare la certezza.

Interazione con D_3 (mappa di divergenza):
    D_3 informa la mask SU DOVE decidere (dove le due griglie differiscono).
    L_entropy incentiva la mask a decidere CON CERTEZZA in quei punti.
    I due meccanismi sono complementari.
"""

import torch
import torch.nn as nn
import math

EPS = 1e-6   # ε per stabilità numerica in log(0)


class EntropyLoss(nn.Module):
    """
    Loss di entropia binaria che polarizza α verso {0, 1}.

    NOTA: Il peso λ_e = 0.01 è mantenuto costante in tutte le fasi
    del curriculum (§6.6). Il modulo calcola la loss senza pesatura interna:
    il peso λ_e è applicato nel CompositeLoss.

    Args:
        eps: Valore ε per stabilità numerica nel logaritmo. Default 1e-6.

    Example:
        >>> loss_fn = EntropyLoss()
        >>> alpha = torch.sigmoid(torch.randn(2, 1, 384, 512))  # (B, 1, H, W)
        >>> loss  = loss_fn(alpha)   # scalar, valore in [0, log(2)]
    """

    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alpha: Confidence mask (B, 1, H, W) o (B, H, W), valori in [0,1].
                   Prodotta da σ(MaskNet(...)) — già nel range [0,1].

        Returns:
            Scalare: entropia media su tutti i pixel del batch.
            Valore in [0, log(2)] ≈ [0, 0.693].

            Il training MINIMIZZA questa loss, quindi MASSIMIZZA la certezza
            (α → 0 o α → 1).
        """
        a = alpha.float()

        # Entropia binaria per pixel: -[a·log(a+ε) + (1-a)·log(1-a+ε)]
        entropy = -(a * torch.log(a + self.eps)
                    + (1.0 - a) * torch.log(1.0 - a + self.eps))

        return entropy.mean()
