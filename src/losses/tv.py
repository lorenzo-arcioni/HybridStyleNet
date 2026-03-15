"""
losses/tv.py

Total Variation Loss — smoothness spaziale dei coefficienti della bilateral grid.

Penalizza la variazione totale dei coefficienti affini A_{ij} della bilateral
grid nello spazio della griglia, incoraggiando trasformazioni cromatiche
spatially smooth. È l'unica loss che agisce direttamente sui parametri
interni della grid, non sull'output I^pred.

Riferimento tesi: §6.5.6
Formula:
    L_TV = (1/|G|) Σ_{i,j} (‖A_{i+1,j} - A_{i,j}‖_F + ‖A_{i,j+1} - A_{i,j}‖_F)

Dove:
    A_{i,j} ∈ R^{3×3} è la matrice di trasformazione cromatica alla cella (i,j)
    |G| = H_g × W_g (numero totale di celle nella griglia)

Interpretazione geometrica:
    L_TV è l'analogo discreto del Matting Laplacian constraint di Deep Photo
    Style Transfer (Luan et al., SIGGRAPH 2017), qui adattato per la bilateral
    grid discreta invece del dominio continuo.

Perché λ_TV = 0.01 (piccolo):
    La bilateral grid è progettata per fare local edits edge-aware.
    Un λ_TV troppo grande collapserebbe la grid verso una trasformazione
    globale uniforme, perdendo la capacità di correzioni locali.
    Il valore 0.01 previene discontinuità numeriche senza sopprimere
    la variazione spaziale legittima.

Differenza da L_id:
    L_id previene che la trasformazione sia troppo AGGRESSIVA (in ampiezza).
    L_TV previene che sia troppo DISCONTINUA (in smoothness spaziale).
    I due regolarizzatori controllano dimensioni ortogonali.
"""

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    """
    Total Variation loss sui coefficienti affini della bilateral grid.

    Input atteso: il tensore della bilateral grid G con i soli coefficienti
    della matrice A (prime 9 componenti dei 12 totali), oppure tutti e 12
    (matrice 3×3 + bias 3 → 12 coefficienti).

    La loss può essere applicata a entrambe le griglie (globale e locale)
    o solo a una di esse, a seconda di come viene chiamata nel composite loss.

    Args:
        use_frobenius: Se True (default), usa la norma di Frobenius per
                       la differenza tra celle. Se False, usa la norma L1.

    Formato atteso della grid (shape):
        (B, 12, H_g, W_g, L_b)  — formato usato da bilateral_grid.py
        oppure
        (B, 12, H_g, W_g)       — dopo riduzione lungo L_b

    Strategia: la TV viene calcolata mediando lungo la dimensione L_b
    (bin di luminanza) se presente, poi applicando la differenza spaziale.

    Example:
        >>> loss_fn = TotalVariationLoss()
        >>> # Grid globale: (B, 12, 8, 8, 8) → media su L_b → (B, 12, 8, 8)
        >>> grid = torch.rand(2, 12, 8, 8, 8)
        >>> loss  = loss_fn(grid)   # scalar tensor
    """

    def __init__(self, use_frobenius: bool = True):
        super().__init__()
        self.use_frobenius = use_frobenius

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: Tensore float32 contenente i coefficienti affini della
                  bilateral grid. Shape attesa:
                    (B, 12, H_g, W_g, L_b)   [formato completo]
                  oppure
                    (B, 12, H_g, W_g)         [già ridotto su L_b]

        Returns:
            Scalare: TV media dei coefficienti su tutte le celle.
        """
        g = grid.float()

        # Riduci lungo L_b se presente (dimensione 4)
        if g.dim() == 5:
            # (B, 12, H_g, W_g, L_b) → media su L_b → (B, 12, H_g, W_g)
            g = g.mean(dim=-1)

        # g shape: (B, 12, H_g, W_g)
        B, C, H_g, W_g = g.shape

        # Differenze orizzontali: A_{i,j+1} - A_{i,j}
        diff_h = g[:, :, :, 1:] - g[:, :, :, :-1]   # (B, 12, H_g, W_g-1)
        # Differenze verticali: A_{i+1,j} - A_{i,j}
        diff_v = g[:, :, 1:, :] - g[:, :, :-1, :]   # (B, 12, H_g-1, W_g)

        if self.use_frobenius:
            # Norma di Frobenius per cella: sqrt(Σ_c coeff^2)
            # equivale a: norma L2 lungo la dimensione dei coefficienti
            tv_h = diff_h.pow(2).sum(dim=1).sqrt()   # (B, H_g, W_g-1)
            tv_v = diff_v.pow(2).sum(dim=1).sqrt()   # (B, H_g-1, W_g)
        else:
            # Norma L1 per cella
            tv_h = diff_h.abs().sum(dim=1)            # (B, H_g, W_g-1)
            tv_v = diff_v.abs().sum(dim=1)            # (B, H_g-1, W_g)

        # Totale: somma TV orizzontale + verticale
        n_cells = H_g * W_g
        tv = (tv_h.sum() + tv_v.sum()) / (B * n_cells)

        return tv
