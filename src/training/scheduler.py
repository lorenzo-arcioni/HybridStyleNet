"""
training/scheduler.py

Learning Rate Scheduler e Loss Curriculum Scheduler  (§6.6, §8.2-8.4).

LRScheduler:
    Wrapper unificato per warmup lineare + cosine annealing.

LossCurriculumScheduler:
    Aggiorna i pesi della loss composita seguendo il curriculum §6.6.
"""

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.optim as optim

from losses.composite import LossWeights, ColorAestheticLoss

logger = logging.getLogger(__name__)


class LRScheduler:
    """
    Scheduler LR con warmup lineare + cosine annealing.

    Schema:
        [0, warmup_steps)      → LR cresce linearmente da 0 a base_lr
        [warmup_steps, T_max)  → Cosine annealing da base_lr a eta_min

    Args:
        optimizer:     Ottimizzatore PyTorch.
        base_lr:       Learning rate di picco.
        warmup_steps:  Numero di step di warmup (in iterazioni, non epoche).
        T_max:         Numero totale di step (warmup incluso).
        eta_min:       LR minimo al termine del cosine annealing.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        warmup_steps: int = 0,
        T_max: int = 10000,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer     = optimizer
        self.base_lr       = base_lr
        self.warmup_steps  = warmup_steps
        self.T_max         = T_max
        self.eta_min       = eta_min
        self._step         = 0

    def step(self) -> float:
        """
        Avanza di uno step e aggiorna il LR di tutti i param group.

        Returns:
            LR corrente dopo l'aggiornamento.
        """
        self._step += 1
        lr = self._compute_lr(self._step)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _compute_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            # Warmup lineare
            return self.base_lr * step / self.warmup_steps
        else:
            # Cosine annealing
            t = step - self.warmup_steps
            T = max(self.T_max - self.warmup_steps, 1)
            cos_val = math.cos(math.pi * t / T)
            return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + cos_val)

    def state_dict(self) -> Dict:
        return {
            "step":          self._step,
            "base_lr":       self.base_lr,
            "warmup_steps":  self.warmup_steps,
            "T_max":         self.T_max,
            "eta_min":       self.eta_min,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._step        = state["step"]
        self.base_lr      = state["base_lr"]
        self.warmup_steps = state["warmup_steps"]
        self.T_max        = state["T_max"]
        self.eta_min      = state["eta_min"]

    def get_lr(self) -> float:
        return self._compute_lr(self._step)


class CosineAnnealingScheduler:
    """
    Cosine annealing semplice per la Fase 3B  (§5.4).

    η(t) = η_min + 0.5 * (η_B - η_min) * (1 + cos(π * t / T_max))

    Args:
        optimizer: Ottimizzatore.
        base_lr:   Learning rate iniziale η_B.
        T_max:     Numero di epoche totali.
        eta_min:   LR minimo.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        T_max: int = 20,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.base_lr   = base_lr
        self.T_max     = T_max
        self.eta_min   = eta_min
        self._epoch    = 0

    def step_epoch(self) -> float:
        """Avanza di un'epoca."""
        self._epoch += 1
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1.0 + math.cos(math.pi * self._epoch / self.T_max)
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def state_dict(self) -> Dict:
        return {"epoch": self._epoch, "base_lr": self.base_lr,
                "T_max": self.T_max, "eta_min": self.eta_min}

    def load_state_dict(self, state: Dict) -> None:
        self._epoch  = state["epoch"]
        self.base_lr = state["base_lr"]
        self.T_max   = state["T_max"]
        self.eta_min = state["eta_min"]


class LossCurriculumScheduler:
    """
    Aggiorna i pesi della ColorAestheticLoss seguendo il curriculum §6.6.

    Curriculum:
        Epoche  1–5  → LossWeights.curriculum_1_5()
        Epoche  6–10 → LossWeights.curriculum_6_10()
        Epoche 11+   → LossWeights.curriculum_11_plus()

    Può essere esteso con un curriculum personalizzato passando
    `curriculum_map`.

    Args:
        loss_fn:        Istanza ColorAestheticLoss da aggiornare.
        curriculum_map: Dizionario {epoch_start: LossWeights}.
                        Le epoche devono essere in ordine crescente.
                        None → usa il curriculum di default.
        verbose:        Se True, logga ogni cambio di pesi.
    """

    _DEFAULT_CURRICULUM = {
        1:  LossWeights.curriculum_1_5,
        6:  LossWeights.curriculum_6_10,
        11: LossWeights.curriculum_11_plus,
    }

    def __init__(
        self,
        loss_fn: ColorAestheticLoss,
        curriculum_map: Optional[Dict[int, callable]] = None,
        verbose: bool = True,
    ) -> None:
        self.loss_fn  = loss_fn
        self.verbose  = verbose
        self._current_weights: Optional[LossWeights] = None

        # Costruisce la mappa ordinata {epoch: factory_fn}
        raw_map = curriculum_map or self._DEFAULT_CURRICULUM
        self._schedule: List[tuple] = sorted(raw_map.items())

    def step_epoch(self, epoch: int) -> Optional[LossWeights]:
        """
        Controlla se i pesi devono cambiare all'epoca `epoch`.

        Args:
            epoch: Epoca corrente (1-indexed).

        Returns:
            Il nuovo LossWeights se è avvenuto un cambio, None altrimenti.
        """
        new_weights = None
        for start_epoch, factory in reversed(self._schedule):
            if epoch >= start_epoch:
                new_weights = factory()
                break

        if new_weights is None:
            return None

        # Aggiorna solo se i pesi sono cambiati
        if self._current_weights is None or \
                vars(new_weights) != vars(self._current_weights):
            self.loss_fn.update_weights(new_weights)
            self._current_weights = new_weights
            if self.verbose:
                logger.info(
                    f"Curriculum epoch {epoch}: "
                    f"λ_ΔE={new_weights.lambda_delta_e}, "
                    f"λ_hist={new_weights.lambda_hist}, "
                    f"λ_perc={new_weights.lambda_perc}, "
                    f"λ_style={new_weights.lambda_style}, "
                    f"λ_cos={new_weights.lambda_cos}, "
                    f"λ_chroma={new_weights.lambda_chroma}, "
                    f"λ_id={new_weights.lambda_id}"
                )
            return new_weights

        return None

    def set_custom_weights(self, weights: LossWeights) -> None:
        """Imposta pesi personalizzati bypassando il curriculum."""
        self.loss_fn.update_weights(weights)
        self._current_weights = weights

    @classmethod
    def from_config(
        cls,
        loss_fn: ColorAestheticLoss,
        cfg: Dict,
    ) -> "LossCurriculumScheduler":
        """
        Costruisce lo scheduler dal dizionario di configurazione adapt.yaml.

        Args:
            loss_fn: Istanza ColorAestheticLoss.
            cfg:     Sezione 'loss_curriculum' del config (es. phase_a).
        """
        curriculum_map: Dict[int, callable] = {}

        epochs_1_5  = cfg.get("epochs_1_5",  None)
        epochs_6_10 = cfg.get("epochs_6_10", None)
        epochs_11   = cfg.get("epochs_11_plus", None)

        if epochs_1_5:
            w = LossWeights.from_dict(epochs_1_5)
            curriculum_map[1]  = lambda w=w: w

        if epochs_6_10:
            w = LossWeights.from_dict(epochs_6_10)
            curriculum_map[6]  = lambda w=w: w

        if epochs_11:
            w = LossWeights.from_dict(epochs_11)
            curriculum_map[11] = lambda w=w: w

        return cls(loss_fn=loss_fn, curriculum_map=curriculum_map or None)


class EarlyStopping:
    """
    Early stopping basato su una metrica di validazione  (§5.4).

    Args:
        patience:  Numero di epoche senza miglioramento prima dello stop.
        mode:      'min' (es. val_delta_e) o 'max' (es. SSIM).
        min_delta: Miglioramento minimo considerato significativo.
        verbose:   Se True, logga i progressi.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 1e-4,
        verbose: bool = True,
    ) -> None:
        assert mode in ("min", "max"), f"mode deve essere 'min' o 'max', got {mode}"
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.verbose   = verbose

        self._best_score:   Optional[float] = None
        self._counter:      int             = 0
        self._best_epoch:   int             = 0
        self.should_stop:   bool            = False

    def step(self, score: float, epoch: int) -> bool:
        """
        Aggiorna lo stato dell'early stopping.

        Args:
            score: Valore della metrica monitorata.
            epoch: Epoca corrente.

        Returns:
            True se il training deve fermarsi.
        """
        improved = False

        if self._best_score is None:
            improved = True
        elif self.mode == "min":
            improved = score < self._best_score - self.min_delta
        else:
            improved = score > self._best_score + self.min_delta

        if improved:
            self._best_score = score
            self._counter    = 0
            self._best_epoch = epoch
            if self.verbose:
                logger.info(
                    f"EarlyStopping: nuovo best score={score:.6f} "
                    f"all'epoca {epoch}"
                )
        else:
            self._counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: nessun miglioramento "
                    f"({self._counter}/{self.patience})"
                )
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"EarlyStopping: training interrotto all'epoca {epoch}. "
                    f"Best epoch={self._best_epoch}, score={self._best_score:.6f}"
                )

        return self.should_stop

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    def state_dict(self) -> Dict:
        return {
            "best_score": self._best_score,
            "counter":    self._counter,
            "best_epoch": self._best_epoch,
        }

    def load_state_dict(self, state: Dict) -> None:
        self._best_score = state["best_score"]
        self._counter    = state["counter"]
        self._best_epoch = state["best_epoch"]