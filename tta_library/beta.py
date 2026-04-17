"""
BETA: Adapting in the Dark — Efficient and Stable Test-Time Adaptation for Black-Box Models.

Reference implementation of the BETA algorithm. Given a frozen black-box target model
`model` (accessible only through its predictions) and a small local helper model
`local_helper`, BETA jointly optimizes:

    * a learnable visual prompter (PadPrompter) that perturbs the input image,
    * a probability fuser (ProbFuser) that harmonizes local and black-box logits,
    * the normalization layers of the local helper.

The update consists of three loss terms:

    1. loss_norm          — reliability-weighted entropy of the local helper on
                            clean images (adapts normalization layers, similar in
                            spirit to EATA).
    2. loss_entropy_fused — reliability-weighted entropy of the harmonized
                            (prompted-local + black-box) probability.
    3. loss_kl            — KL regularization between local predictions on clean
                            vs. prompted images, for stability.

See Algorithm 1 in the paper for the full procedure.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prompter import PadPrompter, ProbFuser


class BETA(nn.Module):
    """
    BETA: joint prompt + normalization adaptation under a black-box constraint.

    Args:
        model:          frozen black-box target model (no gradients).
        local_helper:   trainable small local model (e.g., ResNet-18 / ViT-S).
        prompter:       PadPrompter (learnable border of `pad_size`).
        probfuser:      ProbFuser (soft mixture of local and black-box probs).
        optimizer_vr:   optimizer for prompter (+ fuser) parameters.
        optimizer_norm: optimizer for the norm-layer parameters of local_helper.
        kl_weight:      weight of the KL consistency term (λ in the paper).
        e_margin:       entropy margin for reliable-sample filtering.
        d_margin:       cosine-similarity margin for sample diversity filter.
        imagenet_mask:  optional class mask for ImageNet-R / ImageNet-A style subsets.
        steps:          number of inner adaptation steps per batch.
    """

    def __init__(self,
                 model,
                 local_helper: nn.Module,
                 prompter: PadPrompter,
                 probfuser: ProbFuser,
                 optimizer_vr,
                 optimizer_norm,
                 kl_weight: float = 1.0,
                 e_margin: float = 0.4 * math.log(1000),
                 d_margin: float = 0.05,
                 imagenet_mask=None,
                 steps: int = 1):
        super().__init__()
        self.model = model
        self.local_helper = local_helper
        self.prompter = prompter
        self.probfuser = probfuser

        self.optimizer_vr = optimizer_vr
        self.optimizer_norm = optimizer_norm
        self.steps = steps
        self.kl_weight = kl_weight
        self.imagenet_mask = imagenet_mask
        self.emargin = e_margin
        self.dmargin = d_margin

        self.info = {"used": 0, "skip": 0}
        self.current_model_probs = None

        # Store initial states for resetting across runs / tasks.
        (self.local_helper_state,
         self.prompter_state,
         self.probfuser_state,
         self.optimizer_vr_state,
         self.optimizer_norm_state) = copy_model_and_optimizer(
            self.local_helper, self.prompter, self.probfuser,
            self.optimizer_vr, self.optimizer_norm,
        )

    def forward(self, x):
        for _ in range(self.steps):
            (outputs, model_outputs, local_outputs,
             skip, used, updated_probs) = forward_and_adapt(
                x, self.model, self.local_helper, self.prompter, self.probfuser,
                self.optimizer_vr, self.optimizer_norm,
                self.emargin, self.dmargin, self.kl_weight,
                self.current_model_probs, self.imagenet_mask,
            )
            self.current_model_probs = updated_probs
        self.info["skip"] += skip
        self.info["used"] += used
        return outputs, model_outputs, local_outputs

    def reset(self):
        """Restore all trainable components and optimizers to their initial state."""
        load_model_and_optimizer(
            self.local_helper, self.prompter, self.probfuser,
            self.optimizer_vr, self.optimizer_norm,
            self.local_helper_state, self.prompter_state, self.probfuser_state,
            self.optimizer_vr_state, self.optimizer_norm_state,
        )
        self.info = {"used": 0, "skip": 0}


# ---------------------------------------------------------------------------
# Core adaptation step
# ---------------------------------------------------------------------------

@torch.jit.script
def softmax_entropy_from_prob(p: torch.Tensor) -> torch.Tensor:
    """Entropy of a probability distribution."""
    return -(p * torch.log(p + 1e-8)).sum(1)


@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.enable_grad()
def forward_and_adapt(x, model, local_helper, prompter, probfuser,
                      optimizer_vr, optimizer_norm,
                      emargin, dmargin, kl_weight,
                      current_model_probs, imagenet_mask):
    """One adaptation step of BETA.

    Returns:
        final_outputs_no_grad: harmonized probabilities used for evaluation.
        model_outputs:         raw black-box logits on the prompted input.
        local_outputs:         local helper logits on the prompted input.
        skipped, used, probs:  bookkeeping.
    """
    # 1. Keep normalization layers trainable.
    for m in local_helper.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)

    # 2. Prompted image and forward passes.
    x_prompted = prompter(x)

    local_outputs_clean = local_helper(x)
    local_outputs_prompted = local_helper(x_prompted)

    with torch.no_grad():
        model_outputs = model(x_prompted)

    if imagenet_mask is not None:
        model_outputs = model_outputs[:, imagenet_mask]
        local_outputs_clean = local_outputs_clean[:, imagenet_mask]
        local_outputs_prompted = local_outputs_prompted[:, imagenet_mask]

    # 3. Reliability filtering on clean-image entropy.
    total = x.size(0)
    loss_norm = torch.tensor(0.0, device=x.device)
    loss_entropy_fused = torch.tensor(0.0, device=x.device)
    updated_probs = current_model_probs

    entropys_clean = softmax_entropy(local_outputs_clean)
    reliable_ids = torch.where(entropys_clean < emargin)[0]

    if reliable_ids.numel() > 0:
        reliable_entropys = entropys_clean[reliable_ids]
        reliable_outputs_clean = local_outputs_clean[reliable_ids]

        # Optional diversity filter against the running-average probability.
        if current_model_probs is not None:
            cos_sim = F.cosine_similarity(
                current_model_probs.unsqueeze(0),
                reliable_outputs_clean.softmax(1), dim=1,
            )
            diverse_ids = torch.where(torch.abs(cos_sim) < dmargin)[0]

            if diverse_ids.numel() > 0:
                filt_entropys = reliable_entropys[diverse_ids]
                filt_outputs = reliable_outputs_clean[diverse_ids]
                updated_probs = update_model_probs(
                    current_model_probs, filt_outputs.softmax(1))
                coeff_norm = 1 / torch.exp(filt_entropys.detach() - emargin)
                loss_norm = (filt_entropys * coeff_norm).mean()
        else:
            updated_probs = update_model_probs(
                current_model_probs, reliable_outputs_clean.softmax(1))
            coeff_norm = 1 / torch.exp(reliable_entropys.detach() - emargin)
            loss_norm = (reliable_entropys * coeff_norm).mean()

        # Fused entropy on prompted-image predictions.
        local_prob_prompted = F.softmax(local_outputs_prompted, dim=1)
        model_prob = F.softmax(model_outputs, dim=1)
        fused_prob = probfuser(local_prob_prompted, model_prob)
        entropys_fused = softmax_entropy_from_prob(fused_prob)

        coeff_fused = 1 / torch.exp(entropys_clean[reliable_ids].detach() - emargin)
        loss_entropy_fused = (entropys_fused[reliable_ids] * coeff_fused).mean()

    # 4. KL consistency regularization: clean vs prompted (local view).
    prob_clean = F.softmax(local_outputs_clean, dim=1)
    log_prob_prompted = F.log_softmax(local_outputs_prompted, dim=1)
    loss_kl = F.kl_div(log_prob_prompted, prob_clean, reduction="batchmean")

    total_loss = loss_norm + loss_entropy_fused + kl_weight * loss_kl

    used = reliable_ids.numel()
    skipped = total - used

    if used > 0 and total_loss.requires_grad:
        total_loss.backward()
        optimizer_vr.step()
        optimizer_norm.step()

    optimizer_vr.zero_grad()
    optimizer_norm.zero_grad()

    # Final harmonized outputs for accuracy / ECE logging.
    final_outputs_no_grad = probfuser(
        local_outputs_prompted.softmax(1).detach(),
        model_outputs.softmax(1).detach(),
    )
    return (final_outputs_no_grad, model_outputs, local_outputs_prompted,
            skipped, used, updated_probs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_model_probs(current_model_probs, new_probs, momentum: float = 0.9):
    """EMA over the running probability vector used for the diversity filter."""
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        with torch.no_grad():
            return new_probs.mean(0)
    if new_probs.size(0) == 0:
        return current_model_probs
    with torch.no_grad():
        return momentum * current_model_probs + (1 - momentum) * new_probs.mean(0)


def collect_params(local_helper: nn.Module,
                   prompter: PadPrompter,
                   probfuser: ProbFuser):
    """Return (vr_params, norm_params, fuser_params) for the two optimizers."""
    vr_params, vr_names = [], []
    for n, p in prompter.named_parameters():
        vr_params.append(p)
        vr_names.append(f"prompter.{n}")

    norm_params, norm_names = [], []
    for nm, m in local_helper.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for n, p in m.named_parameters():
                if n in ("weight", "bias"):
                    norm_params.append(p)
                    norm_names.append(f"{nm}.{n}")

    fuser_params, fuser_names = [], []
    for n, p in probfuser.named_parameters():
        fuser_params.append(p)
        fuser_names.append(f"probfuser.{n}")

    return vr_params, vr_names, norm_params, norm_names, fuser_params, fuser_names


def configure_model(local_helper: nn.Module,
                    prompter: PadPrompter,
                    probfuser: ProbFuser):
    """Freeze the backbone except normalization layers, prompter and fuser."""
    local_helper.train()
    prompter.train()

    local_helper.requires_grad_(False)
    prompter.requires_grad_(True)
    probfuser.requires_grad_(True)

    for m in local_helper.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # Use batch stats (TTA convention).
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)

    return local_helper, prompter, probfuser


def copy_model_and_optimizer(model, prompter, probfuser,
                             optimizer_vr, optimizer_norm):
    return (deepcopy(model.state_dict()),
            deepcopy(prompter.state_dict()),
            deepcopy(probfuser.state_dict()),
            deepcopy(optimizer_vr.state_dict()),
            deepcopy(optimizer_norm.state_dict()))


def load_model_and_optimizer(model, prompter, probfuser,
                             optimizer_vr, optimizer_norm,
                             model_state, prompter_state, probfuser_state,
                             optimizer_vr_state, optimizer_norm_state):
    model.load_state_dict(model_state, strict=True)
    prompter.load_state_dict(prompter_state, strict=True)
    probfuser.load_state_dict(probfuser_state, strict=True)
    optimizer_vr.load_state_dict(optimizer_vr_state)
    optimizer_norm.load_state_dict(optimizer_norm_state)
