"""
MetaInit: stores and manages the Reptile meta-initialisation for LoRA adapters.

The four LoRAConvPerSubject layers in EEGNeX LoRA mode are:
    block_1_1, block_2_0, block_4_1, block_5_0

Each has lora_A (ModuleList of Conv2d) and lora_B (ModuleList of Conv2d).

MetaInit holds one set of (A_weight, B_weight) tensors per layer,
representing the best initialisation found by Reptile for a new subject.

Key operations (Euclidean — original):
    load_into_slot(model, slot)        copy meta weights into adapter slot i
    extract_from_slot(model, slot)     read adapter slot i back into meta weights
    reptile_update(phi, meta_lr)       move meta toward phi via Euclidean Reptile
    delta_norm(phi_list)               Euclidean health-check metric

Key operations (Grassmann — new):
    reptile_update_grassmann(phi, lr)  move meta toward phi via Riemannian Reptile
    delta_norm_grassmann(phi_list)     geodesic health-check metric

The Grassmann methods update A matrices on the Grassmann manifold G(r, n),
the space of r-dimensional subspaces of R^n. B matrices remain Euclidean.
See grassmann_utils.py for the mathematical background.
"""

import torch
import copy

from grassmann_utils import (
    to_grassmann,
    grassmann_frechet_step,
    geodesic_distance,
)


# Names of the four LoRA layers in EEGNeX LoRA mode
LORA_LAYER_NAMES = ['block_1_1', 'block_2_0', 'block_4_1', 'block_5_0']


class MetaInit:
    """
    Stores meta-initialisation tensors for all LoRA adapter layers.
    These are plain tensors — not nn.Parameters — so they sit outside
    the model's parameter graph and are updated via the Reptile rule
    rather than by an optimiser.

    Supports both Euclidean Reptile (original) and Riemannian Reptile on
    the Grassmann manifold (new). The two update methods are independent
    and can be compared by passing the same MetaInit to different trainers.
    """

    def __init__(self, model, device):
        """
        Initialise meta weights by copying the initial adapter weights
        from adapter slot 0 of the model.
        B weights start at zero (matching existing initialisation).
        A weights start at N(0, 0.02) (matching existing initialisation).
        """
        self.device = device
        self.weights = {}   # {layer_name: {'A': tensor, 'B': tensor}}

        for name in LORA_LAYER_NAMES:
            layer = getattr(model, name)
            # Copy from slot 0 — shapes are identical across slots
            self.weights[name] = {
                'A': layer.lora_A[0].weight.data.clone().to(device),
                'B': layer.lora_B[0].weight.data.clone().to(device),
            }

    # ── Slot I/O (shared by both update modes) ────────────────────────────────

    def load_into_slot(self, model, slot):
        """
        Copy meta weights into adapter slot `slot` of the model.
        Used at the start of each inner loop iteration and at test time.
        """
        with torch.no_grad():
            for name in LORA_LAYER_NAMES:
                layer = getattr(model, name)
                layer.lora_A[slot].weight.data.copy_(self.weights[name]['A'])
                layer.lora_B[slot].weight.data.copy_(self.weights[name]['B'])

    def extract_from_slot(self, model, slot):
        """
        Read adapter weights from slot `slot` back into a dict of tensors.
        Used after the inner loop to collect phi_K.
        Returns a dict with same structure as self.weights.
        """
        phi = {}
        for name in LORA_LAYER_NAMES:
            layer = getattr(model, name)
            phi[name] = {
                'A': layer.lora_A[slot].weight.data.clone(),
                'B': layer.lora_B[slot].weight.data.clone(),
            }
        return phi

    # ── Euclidean Reptile update (original, unchanged) ────────────────────────

    def reptile_update(self, phi_list, meta_lr):
        """
        Apply Reptile update rule given a list of phi dicts
        (one per training subject processed in this iteration).

        theta_meta += meta_lr * mean(phi_s - theta_meta)

        phi_list: list of dicts returned by extract_from_slot
        meta_lr:  Reptile outer step size (epsilon)
        """
        n = len(phi_list)
        with torch.no_grad():
            for name in LORA_LAYER_NAMES:
                delta_A = sum(
                    phi[name]['A'] - self.weights[name]['A']
                    for phi in phi_list
                ) / n

                delta_B = sum(
                    phi[name]['B'] - self.weights[name]['B']
                    for phi in phi_list
                ) / n

                self.weights[name]['A'] += meta_lr * delta_A
                self.weights[name]['B'] += meta_lr * delta_B

    def delta_norm(self, phi_list):
        """
        Compute the norm of the average Euclidean delta across subjects.
        Used as a health-check metric during training.
        """
        n = len(phi_list)
        total = 0.0
        for name in LORA_LAYER_NAMES:
            delta_A = sum(
                phi[name]['A'] - self.weights[name]['A']
                for phi in phi_list
            ) / n
            delta_B = sum(
                phi[name]['B'] - self.weights[name]['B']
                for phi in phi_list
            ) / n
            total += delta_A.norm().item() + delta_B.norm().item()
        return total

    # ── Grassmann shape helpers ───────────────────────────────────────────────

    def _flatten_A(self, A_tensor):
        """
        Reshape Conv2d weight (r, Cin, kH, kW) → (n, r) for Grassmann ops.
        n = Cin * kH * kW — the total input feature dimension per filter.
        The column space of (n, r) is a point on G(r, n).
        """
        r = A_tensor.shape[0]
        return A_tensor.reshape(r, -1).T   # (n, r)

    def _unflatten_A(self, A_flat, original_shape):
        """
        Restore (n, r) back to Conv2d weight shape (r, Cin, kH, kW).
        Inverse of _flatten_A.
        """
        r = original_shape[0]
        return A_flat.T.reshape(original_shape)   # (r, Cin, kH, kW)

    # ── Grassmann Reptile update (new) ────────────────────────────────────────

    def reptile_update_grassmann(self, phi_list, meta_lr):
        """
        Riemannian Reptile update on the Grassmann manifold.

        A matrices (input projection, defines the adapter subspace):
            Updated via Riemannian Reptile on G(r, n).
            The subspace direction is moved along the geodesic toward the
            Fréchet mean of the fine-tuned subject subspaces.
            The scale (||A||_F) is updated separately via Euclidean Reptile.

        B matrices (output projection, maps bottleneck to output):
            Standard Euclidean Reptile — same as reptile_update().
            Less motivated theoretically for Grassmann treatment since B
            does not define an input subspace in the same sense.

        The combined effect: the column space of A_meta moves toward the
        geometric centre of the subjects' fine-tuned column spaces, rather
        than the arithmetic mean of their matrix entries.

        phi_list: list of dicts returned by extract_from_slot
        meta_lr:  Reptile outer step size (epsilon)
        """
        n = len(phi_list)
        with torch.no_grad():
            for name in LORA_LAYER_NAMES:

                # ── A: Riemannian update on Grassmann ─────────────────────────
                A_meta      = self.weights[name]['A']
                A_meta_flat = self._flatten_A(A_meta)           # (n_feat, r)

                # Project meta to Grassmann: orthonormal basis + scale
                Q_meta, scale_meta = to_grassmann(A_meta_flat)

                # Project each subject's fine-tuned A to Grassmann
                Q_list = []
                scales = []
                for phi in phi_list:
                    A_s_flat     = self._flatten_A(phi[name]['A'])
                    Q_s, scale_s = to_grassmann(A_s_flat)
                    Q_list.append(Q_s)
                    scales.append(scale_s)

                # Move Q_meta along geodesic toward Fréchet mean of {Q_s}
                Q_meta_new = grassmann_frechet_step(Q_meta, Q_list, meta_lr)

                # Update scale via Euclidean Reptile (separate from direction)
                mean_scale = sum(scales) / n
                scale_new  = scale_meta + meta_lr * (mean_scale - scale_meta)

                # Reconstruct A_meta: scaled orthonormal basis
                A_flat_new = scale_new * Q_meta_new
                self.weights[name]['A'].copy_(
                    self._unflatten_A(A_flat_new, A_meta.shape)
                )

                # ── B: Euclidean update (unchanged from original) ──────────────
                delta_B = sum(
                    phi[name]['B'] - self.weights[name]['B']
                    for phi in phi_list
                ) / n
                self.weights[name]['B'] += meta_lr * delta_B

    def delta_norm_grassmann(self, phi_list):
        """
        Geodesic health-check metric for the Grassmann update.

        Reports the mean geodesic distance from Q_meta to each subject's Q_s
        (for A matrices) plus the Euclidean B delta norm.

        The geodesic distance is the root-sum-squares of principal angles
        between the two subspaces — a coordinate-free measure of how far
        each subject's adapter has moved in terms of the subspace it spans.
        """
        n = len(phi_list)
        total_geo = 0.0
        total_b   = 0.0

        for name in LORA_LAYER_NAMES:
            A_meta_flat  = self._flatten_A(self.weights[name]['A'])
            Q_meta, _    = to_grassmann(A_meta_flat)

            for phi in phi_list:
                A_s_flat = self._flatten_A(phi[name]['A'])
                Q_s, _   = to_grassmann(A_s_flat)
                total_geo += geodesic_distance(Q_meta, Q_s)

            delta_B   = sum(
                phi[name]['B'] - self.weights[name]['B']
                for phi in phi_list
            ) / n
            total_b += delta_B.norm().item()

        mean_geo = total_geo / (n * len(LORA_LAYER_NAMES))
        return mean_geo + total_b

    # ── Utilities (shared) ────────────────────────────────────────────────────

    def zero_(self):
        """
        Reset meta weights to zero — used for baseline comparison
        to confirm zero-initialisation matches original paper behaviour.
        """
        with torch.no_grad():
            for name in LORA_LAYER_NAMES:
                self.weights[name]['A'].zero_()
                self.weights[name]['B'].zero_()


# ── Parameter helpers (unchanged) ─────────────────────────────────────────────

def get_adapter_params(model, slot):
    """
    Return all adapter parameters for a given slot as a list.
    Used to build an optimiser that only updates one subject's adapters.
    """
    params = []
    for name in LORA_LAYER_NAMES:
        layer = getattr(model, name)
        params.append(layer.lora_A[slot].weight)
        params.append(layer.lora_B[slot].weight)
    return params


def freeze_backbone(model):
    """
    Freeze all parameters except LoRA adapters.
    Used during few-shot adaptation at test time.
    """
    for name, param in model.named_parameters():
        is_adapter = any(
            f'{lname}.lora_A' in name or f'{lname}.lora_B' in name
            for lname in LORA_LAYER_NAMES
        )
        param.requires_grad = is_adapter


def unfreeze_all(model):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
