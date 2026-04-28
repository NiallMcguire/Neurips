"""
MetaInit: stores and manages the Reptile meta-initialisation for LoRA adapters.

The four LoRAConvPerSubject layers in EEGNeX LoRA mode are:
    block_1_1, block_2_0, block_4_1, block_5_0

Each has lora_A (ModuleList of Conv2d) and lora_B (ModuleList of Conv2d).

MetaInit holds one set of (A_weight, B_weight) tensors per layer,
representing the best initialisation found by Reptile for a new subject.

Key operations:
    load_into_slot(model, slot)   — copy meta weights into adapter slot i
    extract_from_slot(model, slot) — read adapter slot i back into meta weights
    reptile_update(phi, meta_lr)  — move meta toward phi via Reptile rule
"""

import torch
import copy


# Names of the four LoRA layers in EEGNeX LoRA mode
LORA_LAYER_NAMES = ['block_1_1', 'block_2_0', 'block_4_1', 'block_5_0']


class MetaInit:
    """
    Stores meta-initialisation tensors for all LoRA adapter layers.
    These are plain tensors — not nn.Parameters — so they sit outside
    the model's parameter graph and are updated via the Reptile rule
    rather than by an optimiser.
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
                # Average delta across subjects
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
        Compute the norm of the average delta across subjects.
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

    def zero_(self):
        """
        Reset meta weights to zero — used for baseline comparison
        to confirm zero-initialisation matches original paper behaviour.
        """
        with torch.no_grad():
            for name in LORA_LAYER_NAMES:
                self.weights[name]['A'].zero_()
                self.weights[name]['B'].zero_()


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
