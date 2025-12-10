"""reproduce annex G"""
import numpy as np
import sys, os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class AppendixGParams:
    n_cells_per_condition: int = 50
    n_perturbations: int = 9   # plus one untreated/control condition
    d_latent: int = 5
    pX: int = 50
    pY: int = 200
    include_control: bool = True
    seed: Optional[int] = 0

def sample_sigma_from_log_normal(mean_log: float, var_log: float, size, rng: np.random.Generator) -> np.ndarray:
    # Here var_log is variance; since Appendix G sets variance 0, we keep it for completeness.
    # the paper writes log σ ~ N(-3, 0) so σ ~ exp(N(-3, 0))
    if var_log == 0:
        return np.exp(np.full(size, mean_log)) # no variance around the mean, all identical
    return np.exp(rng.normal(loc=mean_log, scale=np.sqrt(var_log), size=size))

def simulate_appendix_g(params: AppendixGParams) -> Dict[str, Any]:
    """
    Reproduce the Appendix G synthetic data generation from Ryu/Bunne/Pinello/Regev/Lopez.
    see https://arxiv.org/abs/2405.00838
    
    Returns a dictionary with generated matrices and metadata:
      - X, Y: concatenated observations across conditions
      - labels: condition label per cell (strings: 'control' or 'perturb_k')
      - perturb_index: integer per cell, -1 for control else 1..n_perturbations
      - targeted_dim_per_perturb: dict[perturbation_index] -> targeted latent dimension (1-indexed)
      - effect_size_per_perturb: dict[perturbation_index] -> e(I)
      - q: per-cell penetrance (Beta(1,10))
      - latent_Z: base latent vectors per cell
      - latent_Z_perturbed: perturbed latent vectors per cell
      - AX, AY, bX, bY, sX, sY, zetaX, zetaY: parameters used
    """
    rng = np.random.default_rng(params.seed)
    
    # counts
    n_perturbation = params.n_perturbations + (1 if params.include_control else 0)
    n_cells = n_perturbation * params.n_cells_per_condition
    
    # Draw shared modality parameters (AX, AY, b, s) — once for the whole dataset
    d, pX, pY = params.d_latent, params.pX, params.pY
    AX = rng.normal(0, 1, size=(d, pX))    # AX ~ N(0,1)
    AY = rng.normal(0, 1, size=(d, pY))    # AY ~ N(0,1)
    bX = rng.normal(0, 1, size=(pX,))      # bX ~ N(0,1)
    bY = rng.normal(0, 1, size=(pY,))      # bY ~ N(0,1)
    sX = rng.gamma(shape=1.0, scale=1.0, size=(pX,)) # sX ~ Gamma(1,1)
    sY = rng.gamma(shape=1.0, scale=1.0, size=(pY,)) # sY ~ Gamma(1,1)
    
    # modality-specific latent perturbations: ζX, ζY with μ ~ N(0,1), log σ ~ N(-3, 0)
    muX = rng.normal(0, 1, size=(d,))
    muY = rng.normal(0, 1, size=(d,))
    sigX = sample_sigma_from_log_normal(mean_log=-3.0, var_log=0.0, size=(d,), rng=rng)
    sigY = sample_sigma_from_log_normal(mean_log=-3.0, var_log=0.0, size=(d,), rng=rng)
    zetaX = rng.normal(muX, sigX, size=(d,))  # ζX ~ N(μX, σX)
    zetaY = rng.normal(muY, sigY, size=(d,))  # ζY ~ N(μY, σY)
    
    # as in paper: forward maps fX and fY
    def fX(Z: np.ndarray) -> np.ndarray:
        # X = ((Z + ζX) AX + bX) sX    (feature-wise scaling by sX)
        return ((Z + zetaX) @ AX + bX) * sX
    
    def fY(Z: np.ndarray) -> np.ndarray:
        # Y = ((Z + ζY) AY + bY) sY    (feature-wise scaling by sY)
        return ((Z + zetaY) @ AY + bY) * sY
    
    # variables to store everything
    X_list, Y_list, labels, perturb_index, q_list = [], [], [], [], []
    Z_base_list, Z_pert_list = [], []
    
    # Target dimension per perturbation: t(I) = ((I-1) mod d) + 1   (1-indexed in paper)
    targeted_dim_per_perturb: Dict[int, int] = {}
    
    # Effect size per perturbation: e(I) = max(3, Gamma(1,1))
    effect_size_per_perturb: Dict[int, float] = {}
    
    # Iterate conditions
    perturb_counter = 0
    for cond_idx in range(n_perturbation):
        is_control = (params.include_control and cond_idx == 0)
        # Base latent Z ~ N(0, 0.1)  (variance 0.1 across latent dims)
        Z = rng.normal(0.0, np.sqrt(0.1), size=(params.n_cells_per_condition, d))
        
        if is_control:
            Zp = Z.copy()  # no additional transformations
            label_str = "control"
            curr_pert_index = -1
            # For control, draw no q (but keep shape for uniformity as zeros)
            q_vals = np.zeros(params.n_cells_per_condition, dtype=float)
        else:
            perturb_counter += 1
            I = perturb_counter
            # Determine target dimension, 1-indexed in paper; convert to 0-indexed here because python
            tI_1indexed = ((I - 1) % d) + 1
            tI = tI_1indexed - 1
            targeted_dim_per_perturb[I] = tI_1indexed
            
            # effect size e(I) >= 3, drawn from Gamma(1,1)
            eI = max(3.0, rng.gamma(shape=1.0, scale=1.0))
            effect_size_per_perturb[I] = float(eI)
            
            # Per-cell penetrance q_i ~ Beta(1,10)
            q_vals = rng.beta(a=1.0, b=10.0, size=(params.n_cells_per_condition,))
            
            # Apply shift to targeted dimension: z_i^I = z_i with z[t(I)] += e(I)*q_i
            Zp = Z.copy()
            Zp[:, tI] += eI * q_vals
            
            label_str = f"perturb_{I}"
            curr_pert_index = I
        
        # Generate observations
        X_block = fX(Zp)
        Y_block = fY(Zp)
        
        # Collect
        X_list.append(X_block)
        Y_list.append(Y_block)
        labels.extend([label_str] * params.n_cells_per_condition)
        perturb_index.extend([curr_pert_index] * params.n_cells_per_condition)
        q_list.append(q_vals)
        Z_base_list.append(Z)
        Z_pert_list.append(Zp)
    
    # Concatenate across conditions
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    q = np.concatenate(q_list)
    Z_base = np.vstack(Z_base_list)
    Z_pert = np.vstack(Z_pert_list)
    labels_arr = np.array(labels)
    perturb_index_arr = np.array(perturb_index, dtype=int)
    
    return {
        "X": X, "Y": Y,
        "labels": labels_arr,
        "perturb_index": perturb_index_arr,
        "targeted_dim_per_perturb": targeted_dim_per_perturb,
        "effect_size_per_perturb": effect_size_per_perturb,
        "q": q,
        "latent_Z": Z_base,
        "latent_Z_perturbed": Z_pert,
        "AX": AX, "AY": AY, "bX": bX, "bY": bY, "sX": sX, "sY": sY,
        "zetaX": zetaX, "zetaY": zetaY,
        "params": asdict(params),
    }

if __name__ == "__main__":
    from sklearn.decomposition import PCA
    path_perturbot = "/home/phalafail/MVA/Geometric_data_analysis_project/Perturb-OT/perturbot/"
    if path_perturbot not in sys.path:
        sys.path.append(path_perturbot)
    
    from perturbot.match import (
        get_coupling_cotl, 
        get_coupling_cotl_sinkhorn, 
        get_coupling_egw_labels_ott,
        get_coupling_egw_all_ott,
        get_coupling_eot_ott,
        get_coupling_leot_ott,
        get_coupling_egw_ott,
        get_coupling_cot, 
        get_coupling_cot_sinkhorn, 
        get_coupling_gw_labels,
        get_coupling_fot,
    )
    from perturbot.predict import train_mlp

    params = AppendixGParams(
        n_cells_per_condition=50,
        n_perturbations=9,      # plus one control if include_control=True
        d_latent=5, pX=1000, pY=2000,
        include_control=True,
        seed=0
    )
    data = simulate_appendix_g(params)
    X, Y = data["X"], data["Y"] # shapes: (500, 50), (500, 200)
    labels = data["labels"]  # e.g. 'control', 'perturb_1', etc
    translate_labels = {str(label):i for i, label in enumerate(np.unique(labels))}
    num_labels = np.array([translate_labels[str(label)] for label in labels])

    X_dict = {k: X[num_labels == k] for k in num_labels}
    Y_dict = {k: Y[num_labels == k] for k in num_labels}

    pca = PCA(n_components=50)
    X_reduced = {k: pca.fit_transform(X[num_labels == k]) for k in num_labels}
    Y_reduced = {k: pca.fit_transform(Y[num_labels == k]) for k in num_labels}

    q = data["q"] # penetrance

    # Return a brief summary for the user
    summary = {
        "X_shape": data["X"].shape,
        "Y_shape": data["Y"].shape,
        "n_labels": len(np.unique(data["labels"])),
        "labels_counts": {label: int(np.sum(data["labels"] == label)) for label in np.unique(data["labels"])}
    }

    print(summary)

    # Learn matching in the latent space
    T_dict, log = get_coupling_egw_labels_ott((X_reduced, Y_reduced)) # Other get_coupling_X methods be used

    # # Train MLP based on matching
    model, pred_log = train_mlp((X_dict, Y_dict), T_dict)

    # Learn feature-feature matching
    T_feature, fm_log = get_coupling_fot((X_dict, Y_dict), T_dict)
