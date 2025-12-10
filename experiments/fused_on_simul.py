import numpy as np
import ot

def generate_synthetic_data(n_cells_per_pert=50, n_perturbations=10, d_latent=5, p_X=50, p_Y=200):
    """Génère les données synthétiques décrites en Annexe G """
    print(f"Génération de {n_cells_per_pert * n_perturbations} cellules synthétiques...")
    n_total_cells = n_cells_per_pert * n_perturbations

    # Z ~ N(0, 0.1)
    Z_base = np.random.normal(0.0, np.sqrt(0.1), (n_total_cells, d_latent))

    # A_X, A_Y, b_X, b_Y, s_X, s_Y
    A_X = np.random.normal(0.0, 1.0, (d_latent, p_X))
    A_Y = np.random.normal(0.0, 1.0, (d_latent, p_Y))
    b_X = np.random.normal(0.0, 1.0, (p_X,))
    b_Y = np.random.normal(0.0, 1.0, (p_Y,))
    s_X = np.random.gamma(1.0, 1.0, (p_X,))
    s_Y = np.random.gamma(1.0, 1.0, (p_Y,))

    # zeta_X, zeta_Y (Bruit technique) [cite: 871-877]
    std_X = np.sqrt(np.exp(np.random.normal(-3, 0)))
    zeta_X = np.random.normal(0, std_X, (n_total_cells, d_latent))
    std_Y = np.sqrt(np.exp(np.random.normal(-3, 0)))
    zeta_Y = np.random.normal(0, std_Y, (n_total_cells, d_latent))

    # Labels et application des perturbations
    labels = np.repeat(np.arange(n_perturbations), n_cells_per_pert)
    target_dims = np.zeros(n_perturbations, dtype=int)
    target_dims[1:] = (np.arange(n_perturbations - 1) % d_latent) # [cite: 883]
    effect_sizes = np.zeros(n_perturbations)
    effect_sizes[1:] = np.maximum(3.0, np.random.gamma(1.0, 1.0, n_perturbations - 1)) # [cite: 887]
    penetrance = np.random.beta(1.0, 10.0, n_total_cells) # [cite: 889]

    Z_perturbed = Z_base.copy()
    for i in range(n_total_cells):
        label_idx = labels[i]
        if label_idx > 0: # 0 est le contrôle
            target_dim = target_dims[label_idx]
            effect = effect_sizes[label_idx]
            Z_perturbed[i, target_dim] += effect * penetrance[i] # [cite: 891]

    # Génération finale 
    Z_noisy_X = Z_perturbed + zeta_X
    Z_noisy_Y = Z_perturbed + zeta_Y
    X = ((Z_noisy_X @ A_X) + b_X) * s_X
    Y = ((Z_noisy_Y @ A_Y) + b_Y) * s_Y

    return X, Y, labels

def format_data_for_coupling(X, Y, labels):
    """
    Convertit les matrices X, Y et le tableau de labels
    en un format de dictionnaire attendu par les fonctions de couplage.
    """
    X_dict = {}
    Y_dict = {}
    unique_labels = np.unique(labels)
    
    for l_numpy in unique_labels:
        # 1. Convertit le type numpy (ex: np.int64) en int Python
        l_python = int(l_numpy) 
        
        # 2. Trouve les indices en utilisant la clé numpy d'origine
        indices = np.where(labels == l_numpy)[0]
        
        # 3. Stocke avec la clé int Python propre
        X_dict[l_python] = X[indices]
        Y_dict[l_python] = Y[indices]
        
    return (X_dict, Y_dict)

if __name__ == "__main__":
    
    # 1. Générer toutes les données (0-9)
    X_full, Y_full, labels_full = generate_synthetic_data(
        n_cells_per_pert=50, 
        n_perturbations=10, 
        d_latent=5, 
        p_X=50, 
        p_Y=200
    )
  
    n_total = X_full.shape[0]
    test_frac = 0.2  # 20% pour le test
    n_test = int(n_total * test_frac)

    # Mélange aléatoire des indices
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    # Indices train / test
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Création des splits
    X_train = X_full[train_indices]
    Y_train = Y_full[train_indices]
    labels_train = labels_full[train_indices]

    X_test = X_full[test_indices]
    Y_test = Y_full[test_indices]
    labels_test = labels_full[test_indices]

    #réordonner pour afficher joliment

    sort_idx_train = np.argsort(labels_train)
    X_train = X_train[sort_idx_train]
    Y_train = Y_train[sort_idx_train]
    labels_train = labels_train[sort_idx_train]

    sort_idx_test = np.argsort(labels_test)
    X_test = X_test[sort_idx_test]
    Y_test = Y_test[sort_idx_test]
    labels_test = labels_test[sort_idx_test]



    #data_tuple_train = format_data_for_coupling(
    #    X_train, 
    #    Y_train, 
    #    labels_train 
    #)
    #data_tuple_test = format_data_for_coupling(
    #    X_test, 
    #    Y_test, 
    #    labels_test 
    #)


    C1 = ot.dist(X_train, X_train, metric='euclidean')**2
    C2 = ot.dist(Y_train, Y_train, metric='euclidean')**2
    C1=C1/C1.max()
    C2=C2/C2.max()

    # -----------------------------
    # 2. Matrice de coût fused M
    # -----------------------------
    # Ici M[i,j] = 1 si labels différents, 0 sinon
    n_train = X_train.shape[0]
    m_train = Y_train.shape[0]
    labels_X = labels_train[:, np.newaxis]  # shape (n_train, 1)
    labels_Y = labels_train[np.newaxis, :]  # shape (1, m_train)

    # M = 1 si labels différents
    M = (labels_X != labels_Y).astype(float)

    epsilon = 0.001
    alpha = 0.99
    max_iter = 500

    print("Calcul du transport plan Fused Gromov-Wasserstein (alpha=1 -> GW uniquement)...")
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M, C1, C2, alpha=alpha, epsilon=epsilon, max_iter=max_iter, verbose=True
    )

    print("C1 :", C1.shape, C1)
    print("C2 :", C2.shape, C2)
    print("M :",M.shape,  M)

    print(f"Transport plan T (shape): {T.shape}")
    print(T)
    import matplotlib.pyplot as plt
    plt.imshow(T, cmap='viridis')
    plt.colorbar()
    plt.title(f"Coupling for alpha = {alpha}, epsilon = {epsilon}")
    plt.show()