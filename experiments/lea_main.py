import sys
import os
import numpy as np
from sklearn.decomposition import PCA

dossier_actuel = os.path.dirname(os.path.abspath(__file__))
chemin_projet = os.path.join(dossier_actuel, "Perturb-OT-main", "perturbot", "perturbot", "match")
sys.path.append(chemin_projet)

print(f"Chemin ajouté à Python : {chemin_projet}")

from ott_egwl import (
    get_coupling_egw_labels_ott
)

chemin_predict = os.path.join(dossier_actuel, "Perturb-OT-main", "perturbot", "perturbot", "predict")
sys.path.append(chemin_predict)
print(f"Chemin ajouté : {chemin_predict}")


import mlp

## Génération des données synthétiques
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

# --- 2. Formatage et Séparation ---

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
    
    # 2. Créer les ensembles d'entraînement (0-7) et de test (8-9)
  
    
    train_labels_list = np.arange(8) # Labels 0 à 7
    test_labels_list = np.arange(8, 10) # Labels 8 et 9

    train_mask = np.isin(labels_full, train_labels_list)
    test_mask = np.isin(labels_full, test_labels_list)

    X_train_array = X_full[train_mask]
    Y_train_array = Y_full[train_mask]
    labels_train_array = labels_full[train_mask]
    
    X_test_array = X_full[test_mask]
    Y_test_array = Y_full[test_mask]
    labels_test_array = labels_full[test_mask]

    data_tuple_train = format_data_for_coupling(
        X_train_array, 
        Y_train_array, 
        labels_train_array 
    )
    data_tuple_test = format_data_for_coupling(
        X_test_array, 
        Y_test_array, 
        labels_test_array 
    )


    print("\n--- ÉTAPE 1 : Entraînement du LabeledEGWOT ---")
    
 
    lgw = get_coupling_egw_labels_ott(data_tuple_train)

    print("Entraînement du couplage terminé.")
    print(lgw)

    predictor = MLP() 
    predictor.fit(lgw[0], data_tuple_train)
    print("Prédicteur entraîné.")
    Y_predicted_dict = predictor.predict(data_tuple_test[0])

    print("Prédiction terminée !")
    print(f"Labels prédits: {list(Y_predicted_dict.keys())}")

    pred_label_8 = Y_predicted_dict[8]
    true_label_8 = data_tuple_test[1][8]
    print(f"Forme de Y prédit (label 8): {pred_label_8.shape}")
    print(f"Forme de Y réel (label 8):   {true_label_8.shape}")
