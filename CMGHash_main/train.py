from tqdm import tqdm
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from LoadData_CMG import *
from sklearn.preprocessing import StandardScaler # for two Amazon datasets only
import yaml
import math
import traceback
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")

def calculate_map(binary_codes_np, ground_truth_labels_np, top_k=None, batch_size_map=1000):
    """
    Calculates mAP for given binary hash codes and true labels.
    :param binary_codes_np: (N, K_BITS) numpy array, N samples of K_BITS hash codes (-1 or 1)
    :param ground_truth_labels_np: (N,) numpy array, N samples of true class labels
    :param top_k: Calculate mAP@top_k, if None considers all retrieved results
    :param batch_size_map: Batch size for mAP calculation to prevent OOM errors
    :return: mAP value
    """
    num_samples, num_bits = binary_codes_np.shape
    mAP = 0.0
    num_batches = (num_samples + batch_size_map - 1) // batch_size_map

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size_map
        end_idx = min((batch_idx + 1) * batch_size_map, num_samples)
        current_batch_size = end_idx - start_idx

        if current_batch_size == 0:
            continue

        query_codes_batch = binary_codes_np[start_idx:end_idx]
        query_labels_batch = ground_truth_labels_np[start_idx:end_idx]

        batch_AP_sum = 0.0
        for i in range(current_batch_size):
            query_code = query_codes_batch[i]
            query_label = query_labels_batch[i]
            query_original_idx = start_idx + i

            distances = np.zeros(num_samples)
            for db_idx in range(num_samples):
                distances[db_idx] = 0.5 * (num_bits - np.dot(query_code, binary_codes_np[db_idx]))

            sorted_indices = np.argsort(distances)
            retrieved_indices = [idx for idx in sorted_indices if idx != query_original_idx]

            if top_k is not None:
                retrieved_indices = retrieved_indices[:top_k]

            num_retrieved = len(retrieved_indices)
            if num_retrieved == 0:
                continue

            relevant_mask = (ground_truth_labels_np[retrieved_indices] == query_label)
            num_relevant = np.sum(relevant_mask)

            if num_relevant == 0:
                continue

            precision_at_k = np.cumsum(relevant_mask) / (np.arange(num_retrieved) + 1)
            average_precision = np.sum(precision_at_k * relevant_mask) / num_relevant
            batch_AP_sum += average_precision
        mAP += batch_AP_sum
    return mAP / num_samples

class HEN_PyTorch(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        """
        PyTorch Hash Encoding Network (HEN).
        :param input_dim: Input feature dimension.
        :param output_dim: Output hash code bits (K_BITS).
        :param hidden_dim: Optional hidden layer dimension. If None, a single linear layer is used.
        """
        super(HEN_PyTorch, self).__init__()
        if hidden_dim and hidden_dim > 0:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            print(f"HEN initialized with hidden layer: {hidden_dim}")
        else:
            self.fc1 = nn.Linear(input_dim, output_dim)
            print(f"HEN initialized as single linear layer.")
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Forward propagation.
        :param x: Input feature tensor (N, input_dim).
        :return: Continuous hash embeddings (N, output_dim), range [-1, 1].
        """
        if hasattr(self, 'fc2'):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
        else:
            x = self.fc1(x)
        return self.tanh(x)

def go_run(dataname, X_list_original, gnd_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        with open('config.yaml', 'r') as f_conf:
            config_data = yaml.safe_load(f_conf)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure it exists in the current directory.")
        return 0.0

    dataset_config = config_data.get(dataname, {})
    K_BITS = dataset_config.get('K_BITS', 16)
    alpha = dataset_config.get('BETA', 0.1)  
    beta = dataset_config.get('DELTA', 0.1)   
    TAU = dataset_config.get('TAU', 0.4)
    HEN_HIDDEN_DIM_config = dataset_config.get('HEN_HIDDEN_DIM', None)
    w_con = dataset_config.get('alpha', 0.5) 
    gamma_view_reg = float(dataset_config.get('gama', -4.0))
    eta_view_reg_coeff = dataset_config.get('ETA', 2.0)
    learning_rate_hen = dataset_config.get('lr_hen', 0.005)
    epochs = dataset_config.get('epochs', 50)
    eval_interval = dataset_config.get('eval_interval', 1)
    k_nn_val = dataset_config.get('k_nn', 10)

    N = X_list_original[0].shape[0]
    I_np = np.eye(N)

    num_view_from_data = len(X_list_original) - 1
    print(f"Now {dataname} are tested! Data provides {num_view_from_data} graph view(s).")

    H_filtered_list_np = []
    common_feature = X_list_original[0].copy()
    num_actual_views_for_loss = num_view_from_data
    for v_idx in range(num_actual_views_for_loss):
        A_orig = X_list_original[v_idx + 1].copy()
        A_proc = A_orig + I_np
        D_sum = np.sum(A_proc, axis=1)
        D_inv_sqrt = np.power(D_sum, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt_diag = np.diagflat(D_inv_sqrt)
        A_norm = D_inv_sqrt_diag.dot(A_proc).dot(D_inv_sqrt_diag)
        Ls = I_np - A_norm

        H_v_filtered = common_feature.copy()
        for _ in range(config_data.get('graph_filter_m', 2)):
            H_v_filtered = (I_np - config_data.get('graph_filter_s', 0.5) * Ls).dot(H_v_filtered)
        H_filtered_list_np.append(H_v_filtered)

    lambda_view_weights_np = np.array([1.0 / num_actual_views_for_loss for _ in range(num_actual_views_for_loss)], dtype=np.float32)
    print('Graph filtering complete.')

    nbrs_inx_list_np = []
    knn_dir = "./knn_data"
    if not os.path.exists(knn_dir):
        os.makedirs(knn_dir)

    try:
        for v_idx in range(num_actual_views_for_loss):
            idx_path = os.path.join(knn_dir, f"nbrs{k_nn_val}_{dataname}_view{v_idx}.npy")
            idx = np.load(idx_path).astype(np.int_)
            nbrs_inx_list_np.append(idx)
            print(f"Loaded kNN for view {v_idx} from {idx_path}")
    except Exception as e:
        print(f"Failed to load kNN files (Error: {e}). Computing kNN now...")
        for v_idx in range(num_actual_views_for_loss):
            X_nb_for_knn = np.array(H_filtered_list_np[v_idx])
            sklearn_nbrs = NearestNeighbors(n_neighbors=k_nn_val + 1, algorithm='auto').fit(X_nb_for_knn)
            _, idx = sklearn_nbrs.kneighbors(X_nb_for_knn)
            nbrs_v_current = idx[:, 1:k_nn_val+1].astype(np.int_)

            # save_path = os.path.join(knn_dir, f"nbrs{k_nn_val}_{dataname}_view{v_idx}.npy")
            # np.save(save_path, nbrs_v_current)
            # print(f"Saved kNN for view {v_idx} to {save_path}")
            nbrs_inx_list_np.append(nbrs_v_current)

    nbrs_inx_list_torch = [torch.from_numpy(n).long().to(device) for n in nbrs_inx_list_np]
    
    
    """ For the two Amazon datasets, the input layer requires normalization to avoid excessively large values.
    
    # std_val = H_concatenated_np.std()
    # formatted_std = std_val if std_val > 1e-9 else 0.0
    # print(f"Original H_concatenated_np stats: min={H_concatenated_np.min():.4f}, max={H_concatenated_np.max():.4f}, mean={H_concatenated_np.mean():.4f}, std={formatted_std:.4f}")
    # scaler = StandardScaler()
    # H_concatenated_np_scaled = scaler.fit_transform(H_concatenated_np)
    # hen_input_dim = H_concatenated_np_scaled.shape[1]
    
    # hen_model = HEN_PyTorch(input_dim=hen_input_dim, output_dim=K_BITS, hidden_dim=HEN_HIDDEN_DIM_config).to(device)
    # optimizer_hen = optim.Adam(hen_model.parameters(), lr=learning_rate_hen, weight_decay=dataset_config.get('weight_decay', 1e-5))   
       
    """

    H_concatenated_np = np.concatenate(H_filtered_list_np, axis=1) if len(H_filtered_list_np) > 1 else H_filtered_list_np[0]
    hen_input_dim = H_concatenated_np.shape[1] 

    hen_model = HEN_PyTorch(input_dim=hen_input_dim, output_dim=K_BITS, hidden_dim=HEN_HIDDEN_DIM_config).to(device)
    optimizer_hen = optim.Adam(hen_model.parameters(), lr=learning_rate_hen)

    H_input_torch = torch.from_numpy(H_concatenated_np).float().to(device)
    gnd_torch = torch.from_numpy(gnd_np).long().to(device)

    best_U_continuous_torch = torch.zeros((N, K_BITS), device=device)
    best_mAP = 0.0
    best_epoch_mAP = 0

    print(f"CMGHash Config: w_con={w_con}, gamma_vr={gamma_view_reg}, K={K_BITS}, alpha={alpha}, beta={beta}, TAU={TAU}, LR_HEN={learning_rate_hen}, k_nn={k_nn_val}")
    loss_last_CMGHash = float('inf')

    for epoch in range(epochs):
        hen_model.train()
        optimizer_hen.zero_grad()

        U_continuous_torch = hen_model(H_input_torch)

        total_L_CH_value_torch = torch.tensor(0.0, device=device)
        U_norm_torch = F.normalize(U_continuous_torch, p=2, dim=1)
        sim_matrix_torch = torch.matmul(U_norm_torch, U_norm_torch.T)

        for v_idx in range(num_actual_views_for_loss):
            L_CH_v_epoch_torch = torch.tensor(0.0, device=device)
            lambda_v_current = torch.tensor(lambda_view_weights_np[v_idx], device=device, dtype=torch.float32)

            for i in range(N):
                positive_indices = nbrs_inx_list_torch[v_idx][i]
                sim_i_positives = sim_matrix_torch[i, positive_indices]

                mask_not_i = torch.ones(N, dtype=torch.bool, device=device)
                mask_not_i[i] = False
                sim_i_all_others = sim_matrix_torch[i, mask_not_i]

                if sim_i_all_others.numel() == 0:
                    log_sum_exp_negatives_i = torch.tensor(float('-inf'), device=device)
                else:
                    log_sum_exp_negatives_i = torch.logsumexp(sim_i_all_others / TAU, dim=0)

                log_softmax_val = sim_i_positives / TAU - log_sum_exp_negatives_i
                L_CH_v_epoch_torch -= torch.sum(log_softmax_val)

            total_L_CH_value_torch += lambda_v_current * (L_CH_v_epoch_torch / N)

        B_binary_current_torch = torch.sign(U_continuous_torch)
        L_Q_value_torch = torch.mean((U_continuous_torch - B_binary_current_torch)**2)

        L_BB_value_torch = torch.tensor(0.0, device=device)
        if beta > 1e-9: 
            mean_U_k_torch = torch.mean(U_continuous_torch, dim=0)
            L_BB_value_torch = torch.mean(mean_U_k_torch**2)

        total_loss_for_hen_torch = w_con * total_L_CH_value_torch + \
                                     alpha * L_Q_value_torch + \
                                     beta * L_BB_value_torch
        total_loss_for_hen_torch.backward()
        optimizer_hen.step()

        current_total_loss_np = total_loss_for_hen_torch.item()
        if epoch > 10 and abs(current_total_loss_np - loss_last_CMGHash) <= 1e-7 * abs(loss_last_CMGHash) :
            print(f"HEN Loss Converged at epoch {epoch+1}. Current loss: {current_total_loss_np:.7f}, Last loss: {loss_last_CMGHash:.7f}")
        loss_last_CMGHash = current_total_loss_np

        with torch.no_grad():
            U_fixed_for_lambda_torch = hen_model(H_input_torch)
            U_norm_fixed_torch = F.normalize(U_fixed_for_lambda_torch, p=2, dim=1)
            sim_matrix_fixed_torch = torch.matmul(U_norm_fixed_torch, U_norm_fixed_torch.T)

            for v_idx in range(num_actual_views_for_loss):
                M_v_torch = torch.tensor(0.0, device=device)
                for i in range(N):
                    positive_indices = nbrs_inx_list_torch[v_idx][i]
                    mask_not_i_fixed = torch.ones(N, dtype=torch.bool, device=device); mask_not_i_fixed[i] = False
                    sim_i_all_others_fixed = sim_matrix_fixed_torch[i, mask_not_i_fixed]

                    if sim_i_all_others_fixed.numel() == 0:
                        log_sum_exp_negatives_i_fixed = torch.tensor(float('-inf'), device=device)
                    else:
                        log_sum_exp_negatives_i_fixed = torch.logsumexp(sim_i_all_others_fixed / TAU, dim=0)

                    sim_i_positives_fixed = sim_matrix_fixed_torch[i, positive_indices]
                    log_softmax_val_fixed = sim_i_positives_fixed / TAU - log_sum_exp_negatives_i_fixed
                    M_v_torch -= torch.sum(log_softmax_val_fixed)

                M_v_np = (M_v_torch / N).item()

                if M_v_np > 1e-9:
                    if gamma_view_reg == 0 or gamma_view_reg == 1.0:
                        lambda_view_weights_np[v_idx] = 1.0 / num_actual_views_for_loss
                    else:
                        val_inside_power = (-M_v_np / (eta_view_reg_coeff * gamma_view_reg))
                        if val_inside_power > 0:
                            lambda_view_weights_np[v_idx] = (val_inside_power)**(1.0 / (gamma_view_reg - 1.0))
                        else:
                            lambda_view_weights_np[v_idx] = 1.0 / num_actual_views_for_loss
                else:
                    lambda_view_weights_np[v_idx] = 1.0 / num_actual_views_for_loss

            sum_lambda_np = np.sum(lambda_view_weights_np)
            if sum_lambda_np > 1e-9 and np.all(np.isfinite(lambda_view_weights_np)):
                lambda_view_weights_np = lambda_view_weights_np / sum_lambda_np
            else:
                lambda_view_weights_np = np.array([1.0 / num_actual_views_for_loss for _ in range(num_actual_views_for_loss)], dtype=np.float32)

        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            hen_model.eval()
            with torch.no_grad():
                U_eval_torch = hen_model(H_input_torch)
                B_for_eval_np = torch.sign(U_eval_torch).cpu().numpy()

            current_mAP_val = calculate_map(B_for_eval_np, gnd_np)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {current_total_loss_np:.4f} - mAP: {current_mAP_val:.4f}")
            L_CH_item = total_L_CH_value_torch.item()
            L_Q_item = L_Q_value_torch.item()
            L_BB_item = L_BB_value_torch.item()
            print(f"      L_CH: {w_con*L_CH_item:.4f}, L_Q: {alpha*L_Q_item:.4f}, L_BB: {beta*L_BB_item:.4f}")
            print(f"      Lambda_views: {[round(float(lv), 3) for lv in lambda_view_weights_np]}")

            if current_mAP_val > best_mAP:
                best_mAP = current_mAP_val
                best_U_continuous_torch = U_eval_torch.detach().clone()
                best_epoch_mAP = epoch + 1

    final_results_log = f"Data={dataname} w_con={w_con} gamma_vr={gamma_view_reg} K={K_BITS} alpha={alpha} beta={beta} TAU={TAU} best_mAP={best_mAP:.4f} at_epoch={best_epoch_mAP}"
    print(final_results_log)

    # results_dir = "./results_cmghash"
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)

    # log_file_path = os.path.join(results_dir, f"CMGHash_{dataname}_K{K_BITS}_results.txt")
    # with open(log_file_path, 'a') as fl:
    #     fl.write(final_results_log + '\n')

    return best_mAP

if __name__ == "__main__":

    dataname_key = "ACM"
    X_data_views, gnd_labels_np = Acm()
    go_run(X_list_original=X_data_views, gnd_np=gnd_labels_np, dataname=dataname_key)