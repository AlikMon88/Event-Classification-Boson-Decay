import uproot
import pandas as pd
import random
import numpy as np
import os
from scipy.stats import zscore

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Example usage:
file_paths = {
    "Zbb": "../data/sec_c/Zbb.root",
    "Zcc": "../data/sec_c/Zcc.root",
    "Zss": "../data/sec_c/Zss.root"
}


base_dir = os.path.dirname(os.path.abspath(__file__))
absolute_paths = {key: os.path.join(base_dir, value) for key, value in file_paths.items()}


def load_data(event_ratio=0.3):
    """
    Loads ROOT files and extracts relevant features into a Pandas DataFrame.
    
    Parameters:
        file_paths (dict): Dictionary where keys are class labels and values are ROOT file paths.
        labels (dict): Dictionary mapping class labels to numeric values.
    
    Returns:
        pd.DataFrame: A DataFrame containing the extracted features and labels.
    """
    data = []
    labels = {"Zbb": 0, "Zcc": 1, "Zss": 2}
    
    for label, file_path in absolute_paths.items():
        with uproot.open(file_path) as f:
            tree = f["events"]  # Adjust this if the tree name is different
            
            # Load features
            thrust_x = tree["Thrust_x"].array(library="np")
            thrust_y = tree["Thrust_y"].array(library="np")
            thrust_z = tree["Thrust_z"].array(library="np")
            n_particles = tree["nParticle"].array(library="np")
            n_vertices = tree["nVertex"].array(library="np")
            vertex_ntracks = tree["Vertex_ntracks"].array(library="np")
            vertex_chi2 = tree["Vertex_chi2"].array(library="np")
            particle_p = tree["Particle_p"].array(library="np")
            particle_pt = tree["Particle_pt"].array(library="np")
            
            # Convert variable-length arrays to fixed statistics
            vertex_chi2_mean = np.array([np.mean(np.array(v, dtype=float)) if len(v) > 0 else 0 for v in vertex_chi2])
            vertex_chi2_std = np.array([np.std(np.array(v, dtype=float)) if len(v) > 0 else 0 for v in vertex_chi2])

            # Store data in a structured format
            for i in range(int(len(thrust_x) * event_ratio)):
                data.append([
                    thrust_x[i], thrust_y[i], thrust_z[i],
                    n_particles[i], n_vertices[i], vertex_chi2[i],
                    vertex_chi2_mean[i], vertex_ntracks[i], vertex_chi2_std[i],#, particle_phi[i], particle_eta[i],
                    particle_p[i], particle_pt[i],
                    labels[label]  # Numeric label
                ])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        "Thrust_x", "Thrust_y", "Thrust_z",
        "nParticle", "nVertex", "Vertex_chi2", "Vertex_ntracks",
        "Vertex_chi2_mean", "Vertex_chi2_std",
        "Particle_p", "Particle_pt",
        "label"
    ])

    # Convert object-type arrays to numerical lists
    def safe_eval_array(arr):
        return np.array([_a.mean() for _a in arr])

    # Apply conversion to array-type columns
    df['Particle_p_mean'] = safe_eval_array(np.array(df.Particle_p))
    df['Particle_pt_mean'] = safe_eval_array(np.array(df.Particle_pt))
    df['Vertex_chi2_mean'] = safe_eval_array(np.array(df.Vertex_chi2_mean))

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df

def data_prepare(df, target_cols, pad_val=-999):
    """
    Prepares a combined sequence array for RNN input, repeating scalar features across sequence steps
    and padding variable-length sequence features.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_cols (list): List of sequence feature columns (e.g., ['Particle_p', 'Particle_pt']).
        pad_val (float): Padding value for shorter sequences.
    
    Returns:
        np.ndarray: Processed array of shape (n_events, max_seq_len, n_features).
    """
    # Determine the maximum sequence length across the target columns.
    max_seq_len = max([max(len(seq) for seq in df[col]) for col in target_cols])
    print('Max-Sequence-Len:', max_seq_len)
    
    n_events = len(df)
    
    # Identify scalar (event-level) columns (all columns not in the sequence targets)
    not_target_cols = [col for col in df.columns if col not in target_cols]
    n_scalar_features = len(not_target_cols)
    n_seq_features = len(target_cols)
    n_total_features = n_scalar_features + n_seq_features
    
    print('Scalar-Features (Event-level):', n_scalar_features,
          ', Sequence-Features (Particle-level):', n_seq_features)
    
    # Prepare output array with a pad value
    arr = np.full((n_events, max_seq_len, n_total_features), pad_val, dtype=np.float32)
    
    for i in range(n_events):
        # Repeat scalar features across the sequence length
        scalar_vals = df.iloc[i][not_target_cols].values.astype(np.float32)
        arr[i, :, :n_scalar_features] = np.repeat(scalar_vals[None, :], max_seq_len, axis=0)
        
        # Process each sequence feature
        for j, col in enumerate(target_cols):
            # Convert the stored list/array into a numpy array
            seq_vals = np.array(df.iloc[i][col], dtype=np.float32)
            seq_len = len(seq_vals)
            arr[i, :seq_len, n_scalar_features + j] = seq_vals
    
    return arr


def z_outlier_filter(x, y):
    """
    Performs outlier filtering on scalar features within each event.
    Only the first 10 scalar features are used for computing z-scores,
    and events with any feature beyond 4 standard deviations are removed.
    
    Args:
        x (np.ndarray): Input data with shape (n_events, max_seq_len, n_features).
        y (np.ndarray): Labels corresponding to each event.
    
    Returns:
        tuple: A tuple (X_filtered, y_filtered) with outliers removed.
    """
    # Reshape first 10 features from all sequence steps into 2D array (n_events, features)
    tmp_x = x[:, 0, :10].reshape(len(x), -1)
    z_scores = np.abs(zscore(tmp_x, axis=0))
    # Create a mask of events where all z-scores are within 4 standard deviations.
    mask = z_scores < 4
    row_mask = np.all(mask, axis=1)
    
    X_filtered = x[row_mask]
    y_filtered = y[row_mask]
    
    return X_filtered, y_filtered


def load_data_generator(event_ratio=0.3, batch_size=100):
    """
    Data generator that iterates over ROOT files, processes events in batches,
    creates sequence features, and applies outlier filtering.
    
    Parameters:
        event_ratio (float): Fraction of events to sample from each ROOT file/chunk.
        batch_size (int): Number of events per batch.
    
    Yields:
        Tuple (X_filtered, y_filtered):
            - X_filtered: Preprocessed input data array (after sequence creation and outlier filtering).
            - y_filtered: Labels array after outlier filtering.
    """
    labels = {"Zbb": 0, "Zcc": 1, "Zss": 2}
    # Define which branches (features) to iterate over in the ROOT tree.
    branches = [
        "Thrust_x", "Thrust_y", "Thrust_z",
        "nParticle", "nVertex",
        "Vertex_chi2", "Vertex_ntracks",
        "Particle_p", "Particle_pt"
    ]
    # Target sequence columns (here, we treat Particle-level variables as sequences)
    target_cols = ["Particle_p", "Particle_pt", 'Vertex_chi2']
    
    batch_data = []  # to accumulate events for the current batch
    
    # Iterate over each ROOT file (class)
    for label, file_path in absolute_paths.items():
        with uproot.open(file_path) as f:
            tree = f["events"]  # adjust if tree name is different
            
            # Use the tree iterator to read in chunks
            for arrays in tree.iterate(branches, library="np", step_size = batch_size):
                # Determine how many events to keep in this chunk
                n_events_chunk = int(len(arrays["Thrust_x"]) * event_ratio)
                for i in range(n_events_chunk):
                    # Calculate per-event statistics for the variable-length array Vertex_chi2
                    event_vertex_chi2 = arrays["Vertex_chi2"][i]
                    if len(event_vertex_chi2) > 0:
                        chi2_mean = np.mean(np.array(event_vertex_chi2, dtype=float))
                        chi2_std = np.std(np.array(event_vertex_chi2, dtype=float))
                    else:
                        chi2_mean = 0
                        chi2_std = 0
                    
                    # Append the event as a list; note that some variables are stored as variable-length arrays.
                    event = [
                        arrays["Thrust_x"][i],
                        arrays["Thrust_y"][i],
                        arrays["Thrust_z"][i],
                        arrays["nParticle"][i],
                        arrays["nVertex"][i],
                        arrays["Vertex_chi2"][i],  # keep variable-length array for later sequence pad if needed
                        arrays["Vertex_ntracks"][i],
                        chi2_mean,
                        chi2_std,
                        arrays["Particle_p"][i],  # sequence feature
                        arrays["Particle_pt"][i],  # sequence feature
                        labels[label]
                    ]
                    batch_data.append(event)
                    
                    # When we have enough events, process and yield a batch.
                    if len(batch_data) >= batch_size:
                        # Create a DataFrame from the batch data.
                        df_batch = pd.DataFrame(batch_data, columns=[
                            "Thrust_x", "Thrust_y", "Thrust_z",
                            "nParticle", "nVertex", "Vertex_chi2", "Vertex_ntracks",
                            "Vertex_chi2_mean", "Vertex_chi2_std",
                            "Particle_p", "Particle_pt",
                            "label"
                        ])
                        
                        # Create combined sequence array for RNN input.
                        X = data_prepare(df_batch, target_cols, pad_val=-999)
                        y = df_batch["label"].values
                        
                        # Apply outlier filtering.
                        X_filtered, y_filtered = z_outlier_filter(X, y)
                        
                        yield (X_filtered, y_filtered)
                        
                        # Reset batch container.
                        batch_data = []
                        
    # Process and yield any remaining events after finishing the iteration.
    if batch_data:
        df_batch = pd.DataFrame(batch_data, columns=[
            "Thrust_x", "Thrust_y", "Thrust_z",
            "nParticle", "nVertex", "Vertex_chi2", "Vertex_ntracks",
            "Vertex_chi2_mean", "Vertex_chi2_std",
            "Particle_p", "Particle_pt",
            "label"
        ])
        X = data_prepare(df_batch, target_cols, pad_val=-999)
        y = df_batch["label"].values
        X_filtered, y_filtered = z_outlier_filter(X, y)
        yield (X_filtered, y_filtered)


if __name__ == '__main__':
    # For testing the generator:
    print('Running data generator...')
    generator = load_data_generator(event_ratio=0.3, batch_size=50)
    for i, (X_batch, y_batch) in enumerate(generator):
        print(f'Batch {i+1}: X shape = {X_batch.shape}, y shape = {y_batch.shape}')
        # For demonstration, break after a few batches.
        if i >= 2:
            break