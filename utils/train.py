import numpy as np
import tensorflow as tf
import os
import argparse
import pickle as pkl
from tensorflow.keras.layers import Normalization
from .model import *  # Import your model architecture

epochs = 1


''' 
... MAIN - HYPERPARAMETER CHANGE IN HERE ... 
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_dnn(train, val, epochs=1, num_classes=3):
    """
    Function to train the DeepSet model.
    Args:
    - train: tuple (X_train, y_train)
    - val: tuple (X_val, y_val)
    - hidden_dim_phi: Number of hidden units in the φ network
    - hidden_dim_rho: Number of hidden units in the ρ network
    - num_classes: Number of output classes
    """
    
    # Define absolute paths for saving the model and metrics
    save_path = os.path.join(BASE_DIR, '..', 'saves', 'models', 'dnn_model.h5')
    metrics_path = os.path.join(BASE_DIR, '..', 'saves', 'metrics', 'dnn_metrics.pkl')

    # Set up callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=3
        )
    ]

    X_train, y_train = train
    X_val, y_val = val

    # Feature-wise Normalization (like StandardScaler)
    normalizer = Normalization(axis=-1)  
    normalizer.adapt(X_train)  # Adapt to the training data

    # Build the DeepSet model
    dnn_model = dnn_classifier(
        input_shape=X_train.shape[1:], 
        num_classes=num_classes, 
        normalizer=normalizer, 
    )

    # Train the model
    history_dnn = dnn_model.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save the trained model
    dnn_model.save(save_path)
    
    # Save training metrics
    with open(metrics_path, 'wb') as f:
        pkl.dump(history_dnn.history, f)

    return history_dnn, dnn_model


def train_deepset(train, val, epochs=1, hidden_dim_phi=64, hidden_dim_rho=32, num_classes=3):
    """
    Function to train the DeepSet model.
    Args:
    - train: tuple (X_train, y_train)
    - val: tuple (X_val, y_val)
    - hidden_dim_phi: Number of hidden units in the φ network
    - hidden_dim_rho: Number of hidden units in the ρ network
    - num_classes: Number of output classes
    """
    
    # Define absolute paths for saving the model and metrics
    save_path = os.path.join(BASE_DIR, '..', 'saves', 'models', 'deepset_model.h5')
    metrics_path = os.path.join(BASE_DIR, '..', 'saves', 'metrics', 'deepset_metrics.pkl')
    
    # Set up callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=3
        )
    ]

    X_train, y_train = train
    X_val, y_val = val

    # Feature-wise Normalization (like StandardScaler)
    normalizer = Normalization(axis=-1)  
    normalizer.adapt(X_train)  # Adapt to the training data

    # model - v2
    deepset_model = deepset_classifier_v2(
        input_shape=X_train.shape[1:], 
        hidden_dim_phi=hidden_dim_phi, 
        hidden_dim_rho=hidden_dim_rho, 
        num_classes=num_classes, 
        normalizer=normalizer, 
        dropout_rate=0.3
    )

    # # model - v1
    # deepset_model = deepset_classifier_v1(
    #     input_shape=X_train.shape[1:], 
    #     hidden_dim_phi=hidden_dim_phi, 
    #     hidden_dim_rho=hidden_dim_rho, 
    #     num_classes=num_classes, 
    #     normalizer=normalizer, 
    # )
    
    history_deepset = deepset_model.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save the trained model
    deepset_model.save(save_path)
    
    # Save training metrics
    with open(metrics_path, 'wb') as f:
        pkl.dump(history_deepset.history, f)

    return history_deepset, deepset_model


def train_gru(train, val, epochs=1, hidden_dim=32, num_layers=2, num_classes=3, bidirectional=True, dropout_rate=0.3):
    """
    Function to train the GRU model.
    Args:
    - train: tuple (X_train, y_train)
    - val: tuple (X_val, y_val)
    - hidden_dim: Number of hidden units in the GRU layers
    - num_layers: Number of GRU layers
    - num_classes: Number of output classes
    - bidirectional: Whether to use bidirectional GRU layers
    - dropout_rate: Dropout rate between layers
    """
    
    # Define absolute paths for saving the model and metrics
    save_path = os.path.abspath(os.path.join(BASE_DIR,'..', 'saves', 'models', 'gru_model.h5'))
    metrics_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'saves', 'metrics', 'gru_metrics.pkl'))

    # Set up callbacks for early stopping and learning rate reduction
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, 
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, 
            patience=3
        )
    ]

    X_train, y_train = train
    X_val, y_val = val

    # Feature-wise Normalization (like StandardScaler)
    normalizer = Normalization(axis=-1)  
    normalizer.adapt(X_train)  # Adapt to the training data

    # # model - v2
    gru_model = gru_classifier_v2(
        input_shape=X_train.shape[1:], 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_classes=num_classes, 
        bidirectional=bidirectional, 
        normalizer=normalizer,
        dropout=dropout_rate
    )

    # model - v1
    # gru_model = gru_classifier_v1(
    # input_shape=X_train.shape[1:], 
    # normalizer=normalizer, 
    # hidden_dim=hidden_dim, 
    # num_layers=num_layers, 
    # num_classes=num_classes, 
    # bidirectional=bidirectional,
    # dropout=dropout_rate
    # )

    # Compile the model
    gru_model.compile(optimizer='adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy'])

    # Train the model
    history_gru = gru_model.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        validation_data=(X_val, y_val),
        batch_size=32,
        callbacks=callbacks
    )
    
    # Save the trained model
    gru_model.save(save_path)
    
    # Save training metrics
    with open(metrics_path, 'wb') as f:
        pkl.dump(history_gru.history, f)

    return history_gru, gru_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model (GRU or DeepSet or dnn).")
    parser.add_argument(
        '--model', 
        choices=['gru', 'deepset', 'dnn'], 
        required=True, 
        help="Model type to train: 'gru' or 'deepset'."
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    print(f"Running __train.py__ with model type: {args.model} ...")

    if args.model == 'gru' or args.model == 'deepset':

        # Load the preprocessed data (replace with your actual file paths)
        loaded_data = np.load(os.path.abspath('data/preprocessed/data_seq.npz'))
        
        X_train = loaded_data['x_train']
        X_val = loaded_data['x_val']
        X_test = loaded_data['x_test']
        y_train = loaded_data['y_train']
        y_val = loaded_data['y_val']
        y_test = loaded_data['y_test']

        print(X_train.shape, X_val.shape, X_test.shape)
        print(y_train.shape, y_val.shape, y_test.shape)
    
    elif args.model == 'dnn':

        loaded_data = np.load(os.path.abspath('data/preprocessed/data_seq.npz'))
    
        X_train = loaded_data['x_train']
        X_val = loaded_data['x_val']
        X_test = loaded_data['x_test']
        y_train = loaded_data['y_train']
        y_val = loaded_data['y_val']
        y_test = loaded_data['y_test']

        print(X_train.shape, X_val.shape, X_test.shape)
        print(y_train.shape, y_val.shape, y_test.shape)

    if args.model == 'gru':
        # Train GRU model
        gru_model = train_gru(
            (X_train, y_train), 
            (X_val, y_val),
            hidden_dim=128,
            num_layers=2,
            num_classes=4
        )
    elif args.model == 'deepset':
        # Train DeepSet model
        deepset_model = train_deepset(
            (X_train, y_train), 
            (X_val, y_val),
            hidden_dim_phi=64,
            hidden_dim_rho=32,
            num_classes=4
        )

    elif args.model == 'dnn':

        dnn_model = train_dnn(
            (X_train, y_train), 
            (X_val, y_val),
            num_classes=4
        )