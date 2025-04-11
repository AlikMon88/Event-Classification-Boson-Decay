import tensorflow as tf
from tensorflow.keras import layers, models

def dnn_classifier(input_shape, normalizer, num_classes=3):
    """
    Build a neural network model for event classification.
    Uses only event-level features from your dataset.
    """
    # Input layer for event-level features
    input_layer = layers.Input(shape=input_shape, name='event_input')
    
    # Dense layers for processing event-level features
    x = normalizer(input_layer)
    x = layers.Dense(64, activation='relu')(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='linear')(x)
    
    # Output layer with softmax activation for multi-class classification
    output_layer = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', ## not one-hot encoded
        metrics=['accuracy']
    )
    
    return model

#=================================================================================================#

def gru_classifier_v1(input_shape, normalizer, hidden_dim=128, num_layers=2, num_classes=3,  bidirectional=True, dropout=0.2):
    """
    input_shape: tuple, e.g., (seq_len, input_dim)
    hidden_dim: number of units for GRU layers
    num_layers: how many GRU layers
    num_classes: output classes (e.g., 4)
    bidirectional: whether to use bidirectional GRU layers
    dropout: dropout rate (applied between GRU layers if num_layers > 1)
    """
    inputs = tf.keras.Input(shape=input_shape)  # shape = (seq_len, input_dim)
    x = normalizer(inputs)
    x = tf.keras.layers.Masking(mask_value=-999.0)(x)

    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False  # return full sequence except last layer
        gru_layer = layers.GRU(hidden_dim, return_sequences=return_seq)
        if bidirectional:
            gru_layer = layers.Bidirectional(gru_layer)
        x = gru_layer(x)
        if i < num_layers - 1:
            x = layers.Dropout(dropout)(x)
    
    # x now is the final output vector (batch_size, hidden_dim * (2 if bidirectional else 1))
    outputs = layers.Dense(num_classes)(x)  # logits for each class
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


def gru_classifier_v2(input_shape, normalizer, num_layers=2, hidden_dim=32, num_classes=3, bidirectional=True, dropout=0.3, recurrent_dropout=0.2):
    """
    Enhanced GRU classifier with attention mechanism and residual connections.

    Args:
        input_shape (tuple): (seq_len, input_dim) e.g., shape of each event.
        hidden_dim (int): Number of units for GRU layers.
        num_layers (int): Number of stacked GRU layers.
        num_classes (int): Number of output classes.
        bidirectional (bool): Whether to use bidirectional GRU layers.
        dropout (float): Dropout rate between layers.
        recurrent_dropout (float): Recurrent dropout rate inside GRU cells.
        
    Returns:
        tf.keras.Model: Compiled GRU-based event classifier.
    """
    # Input and initial normalization and masking.
    inputs = tf.keras.Input(shape=input_shape)  # shape = (seq_len, input_dim)
    # Assume 'normalizer' is a prebuilt and adapted Normalization layer.
    x = normalizer(inputs)  
    x = layers.Masking(mask_value=-999.0)(x)
    
    # Variable to hold the residual connection.
    residual = None
    
    # Stack GRU layers.
    for i in range(num_layers):
        # For all layers, return sequences to let subsequent layers or attention process time steps.
        return_seq = True  
        
        gru_layer = layers.GRU(
            hidden_dim, 
            return_sequences=return_seq, 
            dropout=dropout, 
            recurrent_dropout=recurrent_dropout
        )
        
        if bidirectional:
            gru_layer = layers.Bidirectional(gru_layer)
            
        x_new = gru_layer(x)
        
        # If there is a previous layer's output and dimensions match, add a residual connection.
        if residual is not None and x_new.shape[-1] == residual.shape[-1]:
            x_new = layers.Add()([x_new, residual])
        
        # Update residual and then apply BatchNormalization.
        residual = x_new
        x = layers.BatchNormalization()(x_new)
        
        # Optionally add dropout between GRU blocks except for the last layer.
        if i < num_layers - 1:
            x = layers.Dropout(dropout)(x)
    
    # Attention mechanism to aggregate over the time dimension.
    # Compute attention weights.
    # Note: We use a dense layer with a tanh activation to score each time step.
    attention_weights = layers.Dense(1, activation='tanh')(x)   # (batch, seq_len, 1)
    attention_weights = layers.Flatten()(attention_weights)       # (batch, seq_len)
    attention_weights = layers.Activation('softmax')(attention_weights)  # (batch, seq_len)
    # Expand dims to (batch, seq_len, 1) to multiply with x.
    attention_weights = layers.RepeatVector(x.shape[-1])(attention_weights)  
    attention_weights = layers.Permute([2, 1])(attention_weights)  
    # Weighted sum of GRU outputs.
    weighted_sum = layers.Multiply()([x, attention_weights])       
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(weighted_sum)  # (batch, features)
    
    # Alternatively, you could use GlobalAveragePooling1D:
    # x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(hidden_dim, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes)(x)  # logits for each class
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

#=================================================================================================#

def deepset_classifier_v1(input_shape, normalizer, hidden_dim_phi=128, hidden_dim_rho=64, num_classes=3):
    """
    input_shape: tuple, e.g., (set_size, input_dim), where set_size is the number of elements per event.
    hidden_dim_phi: number of units in the φ network (applied to each element).
    hidden_dim_rho: number of units in the ρ network (applied after aggregation).
    num_classes: number of output classes.
    """
    
    inputs = tf.keras.Input(shape=input_shape)  # shape = (set_size, input_dim)
    x = normalizer(inputs)
    x = tf.keras.layers.Masking(mask_value=-999.0)(x)
    
    # The φ network is applied element-wise: use TimeDistributed for shared MLP.
    phi = tf.keras.Sequential([
        layers.Dense(hidden_dim_phi, activation='relu'),
        layers.Dense(hidden_dim_phi, activation='relu')
    ])
    
    # Apply φ to each element independently.
    x = layers.TimeDistributed(phi)(inputs)  # shape = (batch_size, set_size, hidden_dim_phi)
    
    # Aggregate (sum over the set axis). Alternative aggregations: mean, max.
    x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)  # shape: (batch_size, hidden_dim_phi)
    
    # The ρ network
    x = layers.Dense(hidden_dim_rho, activation='relu')(x)
    outputs = layers.Dense(num_classes)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

def deepset_classifier_v2(input_shape, normalizer, hidden_dim_phi=64, hidden_dim_rho=32, num_classes=3, dropout_rate=0.2):
    """
    Enhanced DeepSets classifier for event classification.
    
    Args:
        input_shape (tuple): (set_size, input_dim), where set_size is the number of elements per event.
        hidden_dim_phi (int): number of units in the φ network (applied to each element).
        hidden_dim_rho (int): number of units in the ρ network (applied after aggregation).
        num_classes (int): number of output classes.
        dropout_rate (float): dropout rate used throughout the network.
    
    Returns:
        tf.keras.Model: A compiled DeepSet classifier model.
    """
    # Input, normalization, and masking.
    inputs = tf.keras.Input(shape=input_shape)  # shape = (set_size, input_dim)
    x = normalizer(inputs)  # Pre-adapted normalization layer
    x = layers.Masking(mask_value=-999.0)(x)
    
    # Build the φ (phi) network applied element-wise.
    # We use a sequential model wrapped in TimeDistributed.
    phi_model = tf.keras.Sequential([
        layers.Dense(hidden_dim_phi, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_dim_phi * 2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(hidden_dim_phi * 2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden_dim_phi, activation='relu'),
        layers.BatchNormalization()
    ])
    
    x = layers.TimeDistributed(phi_model)(x)  # Output shape: (batch_size, set_size, hidden_dim_phi)
    
    # Aggregate the per-element representations.
    # Here we combine a sum aggregator with a global max-pooling aggregator.
    sum_agg = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)  # Shape: (batch_size, hidden_dim_phi)
    max_agg = layers.GlobalMaxPooling1D()(x)                         # Shape: (batch_size, hidden_dim_phi)
    agg = layers.Concatenate()([sum_agg, max_agg])                     # Shape: (batch_size, 2 * hidden_dim_phi)
    
    # Build the ρ (rho) network as the classifier head.
    x_rho = layers.Dense(hidden_dim_rho, activation='relu')(agg)
    x_rho = layers.BatchNormalization()(x_rho)
    x_rho = layers.Dropout(dropout_rate)(x_rho)
    # Optionally add a second dense layer for further processing.
    x_rho = layers.Dense(hidden_dim_rho // 2, activation='relu')(x_rho)
    x_rho = layers.BatchNormalization()(x_rho)
    x_rho = layers.Dropout(dropout_rate)(x_rho)
    
    # Final output logits (no activation, so you can use from_logits=True in loss).
    outputs = layers.Dense(num_classes)(x_rho)
    
    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model
