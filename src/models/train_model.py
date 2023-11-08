import wandb
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error

def train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, model_name):
    # Initialize wandb
    wandb.init(project="adv_mla_at3")

    # Log hyperparameters
    config = wandb.config
    config.epochs = 100
    config.batch_size = 32
    config.optimizer = 'adam'
    config.loss = 'mean_squared_logarithmic_error'
    config.metrics = ['mean_squared_error', 'mean_absolute_percentage_error']
    config.learning_rate = 0.001 
    config.layers = 4

    # Define and compile your model
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=[X_train_scaled.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),  
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2), 
        layers.Dense(1)
    ])

    optimizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    model.compile(optimizer=optimizer, loss=config.loss, metrics=config.metrics)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    # Define ModelCheckpoint callback to save the best model
    checkpoint_filepath = "../../models/Kritika/" + model_name
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True)

    # Callback to log validation MSE after each epoch
    def log_val_mse(epoch, logs):
        val_mse = logs['val_mean_squared_error']
        wandb.log({"val_mse": val_mse}, step=epoch)

    log_val_mse_callback = LambdaCallback(on_epoch_end=log_val_mse)

    # Train the model with callbacks
    model.fit(X_train_scaled, y_train, epochs=config.epochs, batch_size=config.batch_size,
              validation_data=(X_test_scaled, y_test), callbacks=[early_stopping, reduce_lr, model_checkpoint, log_val_mse_callback])

    # Load the best model weights
    best_model = load_model(checkpoint_filepath)

    # Make predictions on train and test sets
    train_predictions = best_model.predict(X_train_scaled)
    test_predictions = best_model.predict(X_test_scaled)

    # Calculate RMSE for train and test sets
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

    # Log RMSE scores
    wandb.log({"train_rmse": train_rmse, "test_rmse": test_rmse})

    # Log the model in W&B and link it to the registry
    artifact = wandb.Artifact(name='trained_model', type='model')
    artifact.add_file(checkpoint_filepath)
    wandb.run.log_artifact(artifact)

    wandb.finish()
