import pandas as pd
import matplotlib.pyplot as plt
from perceptron import Perceptron
from sklearn.model_selection import train_test_split

file_path = 'data.csv'
if __name__ == "__main__":
    column_names = ['x1', 'x2', 'x3', 'x4', 'class']
    df = pd.read_csv(file_path, header=None, names=column_names)
    
    class_mapping = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1
    }
    df['class'] = df['class'].map(class_mapping)
    
    X = df[['x1', 'x2', 'x3', 'x4']].values.tolist()
    y = df['class'].values.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    perceptron = Perceptron()
    weights, bias = perceptron.initialize_weights_bias(4)
    
    # Training
    epochs = 5
    learning_rate = 0.1
    
    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        total_error = 0
        correct_train = 0
        
        for i in range(len(X_train)):
            dot_product = perceptron.dot_product(X_train[i])
            predicted = perceptron.sigmoid(dot_product)
            predicted_class = 1 if predicted >= 0.5 else 0
            
            # Calculate error
            error = perceptron.error_function(predicted, y_train[i])
            total_error += abs(error)
            
            # Count correct predictions for training
            if predicted_class == y_train[i]:
                correct_train += 1
            
            # Update weights and bias
            perceptron.update_weights_bias(X_train[i], y_train[i], predicted, learning_rate)
        
        # Calculate training metrics
        train_loss = total_error / len(X_train)
        train_accuracy = (correct_train / len(X_train)) * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        val_error = 0
        correct_val = 0
        
        for i in range(len(X_test)):
            # Forward pass
            dot_product = perceptron.dot_product(X_test[i])
            predicted = perceptron.sigmoid(dot_product)
            predicted_class = 1 if predicted >= 0.5 else 0
            
            # Calculate error
            error = perceptron.error_function(predicted, y_test[i])
            val_error += abs(error)
            
            # Count correct predictions for validation
            if predicted_class == y_test[i]:
                correct_val += 1
        
        # Calculate validation metrics
        val_loss = val_error / len(X_test)
        val_accuracy = (correct_val / len(X_test)) * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch + 1}:")
        print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%\n")
    
    # Plot training and validation metrics
    epochs_range = range(1, epochs + 1)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.xticks(epochs_range)
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


