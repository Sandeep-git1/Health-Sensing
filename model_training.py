import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SleepModelTrainer:
    """
    Sleep breathing irregularity classification using 1D CNN and Conv-LSTM
    with Leave-One-Participant-Out Cross-Validation
    
    Why Leave-One-Participant-Out CV?
    - Prevents data leakage: physiological signals from the same person are highly correlated
    - Better generalization: tests model's ability to work on unseen individuals
    - More realistic evaluation: mimics real-world deployment where model sees new patients
    - Avoids overfitting to participant-specific patterns
    """
    
    def __init__(self, dataset_path, results_dir="Results"):
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Target classes as specified in assignment
        self.target_classes = ['Hypopnea', 'Obstructive Apnea', 'Normal']
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Model parameters
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.001
        
        # Signal parameters (will be set after loading data)
        self.signal_names = []
        self.max_signal_length = 0
        self.n_classes = 0
        
    def load_dataset(self):
        """Load and prepare dataset for modeling"""
        print("Loading dataset...")
        
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        
        # Filter dataset to only include target classes
        filtered_dataset = []
        for window in dataset:
            if window['label'] in self.target_classes:
                filtered_dataset.append(window)
        
        dataset = filtered_dataset
        print(f"Filtered dataset to {len(dataset)} windows with target classes: {self.target_classes}")
        
        # Extract signal names and determine max length
        all_signal_names = set()
        signal_lengths = []
        
        for window in dataset:
            all_signal_names.update(window['signals'].keys())
            for signal_data in window['signals'].values():
                signal_lengths.append(len(signal_data))
        
        self.signal_names = sorted(list(all_signal_names))
        self.max_signal_length = max(signal_lengths) if signal_lengths else 960  # 30s * 32Hz
        
        print(f"Signal names: {self.signal_names}")
        print(f"Max signal length: {self.max_signal_length}")
        
        # Prepare data
        participants = []
        labels = []
        signals_data = []
        
        for window in dataset:
            participants.append(window['participant'])
            labels.append(window['label'])
            
            # Process signals individually to maintain their characteristics
            window_signals = []
            for signal_name in self.signal_names:
                if signal_name in window['signals']:
                    signal = window['signals'][signal_name]
                    # Pad or truncate to max length
                    if len(signal) > self.max_signal_length:
                        signal = signal[:self.max_signal_length]
                    else:
                        signal = np.pad(signal, (0, self.max_signal_length - len(signal)), 'constant')
                    window_signals.append(signal)
                else:
                    # If signal is missing, pad with zeros
                    window_signals.append([0] * self.max_signal_length)
            
            signals_data.append(window_signals)
        
        # Convert to numpy arrays
        X = np.array(signals_data)
        y = np.array(labels)
        participants = np.array(participants)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.n_classes = len(self.label_encoder.classes_)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Class distribution: {Counter(y)}")
        print(f"Participants: {sorted(set(participants))}")
        
        # Ensure we have the expected classes
        actual_classes = set(self.label_encoder.classes_)
        expected_classes = set(self.target_classes)
        if not expected_classes.issubset(actual_classes):
            missing = expected_classes - actual_classes
            print(f"Warning: Missing expected classes: {missing}")
        
        return X, y_encoded, participants, y
    
    def prepare_signals_for_model(self, X):
        """Reshape signals for neural network input"""
        # X shape: (samples, signals, timesteps)
        # Transpose to get (samples, timesteps, signals) for CNN/LSTM input
        X_reshaped = X.transpose(0, 2, 1)
        return X_reshaped
    
    def build_1d_cnn(self, input_shape, n_classes):
        """Build 1D CNN model"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        
        return model
    
    def build_conv_lstm(self, input_shape, n_classes):
        """Build 1D Conv-LSTM model"""
        model = Sequential([
            # Convolutional layers for feature extraction
            Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            # LSTM layers for temporal modeling
            # The Conv1D output naturally has the right shape for LSTM (samples, timesteps, features)
            LSTM(50, return_sequences=True),
            LSTM(50),
            
            # Dense layers for classification
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
        
        return model
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba, class_names):
        """Calculate all required metrics"""
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate sensitivity and specificity for each class
        sensitivity = recall_per_class  # Sensitivity = Recall = TPR
        specificity = []
        
        for i in range(len(class_names)):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        
        specificity = np.array(specificity)
        
        # Store per-class metrics
        metrics['per_class'] = {}
        for i, class_name in enumerate(class_names):
            metrics['per_class'][class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'sensitivity': sensitivity[i],
                'specificity': specificity[i]
            }
        
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names, title, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_fold_results(self, fold, participant, model_name, metrics, history=None):
        """Save individual fold results"""
        fold_dir = os.path.join(self.results_dir, f"fold_{fold}_{participant}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(fold_dir, f"{model_name}_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        
        # Save confusion matrix plot
        cm_path = os.path.join(fold_dir, f"{model_name}_confusion_matrix.png")
        self.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            self.label_encoder.classes_,
            f"{model_name} - Fold {fold} ({participant})",
            cm_path
        )
        
        # Save training history if available
        if history:
            history_path = os.path.join(fold_dir, f"{model_name}_history.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(history.history, f)
    
    def train_and_evaluate(self):
        """Main training and evaluation with Leave-One-Participant-Out CV"""
        print("="*60)
        print("SLEEP BREATHING IRREGULARITY CLASSIFICATION")
        print("="*60)
        print("\nWhy Leave-One-Participant-Out Cross-Validation?")
        print("- Prevents data leakage between participants")
        print("- Tests generalization to unseen individuals")
        print("- Avoids overfitting to participant-specific patterns")
        print("- More realistic evaluation for clinical deployment")
        print("="*60)
        
        # Load dataset
        X, y_encoded, participants, y_original = self.load_dataset()
        
        # Prepare signals
        X_prepared = self.prepare_signals_for_model(X)
        print(f"Prepared data shape: {X_prepared.shape}")
        
        # Get unique participants
        unique_participants = sorted(set(participants))
        n_folds = len(unique_participants)
        
        # Results storage
        results = {
            'cnn_results': [],
            'conv_lstm_results': [],
            'fold_details': []
        }
        
        print(f"\nStarting {n_folds}-Fold Leave-One-Participant-Out Cross-Validation")
        print(f"Participants: {unique_participants}")
        
        # Leave-One-Participant-Out Cross-Validation
        for fold, test_participant in enumerate(unique_participants):
            print(f"\n{'='*60}")
            print(f"FOLD {fold + 1}/{n_folds} - Test Participant: {test_participant}")
            print(f"{'='*60}")
            
            # Split data
            train_mask = participants != test_participant
            test_mask = participants == test_participant
            
            X_train, X_test = X_prepared[train_mask], X_prepared[test_mask]
            y_train, y_test = y_encoded[train_mask], y_encoded[test_mask]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Convert to categorical
            y_train_cat = to_categorical(y_train, num_classes=self.n_classes)
            y_test_cat = to_categorical(y_test, num_classes=self.n_classes)
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            print(f"Train set: {X_train.shape[0]} samples")
            print(f"Test set: {X_test.shape[0]} samples")
            print(f"Class weights: {class_weight_dict}")
            
            # Train and evaluate both models
            for model_name, model_builder in [('CNN', self.build_1d_cnn), 
                                            ('Conv-LSTM', self.build_conv_lstm)]:
                print(f"\n{'-'*40}")
                print(f"Training {model_name}")
                print(f"{'-'*40}")
                
                # Build model
                model = model_builder(X_train_scaled.shape[1:], self.n_classes)
                model.compile(
                    optimizer=Adam(learning_rate=self.learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"Model architecture:")
                model.summary()
                
                # Callbacks
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
                ]
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train_cat,
                    validation_data=(X_test_scaled, y_test_cat),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    class_weight=class_weight_dict,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Predictions
                y_pred_proba = model.predict(X_test_scaled)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    y_test, y_pred, y_pred_proba, 
                    self.label_encoder.classes_
                )
                
                # Save fold results
                self.save_fold_results(fold + 1, test_participant, model_name, metrics, history)
                
                # Store results
                fold_result = {
                    'fold': fold + 1,
                    'test_participant': test_participant,
                    'model': model_name,
                    'metrics': metrics
                }
                
                if model_name == 'CNN':
                    results['cnn_results'].append(fold_result)
                else:
                    results['conv_lstm_results'].append(fold_result)
                
                # Print fold results
                print(f"\n{model_name} Results - Fold {fold + 1}:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision (macro): {metrics['precision_macro']:.4f}")
                print(f"Recall (macro): {metrics['recall_macro']:.4f}")
                
                print(f"\nPer-class metrics:")
                for class_name, class_metrics in metrics['per_class'].items():
                    print(f"  {class_name}:")
                    print(f"    Precision: {class_metrics['precision']:.4f}")
                    print(f"    Recall: {class_metrics['recall']:.4f}")
                    print(f"    Sensitivity: {class_metrics['sensitivity']:.4f}")
                    print(f"    Specificity: {class_metrics['specificity']:.4f}")
        
        # Calculate aggregated results
        self.calculate_and_save_aggregated_results(results)
        
        return results
    
    def calculate_and_save_aggregated_results(self, results):
        """Calculate and save aggregated results across all folds"""
        print(f"\n{'='*60}")
        print("AGGREGATED RESULTS")
        print(f"{'='*60}")
        
        aggregated_results = {}
        
        for model_type in ['cnn_results', 'conv_lstm_results']:
            model_name = 'CNN' if model_type == 'cnn_results' else 'Conv-LSTM'
            model_results = results[model_type]
            
            # Extract metrics across folds
            accuracies = [r['metrics']['accuracy'] for r in model_results]
            precisions = [r['metrics']['precision_macro'] for r in model_results]
            recalls = [r['metrics']['recall_macro'] for r in model_results]
            
            # Per-class metrics
            per_class_metrics = {}
            for class_name in self.label_encoder.classes_:
                per_class_metrics[class_name] = {
                    'precision': [r['metrics']['per_class'][class_name]['precision'] for r in model_results],
                    'recall': [r['metrics']['per_class'][class_name]['recall'] for r in model_results],
                    'sensitivity': [r['metrics']['per_class'][class_name]['sensitivity'] for r in model_results],
                    'specificity': [r['metrics']['per_class'][class_name]['specificity'] for r in model_results]
                }
            
            # Calculate statistics
            aggregated_results[model_name] = {
                'accuracy': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'values': accuracies
                },
                'precision_macro': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'values': precisions
                },
                'recall_macro': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'values': recalls
                },
                'per_class': {}
            }
            
            for class_name, class_metrics in per_class_metrics.items():
                aggregated_results[model_name]['per_class'][class_name] = {
                    'precision': {
                        'mean': np.mean(class_metrics['precision']),
                        'std': np.std(class_metrics['precision']),
                        'values': class_metrics['precision']
                    },
                    'recall': {
                        'mean': np.mean(class_metrics['recall']),
                        'std': np.std(class_metrics['recall']),
                        'values': class_metrics['recall']
                    },
                    'sensitivity': {
                        'mean': np.mean(class_metrics['sensitivity']),
                        'std': np.std(class_metrics['sensitivity']),
                        'values': class_metrics['sensitivity']
                    },
                    'specificity': {
                        'mean': np.mean(class_metrics['specificity']),
                        'std': np.std(class_metrics['specificity']),
                        'values': class_metrics['specificity']
                    }
                }
            
            # Print results
            print(f"\n{model_name} Results (Mean ± Std):")
            print(f"Accuracy: {aggregated_results[model_name]['accuracy']['mean']:.4f} ± {aggregated_results[model_name]['accuracy']['std']:.4f}")
            print(f"Precision (macro): {aggregated_results[model_name]['precision_macro']['mean']:.4f} ± {aggregated_results[model_name]['precision_macro']['std']:.4f}")
            print(f"Recall (macro): {aggregated_results[model_name]['recall_macro']['mean']:.4f} ± {aggregated_results[model_name]['recall_macro']['std']:.4f}")
            
            print(f"\nPer-class metrics:")
            for class_name in self.label_encoder.classes_:
                class_results = aggregated_results[model_name]['per_class'][class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {class_results['precision']['mean']:.4f} ± {class_results['precision']['std']:.4f}")
                print(f"    Recall: {class_results['recall']['mean']:.4f} ± {class_results['recall']['std']:.4f}")
                print(f"    Sensitivity: {class_results['sensitivity']['mean']:.4f} ± {class_results['sensitivity']['std']:.4f}")
                print(f"    Specificity: {class_results['specificity']['mean']:.4f} ± {class_results['specificity']['std']:.4f}")
        
        # Save aggregated results
        aggregated_path = os.path.join(self.results_dir, 'aggregated_results.pkl')
        with open(aggregated_path, 'wb') as f:
            pickle.dump(aggregated_results, f)
        
        # Save detailed results
        detailed_path = os.path.join(self.results_dir, 'detailed_results.pkl')
        with open(detailed_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Create summary report
        self.create_summary_report(aggregated_results)
        
        print(f"\n✓ Results saved to {self.results_dir}")
        print(f"✓ Aggregated results saved to {aggregated_path}")
        print(f"✓ Detailed results saved to {detailed_path}")
    
    def create_summary_report(self, aggregated_results):
        """Create a comprehensive summary report"""
        report_path = os.path.join(self.results_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("SLEEP BREATHING IRREGULARITY CLASSIFICATION - SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("EXPERIMENTAL SETUP:\n")
            f.write(f"- Cross-validation: Leave-One-Participant-Out\n")
            f.write(f"- Models: 1D CNN, 1D Conv-LSTM\n")
            f.write(f"- Target Classes: {', '.join(self.target_classes)}\n")
            f.write(f"- Actual Classes: {', '.join(self.label_encoder.classes_)}\n")
            f.write(f"- Signal length: {self.max_signal_length} samples\n")
            f.write(f"- Signals: {', '.join(self.signal_names)}\n\n")
            
            f.write("WHY LEAVE-ONE-PARTICIPANT-OUT CV?\n")
            f.write("- Prevents data leakage: physiological signals from same person are correlated\n")
            f.write("- Better generalization: tests ability to work on unseen individuals\n")
            f.write("- Realistic evaluation: mimics real-world deployment scenario\n")
            f.write("- Avoids overfitting to participant-specific patterns\n\n")
            
            for model_name, results in aggregated_results.items():
                f.write(f"{model_name} RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}\n")
                f.write(f"Precision (macro): {results['precision_macro']['mean']:.4f} ± {results['precision_macro']['std']:.4f}\n")
                f.write(f"Recall (macro): {results['recall_macro']['mean']:.4f} ± {results['recall_macro']['std']:.4f}\n\n")
                
                f.write("Per-class metrics:\n")
                for class_name in self.label_encoder.classes_:
                    class_results = results['per_class'][class_name]
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {class_results['precision']['mean']:.4f} ± {class_results['precision']['std']:.4f}\n")
                    f.write(f"    Recall: {class_results['recall']['mean']:.4f} ± {class_results['recall']['std']:.4f}\n")
                    f.write(f"    Sensitivity: {class_results['sensitivity']['mean']:.4f} ± {class_results['sensitivity']['std']:.4f}\n")
                    f.write(f"    Specificity: {class_results['specificity']['mean']:.4f} ± {class_results['specificity']['std']:.4f}\n")
                f.write("\n")
        
        print(f"✓ Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Train sleep breathing irregularity classification models')
    parser.add_argument('-dataset', type=str, default='Dataset/dataset.pkl',
                       help='Path to dataset pickle file (default: Dataset/dataset.pkl)')
    parser.add_argument('-results_dir', type=str, default='Results',
                       help='Directory to save results (default: Results)')
    parser.add_argument('-epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('-batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('-learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"✗ Error: Dataset file '{args.dataset}' not found")
        print("Please run create_dataset.py first")
        return
    
    # Initialize trainer
    trainer = SleepModelTrainer(args.dataset, args.results_dir)
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    
    # Train and evaluate
    try:
        results = trainer.train_and_evaluate()
        print(f"\n{'='*60}")
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Results saved to: {args.results_dir}")
        print(f"Check the summary report for detailed analysis")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()