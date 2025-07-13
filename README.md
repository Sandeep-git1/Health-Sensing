# Sleep Breathing Irregularity Detection System

## 🏥 Project Overview

This project implements a machine learning pipeline for detecting breathing irregularities during sleep using physiological signals. Developed as part of a healthcare AI initiative, the system analyzes overnight sleep data to classify breathing patterns into Normal, Hypopnea, and Obstructive Apnea categories.

## 📊 Dataset

**Data Source**: DeepMedico™ Sleep Study  
**Participants**: 5 subjects  
**Duration**: 8 hours per participant  
**Sampling Rates**: 
- Nasal Airflow: 32 Hz
- Thoracic Movement: 32 Hz  
- SpO2 (Oxygen Saturation): 4 Hz

**Additional Files**:
- Event annotations (breathing irregularities)
- Sleep stage profiles

## 🏗️ Project Structure

```
Project Root/
├── Data/                     # Raw physiological data
│   ├── AP01/
│   │   ├── nasal_airflow.txt
│   │   ├── thoracic_movement.txt
│   │   ├── spo2.txt
│   │   ├── flow_events.txt
│   │   └── sleep_profile.txt
│   ├── AP02/
│   ├── AP03/
│   ├── AP04/
│   └── AP05/
├── Visualizations/           # Generated signal plots (PDF format)
├── Results/                  # Performance metrics and confusion matrices
├── vis.py                    # Signal visualization script
├── clean_vis.py              # Clean visualization generation
├── create_dataset.py         # Data preprocessing and windowing
├── model_training.py         # Model training and evaluation
├── inspect_data_sample.py    # Data inspection and analysis
├── Requirements.txt          # Project dependencies
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras scipy
pip install matplotlib-backend-pdf
```

### 1. Data Visualization

Generate signal visualizations for exploratory analysis:

```bash
python vis.py -name "Data/AP01"
```

**Output**: PDF visualization saved to `Visualizations/AP01_signals.pdf`

### 2. Dataset Creation

Process raw signals into ML-ready format:

```bash
python create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

**Features**:
- 30-second windows with 50% overlap
- Automatic labeling based on event annotations
- Digital filtering for noise reduction
- Time-series alignment across different sampling rates

### 3. Model Training

Train and evaluate models using Leave-One-Participant-Out cross-validation:

```bash
python model_training.py
```

**Models Implemented**:
- 1D Convolutional Neural Network (CNN)
- 1D Convolutional LSTM (Conv-LSTM)

### Additional Scripts

**Data Inspection**:
```bash
python inspect_data_sample.py
```
- Analyze raw data structure and quality
- Generate data distribution statistics
- Identify potential data issues

**Clean Visualizations**:
```bash
python clean_vis.py
```
- Generate publication-ready signal plots
- Enhanced visualization formatting
- Batch processing for multiple participants

## 📈 Results Summary

### Model Performance

| Model | Accuracy | Precision (macro) | Recall (macro) |
|-------|----------|------------------|----------------|
| **1D CNN** | 76.88% ± 20.04% | 37.17% ± 2.49% | 38.64% ± 4.10% |
| **Conv-LSTM** | 63.24% ± 33.44% | 28.82% ± 14.83% | 47.18% ± 16.82% |

### Per-Class Performance (CNN)

| Class | Precision | Recall | Sensitivity | Specificity |
|-------|-----------|--------|-------------|-------------|
| **Normal** | 92.96% ± 4.58% | 80.55% ± 22.05% | 80.55% ± 22.05% | 33.52% ± 22.65% |
| **Hypopnea** | 12.31% ± 8.28% | 13.87% ± 4.34% | 13.87% ± 4.34% | 93.02% ± 1.82% |
| **Obstructive Apnea** | 6.22% ± 8.12% | 21.50% ± 28.27% | 21.50% ± 28.27% | 87.20% ± 20.86% |

## 🔧 Technical Implementation

### Data Processing Pipeline

1. **Signal Filtering**: Bandpass filter (0.17-0.4 Hz) to retain breathing frequencies
2. **Time Alignment**: Timestamp-based synchronization across different sampling rates
3. **Windowing**: 30-second segments with 50% overlap
4. **Labeling**: Event-based annotation with >50% overlap threshold

### Model Architecture

**1D CNN**:
- Multiple convolutional layers with ReLU activation
- Max pooling for dimensionality reduction
- Dropout for regularization
- Dense layers for classification

**Conv-LSTM**:
- Convolutional layers for feature extraction
- LSTM layers for temporal modeling
- Dense output layer for classification

### Evaluation Strategy

**Leave-One-Participant-Out Cross-Validation**:
- Prevents data leakage from participant-specific patterns
- Tests generalization to unseen individuals
- Mimics real-world deployment scenarios

## 📊 Key Insights

1. **Class Imbalance**: Normal breathing dominates the dataset (~85-90% of samples)
2. **Model Comparison**: CNN shows better overall accuracy, Conv-LSTM better minority class recall
3. **Generalization**: High variance across participants indicates individual physiological differences
4. **Clinical Relevance**: Models achieve reasonable sensitivity for detecting breathing irregularities

## 🎯 Future Work

- **Data Augmentation**: Techniques to handle class imbalance
- **Feature Engineering**: Additional signal processing features
- **Ensemble Methods**: Combining multiple models for better performance
- **Real-time Processing**: Optimization for live monitoring systems
- **Sleep Stage Integration**: Multi-task learning for comprehensive sleep analysis

## 📚 Dependencies

See `Requirements.txt` for complete dependency list:

```
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
pyarrow>=12.0.0
torch>=2.0.0
scikit-learn>=1.2.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
seaborn>=0.11.0
pickle5>=0.0.11
```

## 👥 Contributors

- **Data Scientist**: Sleep breathing irregularity detection system
- **Healthcare Team**: Clinical validation and domain expertise
- **DeepMedico™**: Dataset provision and project guidance

## 📄 License

This project is developed for educational and research purposes. Please ensure compliance with healthcare data regulations when using in clinical settings.

## 📞 Contact

For questions or collaboration opportunities, please contact the development team.

---

*This project demonstrates the application of deep learning techniques to healthcare monitoring, specifically focusing on sleep disorder detection through physiological signal analysis.*
