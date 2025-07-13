import pandas as pd
import numpy as np
import os
import argparse
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """Data cleaning class for respiratory signals"""
    
    def __init__(self):
        self.timestamp_formats = [
            '%d.%m.%Y %H:%M:%S,%f',
            '%d.%m.%Y %H:%M:%S.%f',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S,%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S'
        ]
        self.sample_rates = {'flow': 32, 'thoracic': 32, 'spo2': 4}
        
        # Breathing frequency range: 10-24 breaths per minute
        # 10 BrPM = 10/60 = 0.167 Hz
        # 24 BrPM = 24/60 = 0.4 Hz
        self.breathing_freq_range = (0.17, 0.4)
        
        # Low-pass filter cutoff (allow some margin above breathing frequency)
        self.lowpass_cutoff = 2.0  # Hz - removes high frequency noise
        
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp with multiple format support"""
        for fmt in self.timestamp_formats:
            try:
                return pd.to_datetime(timestamp_str, format=fmt)
            except:
                continue
        try:
            return pd.to_datetime(timestamp_str)
        except:
            return None
    
    def load_signal_data(self, file_path, signal_type):
        """Load signal data from file"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Find data start line
            data_start = next((i for i, line in enumerate(lines) 
                             if line.strip().lower() in ['data:', 'data']), 5)
            
            data_lines = []
            for line in lines[data_start:]:
                line = line.strip()
                if line and ';' in line and line.count(';') == 1:
                    parts = line.split(';')
                    if len(parts) >= 2:
                        data_lines.append((parts[0].strip(), parts[1].strip()))
            
            if not data_lines:
                raise ValueError(f"No valid data in {file_path}")
            
            timestamps, values = zip(*data_lines)
            parsed_timestamps = [self.parse_timestamp(ts) or pd.NaT for ts in timestamps]
            numeric_values = pd.to_numeric(values, errors='coerce')
            
            df = pd.DataFrame({'timestamp': parsed_timestamps, 'value': numeric_values})
            df = df.dropna().set_index('timestamp').sort_index()
            
            if len(df) > 1:
                sample_rate = self.sample_rates.get(signal_type, 32)
                freq = f'{1000//sample_rate}ms'
                df = df.resample(freq).mean().interpolate(method='linear')
            
            print(f"Loaded {signal_type}: {len(df)} points")
            return df
        
        except Exception as e:
            print(f"Error loading {signal_type}: {e}")
            return None
    
    def design_filters(self, fs):
        """Design digital filters for noise removal"""
        # Butterworth low-pass filter to remove high-frequency noise
        nyquist = fs / 2
        low_cutoff = min(self.lowpass_cutoff, nyquist * 0.8)  # Ensure cutoff is below Nyquist
        
        # Low-pass filter coefficients
        b_low, a_low = butter(4, low_cutoff / nyquist, btype='low', analog=False)
        
        # Band-pass filter for breathing frequency range (optional additional filtering)
        if self.breathing_freq_range[1] < nyquist:
            b_band, a_band = butter(2, 
                                   [self.breathing_freq_range[0] / nyquist, 
                                    self.breathing_freq_range[1] / nyquist], 
                                   btype='band', analog=False)
        else:
            b_band, a_band = None, None
            
        return (b_low, a_low), (b_band, a_band)
    
    def apply_filters(self, data, fs, signal_type):
        """Apply digital filters to remove noise"""
        if len(data) < 100:  # Need minimum data points for filtering
            print(f"Warning: Not enough data points for filtering {signal_type}")
            return data
        
        # Design filters
        (b_low, a_low), (b_band, a_band) = self.design_filters(fs)
        
        # Apply low-pass filter to remove high-frequency noise
        try:
            filtered_data = filtfilt(b_low, a_low, data)
            print(f"Applied low-pass filter to {signal_type} (cutoff: {self.lowpass_cutoff} Hz)")
        except Exception as e:
            print(f"Error applying low-pass filter to {signal_type}: {e}")
            filtered_data = data
        
        # For respiratory signals, optionally apply band-pass filter
        if signal_type in ['flow', 'thoracic'] and b_band is not None:
            try:
                filtered_data = filtfilt(b_band, a_band, filtered_data)
                print(f"Applied band-pass filter to {signal_type} ({self.breathing_freq_range[0]}-{self.breathing_freq_range[1]} Hz)")
            except Exception as e:
                print(f"Error applying band-pass filter to {signal_type}: {e}")
        
        # Additional smoothing for SpO2 (typically noisier)
        if signal_type == 'spo2':
            try:
                # Apply Savitzky-Golay filter for smoothing
                window_length = min(21, len(filtered_data) // 10)  # Adaptive window
                if window_length % 2 == 0:
                    window_length += 1
                if window_length >= 5:
                    filtered_data = savgol_filter(filtered_data, window_length, 3)
                    print(f"Applied Savitzky-Golay smoothing to {signal_type}")
            except Exception as e:
                print(f"Error applying smoothing to {signal_type}: {e}")
        
        return filtered_data
    
    def clean_signal(self, df, signal_type):
        """Clean a single signal"""
        if df is None or len(df) == 0:
            return None
        
        # Calculate sampling rate
        if len(df) > 1:
            time_diff = (df.index[1] - df.index[0]).total_seconds()
            fs = 1 / time_diff if time_diff > 0 else self.sample_rates.get(signal_type, 32)
        else:
            fs = self.sample_rates.get(signal_type, 32)
        
        print(f"Cleaning {signal_type} signal (fs = {fs:.2f} Hz)")
        
        # Apply filters
        original_values = df['value'].values
        cleaned_values = self.apply_filters(original_values, fs, signal_type)
        
        # Create cleaned dataframe
        cleaned_df = df.copy()
        cleaned_df['value'] = cleaned_values
        
        # Calculate noise reduction metrics
        original_std = np.std(original_values)
        cleaned_std = np.std(cleaned_values)
        noise_reduction = ((original_std - cleaned_std) / original_std) * 100
        
        print(f"Noise reduction for {signal_type}: {noise_reduction:.2f}%")
        
        return cleaned_df
    
    def find_files(self, data_folder):
        """Find signal files in the data folder"""
        files = {}
        for file in os.listdir(data_folder):
            if file.endswith(('.txt', '.csv')):
                file_path = os.path.join(data_folder, file)
                file_lower = file.lower()
                
                if 'events' in file_lower or 'sleep' in file_lower:
                    continue  # Skip event and sleep files
                elif any(kw in file_lower for kw in ['flow', 'nasal']):
                    files['flow'] = file_path
                elif any(kw in file_lower for kw in ['thorac', 'chest']):
                    files['thoracic'] = file_path
                elif 'spo2' in file_lower:
                    files['spo2'] = file_path
        
        return files
    
    def save_cleaned_data(self, cleaned_signals, output_folder, participant_name):
        """Save cleaned signals to files"""
        os.makedirs(output_folder, exist_ok=True)
        
        participant_folder = os.path.join(output_folder, participant_name)
        os.makedirs(participant_folder, exist_ok=True)
        
        for signal_type, data in cleaned_signals.items():
            if data is not None:
                output_file = os.path.join(participant_folder, f'{signal_type}_cleaned.csv')
                
                # Save with timestamp and value columns
                data.to_csv(output_file, index=True, header=['value'])
                print(f"Saved cleaned {signal_type} to {output_file}")
    
    def create_comparison_plot(self, original_signals, cleaned_signals, participant_name, output_folder):
        """Create before/after comparison plots"""
        if not original_signals or not cleaned_signals:
            return
        
        n_signals = len(original_signals)
        fig, axes = plt.subplots(n_signals, 2, figsize=(15, 4 * n_signals))
        
        if n_signals == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Data Cleaning Results - {participant_name}', fontsize=16, fontweight='bold')
        
        signal_names = {
            'flow': 'Nasal Airflow',
            'thoracic': 'Thoracic Movement', 
            'spo2': 'SpO2 Oxygen Saturation'
        }
        
        for i, signal_type in enumerate(original_signals.keys()):
            if signal_type in cleaned_signals:
                original_data = original_signals[signal_type]
                cleaned_data = cleaned_signals[signal_type]
                
                signal_name = signal_names.get(signal_type, signal_type.title())
                
                # Plot original signal
                axes[i, 0].plot(original_data.index, original_data['value'], 
                               color='red', linewidth=0.5, alpha=0.7, label='Original')
                axes[i, 0].set_title(f'{signal_name} - Original', fontsize=12)
                axes[i, 0].set_ylabel('Value')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Plot cleaned signal
                axes[i, 1].plot(cleaned_data.index, cleaned_data['value'], 
                               color='blue', linewidth=0.5, alpha=0.7, label='Cleaned')
                axes[i, 1].set_title(f'{signal_name} - Cleaned', fontsize=12)
                axes[i, 1].set_ylabel('Value')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Format x-axis
                for ax in axes[i, :]:
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save comparison plot
        plot_path = os.path.join(output_folder, f'{participant_name}_cleaning_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {plot_path}")

def clean_participant_data(input_folder, output_folder):
    """Clean data for a single participant"""
    participant_name = os.path.basename(input_folder)
    cleaner = DataCleaner()
    
    print(f"Processing participant: {participant_name}")
    
    # Find signal files
    signal_files = cleaner.find_files(input_folder)
    if not signal_files:
        print(f"No signal files found in {input_folder}")
        return
    
    # Load original signals
    original_signals = {}
    for signal_type, file_path in signal_files.items():
        data = cleaner.load_signal_data(file_path, signal_type)
        if data is not None:
            original_signals[signal_type] = data
    
    if not original_signals:
        print(f"No valid signals loaded for {participant_name}")
        return
    
    # Clean signals
    cleaned_signals = {}
    for signal_type, data in original_signals.items():
        cleaned_data = cleaner.clean_signal(data, signal_type)
        if cleaned_data is not None:
            cleaned_signals[signal_type] = cleaned_data
    
    # Save cleaned data
    cleaner.save_cleaned_data(cleaned_signals, output_folder, participant_name)
    
    # Create comparison plots
    cleaner.create_comparison_plot(original_signals, cleaned_signals, 
                                  participant_name, output_folder)
    
    print(f"Data cleaning completed for {participant_name}")

def main():
    parser = argparse.ArgumentParser(description='Clean sleep study data by removing high-frequency noise')
    parser.add_argument('-input', type=str, required=True, 
                       help='Input folder path (single participant or parent folder)')
    parser.add_argument('-output', type=str, default='Cleaned_Data',
                       help='Output folder path (default: Cleaned_Data)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' not found!")
        return
    
    try:
        # Check if input is a single participant folder or parent folder
        if any(f.endswith(('.txt', '.csv')) for f in os.listdir(args.input)):
            # Single participant folder
            clean_participant_data(args.input, args.output)
        else:
            # Parent folder with multiple participants
            for participant_folder in os.listdir(args.input):
                participant_path = os.path.join(args.input, participant_folder)
                if os.path.isdir(participant_path):
                    clean_participant_data(participant_path, args.output)
        
        print("Data cleaning completed successfully!")
        
    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    main()