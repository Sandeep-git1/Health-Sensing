import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

class SleepDataProcessor:
    """Optimized sleep data processor with reduced redundancy"""
    
    def __init__(self):
        self.timestamp_formats = [
            '%d.%m.%Y %H:%M:%S,%f',
            '%d.%m.%Y %H:%M:%S.%f',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S,%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S'
        ]
        self.signal_color = '#000080'  # Navy blue for all signals
        
        # Updated event colors - only Hypopnea and Apnea with pink and grey colors
        self.event_colors = {
            'Hypopnea': '#FF69B4',           # Pink color for Hypopnea
            'Obstructive Apnea': '#808080',  # Grey color for Apnea
            'Central Apnea': '#808080',      # Grey color for Apnea
            'Mixed Apnea': '#808080',        # Grey color for Apnea
            'Apnea': '#808080'               # Grey color for generic Apnea
        }
        self.sample_rates = {'flow': 32, 'thoracic': 32, 'spo2': 4}
    
    def parse_timestamp(self, timestamp_str):
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
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
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
            # Validate sample rate
            if len(df) > 1:
                time_diff = (df.index[1] - df.index[0]).total_seconds()
                actual_rate = 1 / time_diff if time_diff > 0 else 0
                expected_rate = self.sample_rates.get(signal_type, 32)
                if abs(actual_rate - expected_rate) < 0.1:
                    print(f"Sample rate validation passed for {file_path}: {actual_rate:.2f} Hz")
                else:
                    print(f"Sample rate validation failed for {file_path}: Expected {expected_rate} Hz, got {actual_rate:.2f} Hz")
            print(f"Loaded {signal_type}: {len(df)} points")
            return df
        except Exception as e:
            print(f"Error loading {signal_type}: {e}")
            return None
    
    def load_events(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            events = []
            for line in lines:
                line = line.strip()
                if line and '-' in line and line.count(';') >= 2:
                    try:
                        parts = line.split(';')
                        if len(parts) >= 3:
                            time_range = parts[0].strip()
                            event_type = parts[2].strip()
                            
                            # Filter to only include Hypopnea and Apnea events
                            if 'hypopnea' in event_type.lower() or 'apnea' in event_type.lower():
                                if '-' in time_range:
                                    start_str, end_str = time_range.split('-', 1)
                                    start_str = start_str.strip()
                                    end_str = end_str.strip()
                                    if ' ' not in end_str:
                                        date_part = start_str.split()[0]
                                        end_str = f"{date_part} {end_str}"
                                    start_time = self.parse_timestamp(start_str)
                                    end_time = self.parse_timestamp(end_str)
                                    if start_time and end_time:
                                        events.append({
                                            'start_time': start_time,
                                            'end_time': end_time,
                                            'event_type': event_type
                                        })
                    except:
                        continue
            df = pd.DataFrame(events) if events else pd.DataFrame()
            print(f"Loaded {len(df)} hypopnea and apnea events")
            return df
        except Exception as e:
            print(f"Error loading events: {e}")
            return pd.DataFrame()
    
    def find_files(self, data_folder):
        files = {}
        event_file = None
        sleep_file = None
        for file in os.listdir(data_folder):
            if file.endswith(('.txt', '.csv')):
                file_path = os.path.join(data_folder, file)
                file_lower = file.lower()
                if 'events' in file_lower or 'flow_events' in file_lower:
                    event_file = file_path
                    print(f"Events file: {file_path}")
                elif 'sleep' in file_lower and 'profile' in file_lower:
                    sleep_file = file_path
                    print(f"Sleep file: {file_path}")
                elif any(kw in file_lower for kw in ['flow', 'nasal', 'airflow']) and 'events' not in file_lower:
                    files['flow'] = file_path
                    print(f"Flow file: {file_path}")
                elif any(kw in file_lower for kw in ['thorac', 'chest', 'movement']):
                    files['thoracic'] = file_path
                    print(f"Thoracic file: {file_path}")
                elif 'spo2' in file_lower:
                    files['spo2'] = file_path
                    print(f"SPO2 file: {file_path}")
        return files, event_file, sleep_file

def create_visualization(data_folder, output_dir):
    participant_name = os.path.basename(data_folder)
    processor = SleepDataProcessor()
    print(f"Processing {participant_name}...")
    
    signal_files, event_file, sleep_file = processor.find_files(data_folder)
    if not signal_files:
        print(f"No signal files found in {data_folder}")
        return
    
    signals = {}
    for signal_type, file_path in signal_files.items():
        data = processor.load_signal_data(file_path, signal_type)
        if data is not None:
            signals[signal_type] = data
    
    events = processor.load_events(event_file)
    if not signals:
        print(f"No valid signals loaded for {participant_name}")
        return
    
    # Create figure with proper sizing for 8-hour duration visualization
    fig, axes = plt.subplots(len(signals), 1, figsize=(15, 4 * len(signals)))
    if len(signals) == 1:
        axes = [axes]
    
    fig.suptitle(f'Sleep Study Data - {participant_name}', fontsize=16, fontweight='bold')
    
    # Updated signal info to match test requirements
    signal_info = {
        'flow': ('Nasal Airflow (32 Hz)', 'Flow'),
        'thoracic': ('Thoracic Movement (32 Hz)', 'Movement'),
        'spo2': ('SpO2 - Oxygen Saturation (4 Hz)', 'SpO2 (%)')
    }
    
    for i, (signal_type, data) in enumerate(signals.items()):
        ax = axes[i]
        title, ylabel = signal_info.get(signal_type, (signal_type.title(), 'Value'))
        ax.plot(data.index, data['value'], color=processor.signal_color, linewidth=0.6, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Overlay hypopnea and apnea events only (pink and grey colors)
        if not events.empty:
            plotted_events = set()
            for _, event in events.iterrows():
                event_type = event['event_type']
                # Only plot hypopnea and apnea events
                if 'hypopnea' in event_type.lower() or 'apnea' in event_type.lower():
                    if event_type not in plotted_events:
                        color = processor.event_colors.get(event_type, '#808080')  # Default grey for apnea
                        ax.axvspan(event['start_time'], event['end_time'], 
                                  alpha=0.3, color=color, label=event_type)
                        plotted_events.add(event_type)
                    else:
                        color = processor.event_colors.get(event_type, '#808080')  # Default grey for apnea
                        ax.axvspan(event['start_time'], event['end_time'], 
                                  alpha=0.3, color=color)
            if plotted_events:
                ax.legend(loc='upper right', fontsize=10)
        
        # Format x-axis for 8-hour duration
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # Show every hour
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    axes[-1].set_xlabel('Time (Hours)')
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f'{participant_name}_visualization.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Visualization saved to: {pdf_path}")
    return pdf_path

def main():
    parser = argparse.ArgumentParser(description='Generate sleep study visualizations')
    parser.add_argument('-name', type=str, required=True, 
                       help='Path to participant data folder (e.g., "Data/AP28")')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.name):
        print(f"Error: Data folder '{args.name}' not found!")
        return
    
    try:
        create_visualization(args.name, 'Visualizations')
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error creating visualization: {e}")

if __name__ == "__main__":
    main()