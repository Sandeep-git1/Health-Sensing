import pandas as pd
import numpy as np
import os
import argparse
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DatasetCreator:
    """Dataset creator for sleep study data using cleaned signals"""
    
    def __init__(self, window_size=30, overlap=0.5):
        self.window_size = window_size  # 30 seconds
        self.overlap = overlap  # 50% overlap
        self.step_size = window_size * (1 - overlap)  # 15 seconds
        
        # Target event labels (case-insensitive matching)
        self.target_events = ['hypopnea', 'obstructive apnea', 'obstructive_apnea']
        
        # Timestamp formats to try for event files
        self.timestamp_formats = [
            '%d.%m.%Y %H:%M:%S,%f',
            '%d.%m.%Y %H:%M:%S.%f',
            '%d.%m.%Y %H:%M:%S',
            '%Y-%m-%d %H:%M:%S,%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S'
        ]
        
        # Validation thresholds
        self.min_signal_points = 10  # Minimum data points in a window
        self.max_event_duration = 300  # Maximum event duration in seconds (5 minutes)
        self.min_event_duration = 5   # Minimum event duration in seconds
        
    def parse_timestamp(self, timestamp_str):
        """Parse timestamp with multiple format support"""
        timestamp_str = str(timestamp_str).strip().strip('"')
        
        for fmt in self.timestamp_formats:
            try:
                return pd.to_datetime(timestamp_str, format=fmt)
            except:
                continue
        
        try:
            return pd.to_datetime(timestamp_str)
        except:
            return None
    
    def validate_signals(self, signals):
        """Validate loaded signals for quality and consistency"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if not signals:
            validation_results['valid'] = False
            validation_results['errors'].append("No signals loaded")
            return validation_results
        
        # Check signal lengths and sampling rates
        for signal_name, signal_data in signals.items():
            if len(signal_data) < 100:
                validation_results['warnings'].append(f"{signal_name}: Very short signal ({len(signal_data)} points)")
            
            # Check for excessive missing values
            if signal_data['value'].isna().sum() > len(signal_data) * 0.1:
                validation_results['warnings'].append(f"{signal_name}: >10% missing values")
            
            # Check for constant values (potential sensor issues)
            if signal_data['value'].nunique() < 5:
                validation_results['warnings'].append(f"{signal_name}: Very few unique values (potential sensor issue)")
        
        # Check time alignment
        signal_starts = [signal.index.min() for signal in signals.values()]
        signal_ends = [signal.index.max() for signal in signals.values()]
        
        max_start = max(signal_starts)
        min_end = min(signal_ends)
        
        if max_start >= min_end:
            validation_results['valid'] = False
            validation_results['errors'].append("No overlapping time period across signals")
        else:
            overlap_hours = (min_end - max_start).total_seconds() / 3600
            if overlap_hours < 4:
                validation_results['warnings'].append(f"Short overlap period: {overlap_hours:.1f} hours")
        
        return validation_results
    
    def load_cleaned_signals(self, cleaned_folder):
        """Load cleaned signals from CSV files with validation"""
        signals = {}
        
        # Expected cleaned signal files
        signal_files = {
            'nasal_airflow': 'flow_cleaned.csv',
            'thoracic_movement': 'thoracic_cleaned.csv', 
            'spo2': 'spo2_cleaned.csv'
        }
        
        for signal_type, filename in signal_files.items():
            file_path = os.path.join(cleaned_folder, filename)
            
            if os.path.exists(file_path):
                try:
                    # Load cleaned CSV file
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    
                    # Handle different column structures
                    if len(df.columns) == 1:
                        df.columns = ['value']
                    elif 'value' not in df.columns:
                        # Take the first non-index column as value
                        df = df.iloc[:, [0]]
                        df.columns = ['value']
                    
                    # Sort by timestamp and clean
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep='first')]
                    
                    # Remove invalid values
                    initial_len = len(df)
                    df = df.dropna()
                    df = df[np.isfinite(df['value'])]
                    
                    if len(df) < initial_len * 0.9:
                        print(f"  Warning: Removed {initial_len - len(df)} invalid values from {signal_type}")
                    
                    signals[signal_type] = df
                    print(f"✓ Loaded cleaned {signal_type}: {len(df)} data points")
                    
                except Exception as e:
                    print(f"✗ Error loading {signal_type} from {file_path}: {e}")
            else:
                print(f"✗ File not found: {file_path}")
        
        return signals
    
    def load_events_file(self, data_folder):
        """Load events from the original Data folder with better error handling"""
        events_file = None
        
        # Look for event files in the original data folder
        for file in os.listdir(data_folder):
            if any(keyword in file.lower() for keyword in ['event', 'flow_event', 'flow-event']):
                events_file = os.path.join(data_folder, file)
                break
        
        if not events_file:
            print("  No event file found")
            return pd.DataFrame()
        
        print(f"  Loading events from: {os.path.basename(events_file)}")
        
        try:
            events = []
            
            # Try different parsing approaches
            parsing_successful = False
            
            # Approach 1: Try pandas with different separators
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(events_file, sep=sep, encoding='utf-8', on_bad_lines='skip')
                    if len(df.columns) >= 3 and len(df) > 0:
                        print(f"  Successfully parsed with separator: '{sep}'")
                        for _, row in df.iterrows():
                            if len(row) >= 3:
                                event_type = str(row.iloc[2]).lower().strip()
                                if any(target in event_type for target in self.target_events):
                                    time_range = str(row.iloc[0]).strip()
                                    if '-' in time_range:
                                        start_str, end_str = time_range.split('-', 1)
                                        start_time = self.parse_timestamp(start_str.strip())
                                        end_time = self.parse_timestamp(end_str.strip())
                                        
                                        if start_time and end_time and end_time > start_time:
                                            duration = (end_time - start_time).total_seconds()
                                            if self.min_event_duration <= duration <= self.max_event_duration:
                                                # Normalize event type
                                                if 'hypopnea' in event_type:
                                                    normalized_type = 'Hypopnea'
                                                elif 'obstructive' in event_type:
                                                    normalized_type = 'Obstructive Apnea'
                                                else:
                                                    normalized_type = event_type.title()
                                                
                                                events.append({
                                                    'start': start_time,
                                                    'end': end_time,
                                                    'event': normalized_type,
                                                    'duration': duration
                                                })
                        parsing_successful = True
                        break
                except Exception as e:
                    continue
            
            # Approach 2: Manual line-by-line parsing
            if not parsing_successful:
                print("  Falling back to manual parsing...")
                with open(events_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and ';' in line:
                        parts = [p.strip() for p in line.split(';')]
                        if len(parts) >= 3:
                            event_type = parts[2].lower().strip()
                            if any(target in event_type for target in self.target_events):
                                time_range = parts[0]
                                if '-' in time_range:
                                    try:
                                        start_str, end_str = time_range.split('-', 1)
                                        start_time = self.parse_timestamp(start_str.strip())
                                        
                                        # Handle end time that might be missing date
                                        end_str = end_str.strip()
                                        if start_time and not any(c.isdigit() and len(c) == 4 for c in end_str.split('.')):
                                            date_part = start_time.strftime('%d.%m.%Y')
                                            end_str = f"{date_part} {end_str}"
                                        
                                        end_time = self.parse_timestamp(end_str)
                                        
                                        if start_time and end_time and end_time > start_time:
                                            duration = (end_time - start_time).total_seconds()
                                            if self.min_event_duration <= duration <= self.max_event_duration:
                                                # Normalize event type
                                                if 'hypopnea' in event_type:
                                                    normalized_type = 'Hypopnea'
                                                elif 'obstructive' in event_type:
                                                    normalized_type = 'Obstructive Apnea'
                                                else:
                                                    normalized_type = event_type.title()
                                                
                                                events.append({
                                                    'start': start_time,
                                                    'end': end_time,
                                                    'event': normalized_type,
                                                    'duration': duration
                                                })
                                    except Exception as e:
                                        print(f"  Warning: Error parsing line {line_num}: {e}")
                                        continue
            
            if events:
                events_df = pd.DataFrame(events)
                events_df = events_df.sort_values('start')
                
                # Validate events
                initial_count = len(events_df)
                events_df = events_df[events_df['duration'] > 0]
                
                print(f"✓ Loaded {len(events_df)} valid events")
                if len(events_df) < initial_count:
                    print(f"  Filtered out {initial_count - len(events_df)} invalid events")
                
                print(f"  Event distribution: {events_df['event'].value_counts().to_dict()}")
                print(f"  Duration range: {events_df['duration'].min():.1f}s - {events_df['duration'].max():.1f}s")
                
                return events_df
            else:
                print("✗ No valid events found")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"✗ Error loading events: {e}")
            return pd.DataFrame()
    
    def create_windows(self, signals, events, participant_name):
        """Create 30-second windows with 50% overlap and validation"""
        if not signals:
            return []
        
        # Validate signals first
        validation = self.validate_signals(signals)
        if not validation['valid']:
            print(f"✗ Signal validation failed: {validation['errors']}")
            return []
        
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        
        # Find common time range across all signals
        signal_starts = [signal.index.min() for signal in signals.values()]
        signal_ends = [signal.index.max() for signal in signals.values()]
        
        start_time = max(signal_starts)
        end_time = min(signal_ends)
        
        if start_time >= end_time:
            print(f"✗ No overlapping time found for {participant_name}")
            return []
        
        overlap_hours = (end_time - start_time).total_seconds() / 3600
        print(f"  Creating windows from {start_time} to {end_time} ({overlap_hours:.1f} hours)")
        
        windows = []
        current_time = start_time
        window_id = 0
        
        while current_time + pd.Timedelta(seconds=self.window_size) <= end_time:
            window_end = current_time + pd.Timedelta(seconds=self.window_size)
            
            # Extract signal data for this window
            window_signals = {}
            valid_signals = 0
            
            for signal_name, signal_data in signals.items():
                window_signal = signal_data[
                    (signal_data.index >= current_time) & 
                    (signal_data.index < window_end)
                ]
                
                if len(window_signal) >= self.min_signal_points:
                    valid_signals += 1
                    window_signals[signal_name] = window_signal['value'].values
            
            # Only include windows with at least 2 valid signals
            if valid_signals >= 2:
                # Determine label based on event overlap
                label = self.get_window_label(current_time, window_end, events)
                
                window_data = {
                    'participant': participant_name,
                    'window_id': window_id,
                    'start_time': current_time,
                    'end_time': window_end,
                    'label': label,
                    'signals': window_signals
                }
                
                windows.append(window_data)
                window_id += 1
            
            current_time += pd.Timedelta(seconds=self.step_size)
        
        return windows
    
    def get_window_label(self, window_start, window_end, events):
        """Determine window label based on event overlap (>50% rule)"""
        if events.empty:
            return 'Normal'
        
        window_duration = (window_end - window_start).total_seconds()
        best_label = 'Normal'
        max_overlap_ratio = 0
        
        for _, event in events.iterrows():
            # Calculate overlap
            overlap_start = max(window_start, event['start'])
            overlap_end = min(window_end, event['end'])
            
            if overlap_start < overlap_end:
                overlap_duration = (overlap_end - overlap_start).total_seconds()
                overlap_ratio = overlap_duration / window_duration
                
                # If overlap > 50% and this is the largest overlap, use this label
                if overlap_ratio > 0.5 and overlap_ratio > max_overlap_ratio:
                    max_overlap_ratio = overlap_ratio
                    best_label = event['event']
        
        return best_label
    
    def save_dataset(self, all_windows, output_dir):
        """Save dataset in pickle format with comprehensive metadata"""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if not all_windows:
            print("✗ No windows to save")
            return
        
        # Save complete dataset as pickle
        dataset_path = os.path.join(output_dir, 'dataset.pkl')
        with open(dataset_path, 'wb') as f:
            pickle.dump(all_windows, f)
        
        # Calculate comprehensive statistics
        participants = list(set(w['participant'] for w in all_windows))
        label_counts = Counter(w['label'] for w in all_windows)
        
        # Signal statistics
        signal_stats = {}
        for window in all_windows:
            for signal_name, signal_data in window['signals'].items():
                if signal_name not in signal_stats:
                    signal_stats[signal_name] = {'lengths': [], 'means': [], 'stds': []}
                signal_stats[signal_name]['lengths'].append(len(signal_data))
                signal_stats[signal_name]['means'].append(np.mean(signal_data))
                signal_stats[signal_name]['stds'].append(np.std(signal_data))
        
        # Save metadata
        metadata = {
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step_size,
            'total_windows': len(all_windows),
            'participants': participants,
            'label_distribution': dict(label_counts),
            'signal_statistics': {
                signal: {
                    'avg_length': np.mean(stats['lengths']),
                    'avg_mean': np.mean(stats['means']),
                    'avg_std': np.mean(stats['stds'])
                }
                for signal, stats in signal_stats.items()
            },
            'validation_thresholds': {
                'min_signal_points': self.min_signal_points,
                'max_event_duration': self.max_event_duration,
                'min_event_duration': self.min_event_duration
            }
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Dataset saved to {output_dir}")
        print(f"  Total windows: {len(all_windows)}")
        print(f"  Label distribution: {dict(label_counts)}")
        
        # Save summary CSV for easy inspection
        summary_data = []
        for window in all_windows:
            summary_data.append({
                'participant': window['participant'],
                'window_id': window['window_id'],
                'start_time': window['start_time'],
                'end_time': window['end_time'],
                'label': window['label'],
                'num_signals': len(window['signals']),
                'signal_names': list(window['signals'].keys())
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, 'dataset_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Dataset summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Create labeled dataset from cleaned sleep study data')
    parser.add_argument('-in_dir', type=str, required=True, 
                       help='Input directory containing original participant data folders')
    parser.add_argument('-out_dir', type=str, default='Dataset', 
                       help='Output directory for dataset (default: Dataset)')
    parser.add_argument('-cleaned_dir', type=str, default='Cleaned_Data',
                       help='Directory containing cleaned signals (default: Cleaned_Data)')
    parser.add_argument('-window_size', type=int, default=30,
                       help='Window size in seconds (default: 30)')
    parser.add_argument('-overlap', type=float, default=0.5,
                       help='Overlap ratio (default: 0.5 for 50%)')
    parser.add_argument('-validate', action='store_true',
                       help='Run additional validation checks')
    
    args = parser.parse_args()
    
    # Initialize dataset creator
    creator = DatasetCreator(window_size=args.window_size, overlap=args.overlap)
    
    # Check directories
    if not os.path.exists(args.in_dir):
        print(f"✗ Error: Input directory '{args.in_dir}' not found")
        return
    
    if not os.path.exists(args.cleaned_dir):
        print(f"✗ Error: Cleaned data directory '{args.cleaned_dir}' not found")
        print("Please run the data cleaning step first")
        return
    
    # Find participants
    participants = []
    for item in os.listdir(args.in_dir):
        if os.path.isdir(os.path.join(args.in_dir, item)):
            participants.append(item)
    
    if not participants:
        print(f"✗ No participant folders found in {args.in_dir}")
        return
    
    print(f"Found {len(participants)} participants: {participants}")
    
    all_windows = []
    participant_stats = {}
    
    for participant in participants:
        print(f"\n{'='*60}")
        print(f"Processing participant: {participant}")
        print(f"{'='*60}")
        
        # Load cleaned signals
        cleaned_folder = os.path.join(args.cleaned_dir, participant)
        if not os.path.exists(cleaned_folder):
            print(f"✗ Cleaned data folder not found: {cleaned_folder}")
            continue
        
        signals = creator.load_cleaned_signals(cleaned_folder)
        
        if not signals:
            print(f"✗ No valid cleaned signals loaded for {participant}")
            continue
        
        # Load events from original data folder
        original_folder = os.path.join(args.in_dir, participant)
        events = creator.load_events_file(original_folder)
        
        # Create windows
        windows = creator.create_windows(signals, events, participant)
        
        if windows:
            all_windows.extend(windows)
            
            # Calculate statistics
            labels = [w['label'] for w in windows]
            participant_stats[participant] = {
                'windows': len(windows),
                'labels': Counter(labels),
                'signal_coverage': {signal: len(windows) for signal in signals.keys()}
            }
            
            print(f"✓ Created {len(windows)} windows for {participant}")
            print(f"  Label distribution: {dict(Counter(labels))}")
            
            # Show class balance
            total_windows = len(windows)
            for label, count in Counter(labels).items():
                percentage = (count / total_windows) * 100
                print(f"    {label}: {count} ({percentage:.1f}%)")
        else:
            print(f"✗ No windows created for {participant}")
    
    # Save dataset
    if all_windows:
        creator.save_dataset(all_windows, args.out_dir)
        
        # Print final statistics
        print(f"\n{'='*60}")
        print("DATASET CREATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total participants processed: {len(participant_stats)}")
        print(f"Total windows created: {len(all_windows)}")
        
        all_labels = [w['label'] for w in all_windows]
        label_counts = Counter(all_labels)
        print(f"\nOverall label distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(all_labels)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Check for class imbalance
        print(f"\nClass Balance Analysis:")
        normal_count = label_counts.get('Normal', 0)
        abnormal_count = len(all_labels) - normal_count
        if abnormal_count > 0:
            imbalance_ratio = normal_count / abnormal_count
            print(f"  Normal to Abnormal ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 10:
                print("  ⚠️  High class imbalance detected - consider using appropriate sampling techniques")
        
        print(f"\nPer-participant statistics:")
        for participant, stats in participant_stats.items():
            print(f"  {participant}: {stats['windows']} windows")
            for label, count in stats['labels'].items():
                percentage = (count / stats['windows']) * 100
                print(f"    {label}: {count} ({percentage:.1f}%)")
        
        print(f"\nDataset saved to: {args.out_dir}")
        print("✓ Dataset creation completed successfully!")
        
        # Quality recommendations
        print(f"\n{'='*60}")
        print("QUALITY RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if len(all_windows) < 1000:
            print("⚠️  Small dataset - consider data augmentation techniques")
        
        if label_counts.get('Normal', 0) / len(all_labels) > 0.9:
            print("⚠️  Very high proportion of Normal samples - verify event detection")
        
        if any(count < 50 for count in label_counts.values()):
            print("⚠️  Some classes have very few samples - may affect model performance")
        
        print("✓ Ready for model training!")
    
    else:
        print("✗ No windows were created. Please check your data files.")

if __name__ == "__main__":
    main()