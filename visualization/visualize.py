import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import pandas as pd
from scipy import signal

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_audio_stats(file_path):
    """Load audio file and compute basic statistics."""
    data, sr = sf.read(str(file_path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    duration = len(data) / sr
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))

    return {
        'duration': duration,
        'rms': rms,
        'peak': peak,
        'sample_rate': sr,
        'samples': len(data)
    }


def analyze_dataset_overview(train_clean_path, train_noisy_path, test_clean_path, test_noisy_path):
    """
    Generate overview statistics and visualizations for the entire dataset.
    """
    paths = {
        'Train Clean': Path(train_clean_path),
        'Train Noisy': Path(train_noisy_path),
        'Test Clean': Path(test_clean_path),
        'Test Noisy': Path(test_noisy_path)
    }

    stats = {}
    for name, path in paths.items():
        files = list(path.glob("*.wav"))
        stats[name] = {
            'count': len(files),
            'files': files
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dataset Overview Statistics', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    categories = list(stats.keys())
    counts = [stats[cat]['count'] for cat in categories]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Files', fontsize=12)
    ax.set_title('File Count Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax = axes[0, 1]
    train_total = stats['Train Clean']['count']
    test_total = stats['Test Clean']['count']
    sizes = [train_total, test_total]
    labels = [f'Train\n({train_total} files)', f'Test\n({test_total} files)']
    colors_pie = ['#3498db', '#2ecc71']
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 12})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Train/Test Split', fontsize=14, fontweight='bold')

    print("Analyzing audio durations...")
    durations_data = []
    for name, path in paths.items():
        for f in tqdm(stats[name]['files'][:50], desc=f"Sampling {name}", leave=False):  # Sample first 50
            try:
                duration = load_audio_stats(f)['duration']
                durations_data.append({'Category': name, 'Duration (s)': duration})
            except Exception as e:
                continue

    df_durations = pd.DataFrame(durations_data)
    ax = axes[1, 0]
    if not df_durations.empty:
        df_durations.boxplot(column='Duration (s)', by='Category', ax=ax)
        ax.set_title('Audio Duration Distribution (Sample)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Duration (seconds)', fontsize=12)
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    ax = axes[1, 1]
    ax.axis('off')

    summary_data = []
    for name in categories:
        count = stats[name]['count']
        if name in df_durations['Category'].values:
            avg_dur = df_durations[df_durations['Category'] == name]['Duration (s)'].mean()
            total_dur = avg_dur * count / 60  # Convert to minutes
        else:
            avg_dur = 0
            total_dur = 0
        summary_data.append([name, count, f'{avg_dur:.1f}s', f'{total_dur:.1f}m'])

    table = ax.table(cellText=summary_data,
                     colLabels=['Category', 'Files', 'Avg Duration', 'Est. Total'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    for i in range(4):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Dataset Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig


def analyze_audio_quality(train_clean_path, train_noisy_path, num_samples=30):
    """
    Analyze and compare audio quality metrics between clean and noisy samples.
    """
    clean_path = Path(train_clean_path)
    noisy_path = Path(train_noisy_path)

    clean_files = list(clean_path.glob("*.wav"))[:num_samples]

    data = []
    print(f"Analyzing audio quality for {num_samples} samples...")
    for clean_file in tqdm(clean_files):
        noisy_file = noisy_path / clean_file.name

        if not noisy_file.exists():
            continue

        try:
            clean_stats = load_audio_stats(clean_file)
            noisy_stats = load_audio_stats(noisy_file)

            data.append({
                'Type': 'Clean',
                'RMS Level': clean_stats['rms'],
                'Peak Level': clean_stats['peak'],
                'Duration': clean_stats['duration']
            })
            data.append({
                'Type': 'Noisy',
                'RMS Level': noisy_stats['rms'],
                'Peak Level': noisy_stats['peak'],
                'Duration': noisy_stats['duration']
            })
        except Exception as e:
            continue

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Audio Quality Comparison: Clean vs Noisy', fontsize=16, fontweight='bold')

    # RMS Level comparison
    ax = axes[0, 0]
    df.boxplot(column='RMS Level', by='Type', ax=ax)
    ax.set_title('RMS Level Distribution', fontsize=14)
    ax.set_ylabel('RMS Level', fontsize=12)
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Peak Level comparison
    ax = axes[0, 1]
    df.boxplot(column='Peak Level', by='Type', ax=ax)
    ax.set_title('Peak Level Distribution', fontsize=14)
    ax.set_ylabel('Peak Level', fontsize=12)
    plt.sca(ax)
    plt.xticks(rotation=0)

    # RMS histogram
    ax = axes[1, 0]
    clean_rms = df[df['Type'] == 'Clean']['RMS Level']
    noisy_rms = df[df['Type'] == 'Noisy']['RMS Level']
    ax.hist([clean_rms, noisy_rms], bins=20, label=['Clean', 'Noisy'],
            color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax.set_xlabel('RMS Level', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('RMS Level Histogram', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # SNR estimation scatter
    ax = axes[1, 1]
    clean_peak = df[df['Type'] == 'Clean']['Peak Level'].values
    noisy_peak = df[df['Type'] == 'Noisy']['Peak Level'].values
    min_len = min(len(clean_peak), len(noisy_peak))

    if min_len > 0:
        snr_estimate = 20 * np.log10(clean_peak[:min_len] / (noisy_peak[:min_len] - clean_peak[:min_len] + 1e-10))
        ax.hist(snr_estimate, bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Estimated SNR (dB)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Estimated SNR Distribution', fontsize=14)
        ax.axvline(np.mean(snr_estimate), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(snr_estimate):.1f} dB')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_waveform_comparison(clean_file, noisy_file, duration=5.0):
    """
    Plot waveform comparison between a clean and noisy audio pair.
    """
    clean_data, sr = sf.read(str(clean_file))
    noisy_data, _ = sf.read(str(noisy_file))

    if clean_data.ndim > 1:
        clean_data = clean_data.mean(axis=1)
    if noisy_data.ndim > 1:
        noisy_data = noisy_data.mean(axis=1)

    num_samples = int(duration * sr)
    clean_segment = clean_data[:num_samples]
    noisy_segment = noisy_data[:num_samples]
    time = np.arange(len(clean_segment)) / sr

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Waveform Analysis: {Path(clean_file).stem}', fontsize=16, fontweight='bold')

    # Clean waveform
    ax = axes[0, 0]
    ax.plot(time, clean_segment, color='#3498db', linewidth=0.5)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Clean Audio Waveform', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Noisy waveform
    ax = axes[0, 1]
    ax.plot(time, noisy_segment, color='#e74c3c', linewidth=0.5)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Noisy Audio Waveform', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Clean spectrogram
    ax = axes[1, 0]
    f, t, Sxx = signal.spectrogram(clean_segment, sr, nperseg=1024)
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Clean Audio Spectrogram', fontsize=14)
    ax.set_ylim(0, 8000)
    plt.colorbar(im, ax=ax, label='Power (dB)')

    # Noisy spectrogram
    ax = axes[1, 1]
    f, t, Sxx = signal.spectrogram(noisy_segment, sr, nperseg=1024)
    im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Noisy Audio Spectrogram', fontsize=14)
    ax.set_ylim(0, 8000)
    plt.colorbar(im, ax=ax, label='Power (dB)')

    # Difference waveform
    ax = axes[2, 0]
    difference = noisy_segment - clean_segment
    ax.plot(time, difference, color='#f39c12', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Noise Component (Noisy - Clean)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, duration)

    # Frequency spectrum comparison
    ax = axes[2, 1]
    clean_fft = np.fft.rfft(clean_segment)
    noisy_fft = np.fft.rfft(noisy_segment)
    freqs = np.fft.rfftfreq(len(clean_segment), 1 / sr)

    ax.plot(freqs, 20 * np.log10(np.abs(clean_fft) + 1e-10),
            label='Clean', color='#3498db', alpha=0.7, linewidth=1.5)
    ax.plot(freqs, 20 * np.log10(np.abs(noisy_fft) + 1e-10),
            label='Noisy', color='#e74c3c', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Magnitude (dB)', fontsize=12)
    ax.set_title('Frequency Spectrum Comparison', fontsize=14)
    ax.set_xlim(0, 8000)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_all_visualizations(train_clean_path, train_noisy_path, test_clean_path, test_noisy_path,
                                output_dir='visualizations'):
    """
    Generate all visualizations and save them to files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("Generating Dataset Visualizations")
    print("=" * 60)

    # 1. Dataset overview
    print("\n1. Generating dataset overview...")
    fig1 = analyze_dataset_overview(train_clean_path, train_noisy_path, test_clean_path, test_noisy_path)
    fig1.savefig(output_path / 'dataset_overview.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path / 'dataset_overview.png'}")

    # 2. Audio quality analysis
    print("\n2. Generating audio quality analysis...")
    fig2 = analyze_audio_quality(train_clean_path, train_noisy_path)
    fig2.savefig(output_path / 'audio_quality.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path / 'audio_quality.png'}")

    # 3. Waveform comparison (random sample)
    print("\n3. Generating waveform comparison...")
    clean_files = list(Path(train_clean_path).glob("*.wav"))
    if clean_files:
        sample_file = np.random.choice(clean_files)
        noisy_file = Path(train_noisy_path) / sample_file.name

        if noisy_file.exists():
            fig3 = plot_waveform_comparison(sample_file, noisy_file)
            fig3.savefig(output_path / 'waveform_comparison.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Saved: {output_path / 'waveform_comparison.png'}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {output_path.absolute()}")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    TRAIN_CLEAN_PATH = '../data/train/clean'
    TRAIN_NOISY_PATH = '../data/train/noisy'
    TEST_CLEAN_PATH = '../data/test/clean'
    TEST_NOISY_PATH = '../data/test/noisy'

    generate_all_visualizations(
        TRAIN_CLEAN_PATH,
        TRAIN_NOISY_PATH,
        TEST_CLEAN_PATH,
        TEST_NOISY_PATH,
        output_dir='../visualizations'
    )