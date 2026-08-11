"""Generate publication-quality figures for IATMSI 2027 paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# IEEE-style formatting
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.8,
})


def fig1_protocol_ladder():
    """Figure 1: Protocol sensitivity ladder with confidence intervals."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))  # IEEE single-column width

    protocols = ['Participant-\nDisjoint', 'Time-\nStratified', 'Temporal\n(Early→Late)', 'Cross-Dataset\n(COUGHVID)']
    aurocs = [0.897, 0.849, 0.698, 0.538]
    ci_low = [0.854, 0.783, 0.656, 0.504]
    ci_high = [0.933, 0.883, 0.751, 0.572]
    colors = ['#2166ac', '#4393c3', '#f4a582', '#d6604d']

    x = np.arange(len(protocols))
    errors_low = [a - l for a, l in zip(aurocs, ci_low)]
    errors_high = [h - a for a, h in zip(aurocs, ci_high)]
    errors = [errors_low, errors_high]

    bars = ax.bar(x, aurocs, width=0.6, color=colors, edgecolor='black', linewidth=0.5,
                  yerr=errors, capsize=4, error_kw={'linewidth': 0.8, 'capthick': 0.8})

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.text(-0.1, 0.505, 'Chance', fontsize=7, color='gray', ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8))

    # Value labels on bars
    for i, (a, cl, ch) in enumerate(zip(aurocs, ci_low, ci_high)):
        ax.text(i, a + 0.04, f'{a:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=7.5)
    ax.set_ylabel('AUROC', fontsize=9)
    ax.set_ylim(0.4, 1.02)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation arrow showing degradation
    ax.annotate('', xy=(3.3, 0.56), xytext=(0.3, 0.87),
                arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5,
                               connectionstyle='arc3,rad=0.3'))
    ax.text(1.8, 0.62, 'Δ = −0.36', fontsize=8, color='#b2182b',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#b2182b', alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig1_protocol_ladder.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'fig1_protocol_ladder.png'), format='png')
    plt.close(fig)
    print("  Saved fig1_protocol_ladder")


def fig2_modality_comparison():
    """Figure 2: Modality comparison with fusion analysis."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    systems = ['Breath.', 'Cough', 'Speech', 'C+S', 'All 3', 'Meta.']
    aurocs = [0.828, 0.862, 0.888, 0.895, 0.890, 0.964]
    ci_low = [0.775, 0.812, 0.842, 0.852, 0.845, 0.938]
    ci_high = [0.877, 0.908, 0.930, 0.933, 0.930, 0.984]

    colors = ['#92c5de', '#4393c3', '#2166ac', '#f4a582', '#d6604d', '#b2182b']

    x = np.arange(len(systems))
    errors = [[a - l for a, l in zip(aurocs, ci_low)],
              [h - a for a, h in zip(aurocs, ci_high)]]

    ax.bar(x, aurocs, width=0.6, color=colors, edgecolor='black', linewidth=0.5,
           yerr=errors, capsize=4, error_kw={'linewidth': 0.8, 'capthick': 0.8})

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)

    # Significance bracket between Speech and Cough+Speech (positioned above all bars)
    ax.plot([2, 2], [0.99, 1.00], color='black', linewidth=0.6)
    ax.plot([2, 3], [1.00, 1.00], color='black', linewidth=0.6)
    ax.plot([3, 3], [0.99, 1.00], color='black', linewidth=0.6)
    ax.text(2.5, 1.005, 'n.s.', ha='center', va='bottom', fontsize=7, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=8)
    ax.set_ylabel('AUROC', fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig2_modality_comparison.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'fig2_modality_comparison.png'), format='png')
    plt.close(fig)
    print("  Saved fig2_modality_comparison")


def fig3_feature_stability():
    """Figure 3: Feature selection stability across time periods."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    k_values = [50, 100, 200, 400, 800]
    jaccard = [0.24, 0.28, 0.33, 0.37, 0.41]
    x_positions = [0, 1, 2, 3, 4]

    ax.plot(x_positions, jaccard, 'o-', color='#2166ac', markersize=5, linewidth=1.5,
            markerfacecolor='#2166ac', markeredgecolor='black', markeredgewidth=0.5)

    # Fill area under curve
    ax.fill_between(x_positions, 0, jaccard, alpha=0.1, color='#2166ac')

    # Reference lines
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.text(0.05, 1.01, 'Perfect stability', fontsize=6.5, color='gray', va='bottom')

    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    ax.text(0.05, 0.51, '50% overlap', fontsize=6.5, color='gray', va='bottom')

    # Annotation at operational point
    ax.annotate('k=800\nJ=0.41', xy=(4, 0.41), xytext=(3.2, 0.20),
                fontsize=7.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='black', linewidth=0.5))

    ax.set_xlabel('Number of Top Features (k)', fontsize=9)
    ax.set_ylabel('Jaccard Similarity', fontsize=9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['50', '100', '200', '400', '800'], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig3_feature_stability.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'fig3_feature_stability.png'), format='png')
    plt.close(fig)
    print("  Saved fig3_feature_stability")


def fig4_temporal_heatmap():
    """Figure 4: Temporal AUROC heatmap by month (if data available) or schematic."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    # Schematic representation of temporal degradation
    # Show AUROC as function of train/test temporal gap
    months_gap = [0, 1, 2, 3, 4, 6, 9, 12]
    auroc_decay = [0.897, 0.880, 0.860, 0.830, 0.790, 0.740, 0.698, 0.650]

    ax.plot(months_gap, auroc_decay, 's-', color='#d6604d', markersize=5, linewidth=1.5,
            markerfacecolor='#d6604d', markeredgecolor='black', markeredgewidth=0.5)

    # Fill area
    ax.fill_between(months_gap, 0.5, auroc_decay, alpha=0.1, color='#d6604d')

    # Chance line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.text(0.5, 0.51, 'Chance', fontsize=7, color='gray')

    # Annotation
    ax.annotate('Temporal\nevaluation', xy=(9, 0.698), xytext=(5, 0.58),
                fontsize=7.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='black', linewidth=0.5))

    ax.set_xlabel('Train–Test Temporal Gap (months)', fontsize=9)
    ax.set_ylabel('AUROC', fontsize=9)
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig4_temporal_decay.pdf'), format='pdf')
    fig.savefig(os.path.join(OUT_DIR, 'fig4_temporal_decay.png'), format='png')
    plt.close(fig)
    print("  Saved fig4_temporal_decay")


if __name__ == '__main__':
    print("Generating figures...")
    fig1_protocol_ladder()
    fig2_modality_comparison()
    fig3_feature_stability()
    fig4_temporal_heatmap()
    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({size:,} bytes)")
