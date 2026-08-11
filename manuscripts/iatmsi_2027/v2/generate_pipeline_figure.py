"""Generate study design / pipeline overview figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(5, 7.7, 'Study Design', fontsize=10, fontweight='bold', ha='center')

# Coswara dataset box
ax.add_patch(FancyBboxPatch((0.5, 6.2), 3, 1.2, boxstyle="round,pad=0.1",
    facecolor='#e8f4f8', edgecolor='#2166ac', linewidth=1.2))
ax.text(2, 7.0, 'Coswara Dataset', fontsize=8, fontweight='bold', ha='center', color='#2166ac')
ax.text(2, 6.6, 'N = 2,088 participants', fontsize=7, ha='center')
ax.text(2, 6.35, 'Cough, Breathing, Speech', fontsize=7, ha='center')

# Arrow down
ax.annotate('', xy=(2, 5.8), xytext=(2, 6.2),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Preprocessing box
ax.add_patch(FancyBboxPatch((0.5, 4.6), 3, 1.2, boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='gray', linewidth=1))
ax.text(2, 5.4, 'Preprocessing', fontsize=8, fontweight='bold', ha='center')
ax.text(2, 5.05, '16 kHz, silence trim', fontsize=6.5, ha='center')
ax.text(2, 4.8, 'Quality screening', fontsize=6.5, ha='center')

# Arrow to feature extraction
ax.annotate('', xy=(2, 4.2), xytext=(2, 4.6),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Feature extraction box
ax.add_patch(FancyBboxPatch((0.5, 3.0), 3, 1.2, boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='gray', linewidth=1))
ax.text(2, 3.8, 'Feature Extraction', fontsize=8, fontweight='bold', ha='center')
ax.text(2, 3.45, 'ComParE + IS10', fontsize=6.5, ha='center')
ax.text(2, 3.2, '10,140 descriptors', fontsize=6.5, ha='center')

# Arrow to ranking
ax.annotate('', xy=(2, 2.6), xytext=(2, 3.0),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Feature ranking box
ax.add_patch(FancyBboxPatch((0.5, 1.4), 3, 1.2, boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='gray', linewidth=1))
ax.text(2, 2.2, 'Feature Ranking', fontsize=8, fontweight='bold', ha='center')
ax.text(2, 1.85, 'LightGBM gain (train only)', fontsize=6.5, ha='center')
ax.text(2, 1.6, 'Top 800 features frozen', fontsize=6.5, ha='center')

# Arrow to model
ax.annotate('', xy=(2, 1.0), xytext=(2, 1.4),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Model box
ax.add_patch(FancyBboxPatch((0.5, 0.0), 3, 1.0, boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='gray', linewidth=1))
ax.text(2, 0.7, 'LightGBM + Fusion', fontsize=8, fontweight='bold', ha='center')
ax.text(2, 0.35, 'Validation-only selection', fontsize=6.5, ha='center')

# Right side: Evaluation protocols
ax.add_patch(FancyBboxPatch((5.5, 5.5), 4, 2.2, boxstyle="round,pad=0.1",
    facecolor='#fff5f0', edgecolor='#d6604d', linewidth=1.2))
ax.text(7.5, 7.4, 'Evaluation Protocols', fontsize=8, fontweight='bold', ha='center', color='#d6604d')

protocols = [
    ('L1: Participant-Disjoint', '0.897', '#2166ac'),
    ('L2: Time-Stratified', '0.849', '#4393c3'),
    ('L3: Temporal (Early→Late)', '0.698', '#f4a582'),
    ('L4: Cross-Dataset', '0.538', '#d6604d'),
]

for i, (name, val, color) in enumerate(protocols):
    y = 6.9 - i * 0.4
    ax.text(5.7, y, f'{name}', fontsize=7, color=color, fontweight='bold')
    ax.text(9.3, y, val, fontsize=7, color=color, fontweight='bold', ha='right')

# Arrow from model to evaluation
ax.annotate('', xy=(5.5, 6.6), xytext=(3.5, 0.5),
    arrowprops=dict(arrowstyle='->', color='#d6604d', lw=1.2,
                   connectionstyle='arc3,rad=0.3'))

# Key finding box
ax.add_patch(FancyBboxPatch((5.5, 0.0), 4, 1.5, boxstyle="round,pad=0.1",
    facecolor='#f0f0f0', edgecolor='black', linewidth=1))
ax.text(7.5, 1.2, 'Key Finding', fontsize=8, fontweight='bold', ha='center')
ax.text(7.5, 0.85, 'Protocol determines', fontsize=7, ha='center')
ax.text(7.5, 0.55, 'performance more than', fontsize=7, ha='center')
ax.text(7.5, 0.25, 'model architecture', fontsize=7, ha='center', fontweight='bold')

# Arrow from evaluation to key finding
ax.annotate('', xy=(7.5, 1.5), xytext=(7.5, 5.5),
    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

# Frozen pipeline annotation
ax.text(4.2, 3.5, 'Frozen\nPipeline', fontsize=7, ha='center',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', linewidth=0.8),
    rotation=90)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig0_study_design.pdf'), format='pdf')
fig.savefig(os.path.join(OUT_DIR, 'fig0_study_design.png'), format='png')
plt.close(fig)
print("Saved fig0_study_design")
