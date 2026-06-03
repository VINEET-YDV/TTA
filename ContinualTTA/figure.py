import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Threshold Sensitivity Results
# --------------------------------------------------
thresholds = [0.10, 0.06, 0.05, 0.04, 0.03]
accuracy   = [84.0, 84.7, 86.1, 86.4, 86.1]

# --------------------------------------------------
# Sort thresholds in descending order (left → right)
# --------------------------------------------------
pairs = sorted(zip(thresholds, accuracy), reverse=True)

thresholds = [p[0] for p in pairs]
accuracy   = [p[1] for p in pairs]

# --------------------------------------------------
# Figure Setup
# --------------------------------------------------
plt.figure(figsize=(7.2, 4.5))

# Main curve
plt.plot(
    thresholds,
    accuracy,
    marker='o',
    linewidth=3,
    markersize=9
)

# --------------------------------------------------
# Highlight best point
# --------------------------------------------------
best_idx = np.argmax(accuracy)

plt.scatter(
    thresholds[best_idx],
    accuracy[best_idx],
    s=180,
    zorder=5,
    label='Best Operating Point'
)

# --------------------------------------------------
# Annotation
# --------------------------------------------------
plt.annotate(
    r'Optimal $\tau = 0.04$',
    xy=(thresholds[best_idx], accuracy[best_idx]),
    xytext=(0.072, 85.35),
    arrowprops=dict(
        arrowstyle='->',
        lw=2
    ),
    fontsize=12
)

# --------------------------------------------------
# Labels and Title
# --------------------------------------------------
plt.xlabel(
    'JS Divergence Threshold τ',
    fontsize=15
)

plt.ylabel(
    'Mean Accuracy (%)',
    fontsize=15
)

plt.title(
    'Threshold Sensitivity Under Continual Sequential Shift',
    fontsize=18,
    pad=14
)

# --------------------------------------------------
# Grid and ticks
# --------------------------------------------------
plt.grid(
    True,
    linestyle='--',
    alpha=0.35
)

plt.xticks(
    thresholds,
    [f'{t:.2f}' for t in thresholds],
    fontsize=12
)

plt.yticks(fontsize=12)

# Reverse x-axis for interpretation
plt.gca().invert_xaxis()

# --------------------------------------------------
# Y-axis range
# --------------------------------------------------
plt.ylim(83.7, 86.7)

# --------------------------------------------------
# Region labels
# --------------------------------------------------
plt.text(
    0.095,
    83.82,
    'Under-\nadaptation',
    fontsize=11,
    ha='center'
)

plt.text(
    0.048,
    83.82,
    'Balanced\nadaptation',
    fontsize=11,
    ha='center'
)

plt.text(
    0.031,
    83.82,
    'Over-\nadaptation',
    fontsize=11,
    ha='center'
)

# --------------------------------------------------
# Legend
# --------------------------------------------------
plt.legend(
    frameon=False,
    fontsize=12,
    loc='upper left'
)

# --------------------------------------------------
# Clean layout
# --------------------------------------------------
plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------
plt.savefig(
    'threshold_sensitivity_clean.pdf',
    dpi=300,
    bbox_inches='tight'
)

plt.savefig(
    'threshold_sensitivity_clean.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()