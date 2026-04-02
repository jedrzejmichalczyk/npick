"""Plot triplexer channel responses."""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('triplexer_response.csv', delimiter=',', names=True)
f = data['freq_ghz']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = ['#E53935', '#43A047', '#1E88E5']
labels = ['Ch A (1.8-2.0)', 'Ch B (2.1-2.3)', 'Ch C (2.4-2.6)']
bands = [(1.8, 2.0), (2.1, 2.3), (2.4, 2.6)]

# Top: Filter S11 (standalone, no manifold)
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax1.plot(f, data[f'S11_{ch}_dB'], color=col, lw=1.5, label=f'|S11| {lbl}')
    ax1.axvspan(band[0], band[1], alpha=0.08, color=col)

ax1.set_ylabel('|S11| (dB)')
ax1.set_ylim(-30, 2)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Filter S11 (standalone)')

# Bottom: G11 (matched through manifold)
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax2.plot(f, data[f'G11_{ch}_dB'], color=col, lw=2, label=f'|G11| {lbl}')
    ax2.axvspan(band[0], band[1], alpha=0.08, color=col)

ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('|G11| (dB)')
ax2.set_ylim(-30, 2)
ax2.axhline(-16, color='gray', ls='--', lw=0.8, label='RL spec (-16 dB)')
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_title('Matched G11 (through manifold)')

plt.tight_layout()
plt.savefig('triplexer_initial_response.png', dpi=150, bbox_inches='tight')
print('Saved: triplexer_initial_response.png')
plt.show()
