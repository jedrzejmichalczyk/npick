import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('triplexer_response.csv', delimiter=',', names=True)
f = data['freq_ghz']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = ['#E53935', '#43A047', '#1E88E5']
labels = ['Ch A (1.8-2.0)', 'Ch B (2.1-2.3)', 'Ch C (2.4-2.6)']
bands = [(1.8, 2.0), (2.1, 2.3), (2.4, 2.6)]

# Top: S11 (reflection) per channel
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax1.plot(f, data[f'S11_{ch}_dB'], color=col, lw=1.5, label=f'|S11| {lbl}')
    ax1.axvspan(band[0], band[1], alpha=0.08, color=col)

ax1.set_ylabel('|S11| (dB)')
ax1.set_ylim(-40, 2)
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_title('Triplexer: Per-Channel Filter Response')

# Bottom: S21 (transmission) per channel
for i, (col, lbl, band) in enumerate(zip(colors, labels, bands)):
    ch = chr(ord('A') + i)
    ax2.plot(f, data[f'S21_{ch}_dB'], color=col, lw=2, label=f'|S21| {lbl}')
    ax2.axvspan(band[0], band[1], alpha=0.08, color=col)

ax2.set_xlabel('Frequency (GHz)')
ax2.set_ylabel('|S21| (dB)')
ax2.set_ylim(-50, 2)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('triplexer_response.png', dpi=150)
print('Saved: triplexer_response.png')
plt.show()
