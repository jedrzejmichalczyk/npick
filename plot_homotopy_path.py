"""Plot the homotopy continuation path for the multiplexer coupled solve."""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('continuation_path.csv', delimiter=',', names=True)

lam = data['lambda']
h = data['h']
res = data['residual']
newton = data['newton_iters']
dx_norm = data['dx_norm']
x_norm = data['x_norm']

# Extract x components (12 complex variables = 24 real columns)
n_vars = 12
x_re = np.column_stack([data[f'x{i}_re'] for i in range(n_vars)])
x_im = np.column_stack([data[f'x{i}_im'] for i in range(n_vars)])

# Channel info: 3 channels, 4 vars each
ch_names = ['Ch A (1.8-2.0)', 'Ch B (2.1-2.3)', 'Ch C (2.4-2.6)']
ch_colors = ['#E53935', '#43A047', '#1E88E5']
ch_offsets = [0, 4, 8]
ch_nvars = [4, 4, 4]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# --- Top left: x trajectories (real parts) ---
ax = axes[0, 0]
for ch in range(3):
    off = ch_offsets[ch]
    for v in range(ch_nvars[ch]):
        style = ['-', '--', '-.', ':'][v]
        ax.plot(lam, x_re[:, off + v], linestyle=style, color=ch_colors[ch],
                lw=1.2, label=f'{ch_names[ch]} p{v}' if v == 0 else None)
ax.set_ylabel('Re(x)')
ax.set_title('Polynomial coefficients (real part)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Top right: x trajectories (imaginary parts) ---
ax = axes[0, 1]
for ch in range(3):
    off = ch_offsets[ch]
    for v in range(ch_nvars[ch]):
        style = ['-', '--', '-.', ':'][v]
        ax.plot(lam, x_im[:, off + v], linestyle=style, color=ch_colors[ch],
                lw=1.2, label=f'{ch_names[ch]} p{v}' if v == 0 else None)
ax.set_ylabel('Im(x)')
ax.set_title('Polynomial coefficients (imaginary part)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Mid left: residual and predictor residual ---
ax = axes[1, 0]
ax.semilogy(lam, res, 'k-', lw=1.5, label='Corrector residual')
pred_res = data['predictor_res']
mask = pred_res > 0
ax.semilogy(lam[mask], pred_res[mask], 'r--', lw=1, alpha=0.7, label='Predictor residual')
ax.set_ylabel('||F(x, λ)||')
ax.set_title('Residual along path')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Mid right: step size h ---
ax = axes[1, 1]
mask = h > 0
ax.semilogy(lam[mask], h[mask], 'b-', lw=1.5)
ax.set_ylabel('h (step size)')
ax.set_title('Adaptive step size')
ax.grid(True, alpha=0.3)

# --- Bottom left: Newton iterations per step ---
ax = axes[2, 0]
ax.bar(lam, newton, width=np.diff(np.concatenate([lam, [1.02]])) * 0.8,
       color='steelblue', alpha=0.7, align='edge')
ax.set_xlabel('λ')
ax.set_ylabel('Newton iterations')
ax.set_title('Corrector effort')
ax.grid(True, alpha=0.3)

# --- Bottom right: ||dx|| per step ---
ax = axes[2, 1]
mask = dx_norm > 0
ax.semilogy(lam[mask], dx_norm[mask], 'g-', lw=1.5)
ax.set_xlabel('λ')
ax.set_ylabel('||Δx||')
ax.set_title('Step norm in x-space')
ax.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlim(-0.02, 1.02)

plt.suptitle('Multiplexer Homotopy Continuation Path (λ: 0 → 1)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('homotopy_path.png', dpi=150, bbox_inches='tight')
print('Saved: homotopy_path.png')
plt.show()
