import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('matching_results.csv', delimiter=',', names=True)
f = data['freq']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(f, data['L11_dB'], 'b--', lw=1.5, label=r'$|L_{11}|$ (load)')
ax.plot(f, data['S11_dB'], 'g-', lw=1.5, label=r'$|S_{11}|$ (filter alone)')
ax.plot(f, data['G11_dB'], 'r-', lw=2.0, label=r'$|G_{11}|$ (matched)')
ax.plot(f, data['S21_dB'], 'c-', lw=1.0, alpha=0.6, label=r'$|S_{21}|$ (transmission)')
ax.axvspan(-1, 1, alpha=0.07, color='yellow', label='Passband')
ax.set_xlim(-3, 3)
ax.set_ylim(-50, 5)
ax.set_xlabel('Normalized frequency')
ax.set_ylabel('dB')
ax.set_title('Impedance Matching: Equiripple Newton (Order 8, TZ={2,3})')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('matching_results.png', dpi=150)
plt.show()
