import numpy as np

# Parameters
mean = np.array([0.02239, 0.1178, 67.5, 0.06, 0.965, 3.064])
cov = np.load('covtrainT0.npy', allow_pickle=True)                                    # Filepath
T = 256   # temperature (256 for training, 128 for validation and testing)
N = 20000 # number of cosmologies

print(f"Generating {N} samples with T={T}...")

# Generate samples
d = np.random.multivariate_normal(mean, cov*T, N)
print(f"Generated shape: {d.shape}")

# Apply hard priors from Table I from Yijie's paper (Gaussian Train column)
print("Applying hard priors...")

# Scale for clipping
d_scaled = d.copy()
d_scaled[:, 0] *= 100  # ombh2 -> 100*ombh2
d_scaled[:, 1] *= 10   # omch2 -> 10*omch2
d_scaled[:, 3] *= 10   # tau   -> 10*tau

# Clip to hard priors
d_scaled[:, 0] = np.clip(d_scaled[:, 0], 0, 4)        # 100*ombh2: [0, 4]
d_scaled[:, 1] = np.clip(d_scaled[:, 1], 0, 3)        # 10*omch2: [0, 3]
d_scaled[:, 2] = np.clip(d_scaled[:, 2], 25, 114)     # H0: [25, 114]
d_scaled[:, 3] = np.clip(d_scaled[:, 3], 0.07, 1.5)   # 10*tau: [0.07, 1.5]
d_scaled[:, 4] = np.clip(d_scaled[:, 4], 0.7, 1.3)    # ns: [0.7, 1.3]
d_scaled[:, 5] = np.clip(d_scaled[:, 5], 1.61, 4.5)   # logAs: [1.61, 4.5]

# Unscale back to original units
d[:, 0] = d_scaled[:, 0] / 100  # ombh2
d[:, 1] = d_scaled[:, 1] / 10   # omch2
d[:, 2] = d_scaled[:, 2]        # H0
d[:, 3] = d_scaled[:, 3] / 10   # tau
d[:, 4] = d_scaled[:, 4]        # ns
d[:, 5] = d_scaled[:, 5]        # logAs

print(f"After hard priors: {d.shape}")

# Add dummy nodes (mnu=0.06, w0=-1, wa=0)
print("Adding dummy nodes...")
dummy = np.ones((N, 3))
dummy[:, 0] = 0.06   # mnu
dummy[:, 1] = -1.0   # w0
dummy[:, 2] = 0.0    # wa

# Concatenate
final = np.concatenate([d, dummy], axis=1)

print(f"Final shape: {final.shape}")
print(f"First 3 samples:")
print(final[:3])

# Save
np.save('basetruth_validation_params_20k.npy', final)                                               # Filepath
print(f"\nbasetruth_validation_params_20k.npy")

# Print final ranges
param_names = ['ombh2', 'omch2', 'H0', 'tau', 'ns', 'logAs', 'mnu', 'w0', 'wa']
print("\nFinal parameter ranges:")
for i, name in enumerate(param_names):
    print(f"{name:8s}: [{final[:, i].min():.6f}, {final[:, i].max():.6f}]")