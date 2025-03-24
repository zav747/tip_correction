import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract data from file
file_name = 'polar DU95W180.csv'
data = pd.read_csv(file_name)
polar_alpha = data['Alfa'].values
polar_cl = data['Cl'].values
polar_cd = data['Cd'].values

# define the blade geometry
delta_r_R = 0.01
r_R = np.arange(0.2, 1 + delta_r_R / 2, delta_r_R)

# blade shape
pitch = -2
chord_distribution = 3 * (1 - r_R) + 1
twist_distribution = 14 * (1 - r_R) + pitch

# flow conditions
u_inf = 10
TSR = 10
Radius = 50
Omega = u_inf * TSR / Radius
N_blades = 3

root_R = 0.2
tip_R = 1.0

# solve BEM model
results = np.zeros([len(r_R) - 1, 6])
alpha = np.zeros(len(r_R) - 1)
phi = np.zeros(len(r_R) - 1)

results_without = np.zeros_like(results)
alpha_without = np.zeros_like(alpha)
phi_without = np.zeros_like(phi)

# Loop over blade sections
for i in range(len(r_R) - 1):
    r_mid = (r_R[i] + r_R[i + 1]) / 2
    chord = np.interp(r_mid, r_R, chord_distribution)
    twist = np.interp(r_mid, r_R, twist_distribution)

    # With Prandtl correction
    results[i, :], alpha[i], phi[i] = solve_stream(
        u_inf, r_R[i], r_R[i + 1], root_R, tip_R, Omega, Radius,
        N_blades, chord, twist, polar_alpha, polar_cl, polar_cd, use_prandtl=True)

    # Without Prandtl correction
    results_without[i, :], alpha_without[i], phi_without[i] = solve_stream(
        u_inf, r_R[i], r_R[i + 1], root_R, tip_R, Omega, Radius,
        N_blades, chord, twist, polar_alpha, polar_cl, polar_cd, use_prandtl=False)

phi = phi * 180 / np.pi
phi_without = phi_without * 180 / np.pi

# Plotting function for comparisons
def plot_comparison(x, y1, y2, label1, label2, title, ylabel):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, '--', label=label2)
    plt.title(title)
    plt.xlabel("r/R")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Comparison plots
plot_comparison(r_R[1:], results[:, 0], results_without[:, 0], "a (with Prandtl)", "a (without Prandtl)", "Axial induction factor a", "a")
plot_comparison(r_R[1:], results[:, 1], results_without[:, 1], "a' (with Prandtl)", "a' (without Prandtl)", "Tangential induction factor a'", "a'")
plot_comparison(r_R[1:], alpha, alpha_without, "Alpha (with Prandtl)", "Alpha (without Prandtl)", "Angle of attack", "Alpha [deg]")
plot_comparison(r_R[1:], results[:, 3], results_without[:, 3], "Fn (with Prandtl)", "Fn (without Prandtl)", "Normal force", "Fn [N/m]")
plot_comparison(r_R[1:], results[:, 4], results_without[:, 4], "Ft (with Prandtl)", "Ft (without Prandtl)", "Tangential force", "Ft [N/m]")

# Global performance calculation with Prandtl
rho = 1.225
areas = (r_R[1:]**2 - r_R[:-1]**2) * np.pi * Radius**2
dr = (r_R[1:] - r_R[:-1]) * Radius

CT = np.sum(dr * results[:, 3] * N_blades / (0.5 * u_inf**2 * np.pi * Radius**2))
CP = np.sum(dr * results[:, 4] * results[:, 2] * N_blades * Radius * Omega /
            (0.5 * u_inf**3 * np.pi * Radius**2))
CTorque = np.sum(dr * results[:, 4] * N_blades / (0.5 * u_inf**2 * np.pi * Radius**2))

T = CT * 0.5 * rho * u_inf**2 * (2 * np.pi * Radius**2)
P = CP * 0.5 * rho * u_inf**3 * (2 * np.pi * Radius**2)
Torque = CTorque * 0.5 * rho * u_inf**2 * (2 * np.pi * Radius**2)

# Print results
print("CT:", CT)
print("CP:", CP)
print("CTorque:", CTorque)
print("Total Thrust:", T)
print("Total Power:", P)
print("Total Torque:", Torque)

# Original plots (with Prandtl)
plt.figure(figsize=(12, 6))
plt.title('Angle of attack and inflow angle over span')
plt.plot(r_R[1:], alpha, '-', label=r'$\alpha$')
plt.plot(r_R[1:], phi, '-', label=r'$\phi$')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('DEG')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.title('Induction factors over span')
plt.plot(r_R[1:], results[:, 0], '-', label='Axial induction factor')
plt.plot(r_R[1:], results[:, 1], '-', label='Tangential induction factor')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('-')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.title('Normal and azimuthal loading')
plt.plot(r_R[1:], results[:, 3], '-', label='Normal loading')
plt.plot(r_R[1:], results[:, 4], '-', label='Azimuthal loading')
plt.grid()
plt.xlabel('r/R')
plt.ylabel('N')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.title('Axial and tangential induction')
plt.plot(results[:, 2], results[:, 0], '-', label=r'$a$')
plt.plot(results[:, 2], results[:, 1], '-', label=r'$a^,$')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.title(r'Normal and tangential force, non-dimensionalized by $\frac{1}{2} \rho U_\infty^2 R$')
plt.plot(results[:, 2], results[:, 3]/(0.5*u_inf**2*Radius), '-', label=r'F_norm')
plt.plot(results[:, 2], results[:, 4]/(0.5*u_inf**2*Radius), '-', label=r'F_tan')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.title(r'Circulation distribution, non-dimensionalized by $\frac{\pi U_\infty^2}{\Omega * NBlades } $')
plt.plot(results[:, 2], results[:, 5]/(np.pi*u_inf**2/(N_blades*Omega)), '-', label=r'$\Gamma$')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()
