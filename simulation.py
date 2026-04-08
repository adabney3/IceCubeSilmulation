import numpy as np

n_strings = 8
doms_per_string = 60
string_positions = [(x, y) for x in np.linspace(-500, 500, n_strings//2)
                             for y in np.linspace(-500, 500, 2)]
dom_depth = np.linspace(-500, 0, doms_per_string)  # meters below surface
dom_positions = [(x, y, z) for (x, y) in string_positions for z in dom_depth]

n_ice = 1.31
c = 3e8
c_ice = c / n_ice
theta_c = np.arccos(1 / n_ice)
lambda_abs = 100  # absorption length
lambda_scat = 25  # scattering length

def generate_neutrino():
    cos_theta = np.random.uniform(-1, 0)  # upward-going (through Earth)
    phi = np.random.uniform(0, 2 * np.pi)
    sin_theta = np.sqrt(1 - cos_theta**2)
    direction = np.array([sin_theta * np.cos(phi),
                          sin_theta * np.sin(phi),
                          cos_theta])
    vertex = np.array([np.random.uniform(-400, 400),
                       np.random.uniform(-400, 400),
                       np.random.uniform(-450, -50)])
    return vertex, direction

def cherenkov_time(vertex, direction, dom_pos, t0=0):
    """
    Closest approach geometry for a muon track.
    Returns expected photon arrival time at dom_pos.
    """
    r = np.array(dom_pos) - vertex
    s = np.dot(r, direction)          # distance along track to closest approach
    d_perp = np.linalg.norm(r - s * direction)  # perpendicular distance
    # Photon travels d_perp/sin(theta_c) from emission point to DOM
    d_photon = d_perp / np.sin(theta_c)
    t_emit = t0 + s / c_ice           # time muon reaches emission point
    t_arrive = t_emit + d_photon / c_ice
    return t_arrive, d_photon

def simulate_event(vertex, direction):
    hits = []
    for dom in dom_positions:
        t_arr, d = cherenkov_time(vertex, direction, dom)
        p_survive = np.exp(-d / lambda_abs) * 0.25
        if np.random.rand() < p_survive:
            t_measured = t_arr + np.random.normal(0, 5e-9)
            hits.append({'dom': dom, 't': t_measured})
    return hits

vertex, direction = generate_neutrino()
hits = simulate_event(vertex, direction)
print(f"Generated event with {len(hits)} hit DOMs")
print(f"True direction: {np.degrees(np.arccos(direction[2])):.1f}° from zenith")

n_events = 10000
results = []

for _ in range(n_events):
    vertex, direction = generate_neutrino()
    hits = simulate_event(vertex, direction)
    zenith = np.degrees(np.arccos(direction[2]))
    results.append({'n_hits': len(hits), 'zenith': zenith})

n_hits_arr = np.array([r['n_hits'] for r in results])
zeniths_arr = np.array([r['zenith'] for r in results])

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(n_hits_arr, bins=range(0, 30), color='steelblue', edgecolor='white')
ax1.set_xlabel('Number of hit DOMs')
ax1.set_ylabel('Events')
ax1.set_title('Hit multiplicity distribution')

ax2.hist(zeniths_arr, bins=30, color='darkorange', edgecolor='white')
ax2.axvline(90, color='red', linestyle='--', label='Horizon')
ax2.set_xlabel('Zenith angle (degrees)')
ax2.set_ylabel('Events')
ax2.set_title('Neutrino direction distribution')
ax2.legend()

plt.tight_layout()
plt.savefig('icecube_distributions.png', dpi=150)
plt.show()