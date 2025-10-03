#################################### LIBRARY IMPORTS ##########################################
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from scipy.stats import norm
import panel as pn
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'WebAgg', etc.

#################################### CONSTANTS ##########################################
# Max harmonic we want the calculation to go up to: 1 = dipole, 2 = quadrupole, 3 = sextupole, etc.
n_max = 10

# Current magnitude
I0 = 16000

# Vacuum permeability
mu0 = 4 * np.pi * 1e-7

# Reference Radius (Good Field Region): -> No calculations are done here yet, this is just for visualization
ref_rad = 1.5

# Radius at which points will start to plot in the first layer:
r0 = 2

# Radial spacing between conductors
dr = 0.2 

# Number of "wires" in each layer's cable
n_radial = [5, 6, 2]
# A list of the numbers of cables in each layer (per quadrant):
num_cond_quad = [10, 5, 6] 

# Aperture radius:
inner_radius = r0 - dr/2

# First layer radius:
# Controlled by amount of condutors present in block
outer_radius = (r0 + (n_radial[0] - 1) * dr) + dr/2

# Total number of conductors per layer:
num_conductors = 4*num_cond_quad

###################################### FUNCTIONS ########################################
# Functions used for all the calculations:

# Uses the non-ideal B-field equation to calculate the field everywhere in the X, Y space due to a conductor at (x_a, y_a)
def B_harm(x, y, x_a, y_a, I):
    # Position of the conductor
    a = np.sqrt(x_a**2 + y_a**2)
    # Position of field measurement
    r = np.sqrt(x**2 + y**2)

    # Angle between a and y=0
    phi = np.arctan2(y_a, x_a)
    # Angle between r and y=0
    theta = np.arctan2(y, x)

    # Angle between r and a
    ang = phi - theta

    # Replace r=0 with r=1e-20 to avoid divsion by zero
    r = np.where(r == 0, 1e-10, r)

    # Now we compute the field for both cases, adding up harmonics until the n_max harmonic is calculated and added

    # Case for r < a:
    B_r_in = ((mu0 * I) / (2 * np.pi * a)) * np.sum([(r / a)**(n - 1) * np.sin(n * ang) for n in range(1, n_max+1)], axis=0)
    B_theta_in = (-1)*((mu0 * I) / (2 * np.pi * a)) * np.sum([(r / a)**(n - 1) * np.cos(n * ang) for n in range(1, n_max+1)], axis=0)
    
    # Case for r > a:
    B_r_out = ((mu0 * I) / (2 * np.pi * a)) * np.sum([(a / r)**(n + 1) * np.sin(n * ang) for n in range(0, n_max+1)], axis=0)
    B_theta_out = ((mu0 * I) / (2 * np.pi * a)) * np.sum([(a / r)**(n + 1) * np.cos(n * ang) for n in range(0, n_max+1)], axis=0)

    # Use np.where to select the correct field values to return
    # np.where(cond., x, y) -> # If true, takes x; if false, takes y
    B_r = np.where(r > a, B_r_out, B_r_in)
    B_theta = np.where(r > a, B_theta_out, B_theta_in)

    # Convert to cartesian coordinates (notably using the angle theta, not ang)
    B_x = B_r * np.cos(theta) - B_theta * np.sin(theta)
    B_y = B_r * np.sin(theta) + B_theta * np.cos(theta)

    return B_x, B_y

# Makes a list of all the angular values following a cos theta distribution for a given number of conductors
def theta_list_maker(u):
    # Inverse CDF for cos theta distribution: arcsin(2u - 1) maps uniform u to theta
    theta_right = np.arcsin(u) # Density function for 1st quadrant, from [0, pi]
    theta_left = np.pi - theta_right # Reflect to left side for symmetry
    theta_bot_left = (-1)*theta_left # To reflect polor coords: (r, theta) -> (r, -theta)
    theta_bot_right = (-1)*theta_right
    
    theta_list = np.sort(np.concatenate([theta_right, theta_left, theta_bot_left, theta_bot_right])) # Combines values for each quandrant into one list
    return theta_list

# Given an angular distribution and a list of radial values, it assigns an (x, y) value to the conductor and a current depending on the hemisphere
def wire_params(theta_list, radii_list):
    wire_params = []
    for theta in theta_list:
        for r in radii_list:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
    
            # Determine current direction by quadrant
            if x > 0:
                current = (-1)*I0  # Q1 & Q4 (right side)
            elif x < 0:
                current = I0   # Q2 & Q3 (left side)
            else:
                current = 0  # Edge case
            wire_params.append((x, y, current))
    return wire_params

###################################### CALCULATIONS ########################################

# Uniformly spaced cumulative values (excluding 0 and 1 to avoid endpoints)

# Creates a master list of the sample region, which is based on the number of conductors:
u_list = []
for i in range(len(num_cond_quad)):
    u_list.append(np.linspace(0, 1, num_cond_quad[i] + 2)[1:-1])

# Creates a master list of the theta distributions for each layer, which is based on the sampling region we just determined:
parent_theta_list = []
for i in range(len(u_list)):
    parent_theta_list.append(theta_list_maker(u_list[i]))

# Calculates the radial position of each wire in each layer
# Determines the radial position of each layer group of conductors in each layer
# Recall: n_radial = [5, 6, 2] -> Number of "wires" in each layer section (cable)
layer_radii = []
for j in range(len(n_radial)):
    if j == 0: # First layer radii values
        layer_radii.append( [r0 + i * dr for i in range(n_radial[0])] )
    else: # All other layer radii values, adds to the top of the last layer
        layer_radii.append( [(layer_radii[j-1][-1] + dr + i * dr) for i in range(n_radial[j])] ) # BROKEN, need to start at the previous layer rad)

# Maps out the wires and how they should be distributed
# Current stays constant in magntitude for each conductor

# Creates a parent list of the wire params list, this will allow us to plot easier:
total_wire_params = []
for i in range(len(layer_radii)):
    total_wire_params.append(wire_params(parent_theta_list[i], layer_radii[i]))

# bounds = [ x min & max, y min & max ]
# Used for both field calculation and plotting to minimize computation
bounds = [ [-4.8, 4.8], 
           [-4.8, 4.8] 
         ]

# Automatically set bounds to allow viewing of all layers

# Grid setup
# The field is always calculated everywhere in this region
x = np.linspace(bounds[0][0], bounds[0][1], 250)
y = np.linspace(bounds[1][0], bounds[1][1], 250)
X, Y = np.meshgrid(x, y)

Bx_total = np.zeros_like(X)
By_total = np.zeros_like(Y)

# Calculate field due to each wire configuration given by wire params parent list
for params in total_wire_params:
    for x0, y0, I in params:
        Bx, By = B_harm(X, Y, x0, y0, I)
        Bx_total += Bx
        By_total += By

# Total B-field magnitudes are computed
B_mag = np.sqrt(Bx_total**2 + By_total**2)

###################################### PLOTTING ########################################

plt.figure(figsize=(8, 8))

# Stream plot of magnetic field
# Only shows 95th percentile as max to make line magnitudes easier to see
#norm = mcolors.Normalize(vmin=B_mag.min(), vmax=np.percentile(B_mag, 95)) 
strm = plt.streamplot(X, Y, Bx_total, By_total, color=B_mag, cmap='viridis', density=3, zorder=0)
cbar = plt.colorbar(strm.lines, label='Magnetic Field Magnitude (T)')

# plot the positions of each wire
for params in total_wire_params:
    for x0, y0, I in params:
        color = 'r' if I > 0 else 'b'
        plt.plot(x0, y0, 'o', color=color, zorder=2)

# Defines the angular space to plot the cirles
theta_circle = np.linspace(0, 2*np.pi, 500)

# Plot of the reference radius
x_ref = ref_rad * np.cos(theta_circle)
y_ref = ref_rad * np.sin(theta_circle)
plt.plot(x_ref, y_ref, 'k:', zorder=1)

# Draw inner circle
x_circle = inner_radius * np.cos(theta_circle)
y_circle = inner_radius * np.sin(theta_circle)
plt.plot(x_circle, y_circle, 'k--', zorder=1)

# Draw outer circle to contain all conductors (defined by number of conductors)
x_outer = outer_radius * np.cos(theta_circle)
y_outer = outer_radius * np.sin(theta_circle)
plt.plot(x_outer, y_outer, 'k--', zorder=1)

# Layer radius = last point + dr/2

# Plots the radius of each layer
for i in range(len(layer_radii)):
    if i == 0: # Skip the first layer since that plots earlier
        pass
    else: 
        x_layer = (layer_radii[i][-1]+dr/2) * np.cos(theta_circle) # The [-1] index accesses the last value in the list
        y_layer = (layer_radii[i][-1]+dr/2) * np.sin(theta_circle)
        plt.plot(x_layer, y_layer, 'k--', zorder=1)

# Axes & final layout
plt.axhline(0, color='black', linewidth=1, zorder=1)
plt.axvline(0, color='black', linewidth=1, zorder=1)
plt.gca().set_aspect('equal')
plt.xlim(bounds[0][0], bounds[0][1])
plt.ylim(bounds[1][0], bounds[1][1])
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"$\\cos \\theta$ Field with harmonics up to $n=${n_max}")
plt.show()