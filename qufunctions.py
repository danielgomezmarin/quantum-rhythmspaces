# This script contains all the functions needed to generate the quantum path in the script generate_path_and_animation.py.

"""--- IMPORT LIBRARIES ---"""
import numpy as np
import qiskit as qk
from qiskit import transpile
from qiskit_aer import Aer

""" --- DEFINE FUNCTIONS --- """

# Function that maps the range of the derivatives [-1,1] to  t range [0,1]
def map_to_01(der):
    val = (der + 1) / 2
    val = min (1, max(0, val)) # Clip the value to the range [0,1]
    return val

# Function that computes the angles taking t_x and t_y
def compute_t_to_angles(t_x, t_y):
    # Define boundaries of the angles. negative = right bias, positive = left bias
    positive_angles = [-1.73132785, 0.38438828, 1.03052644] # [theta, phi, alpha]
    negative_angles = [1.09339953, 0.18548388, 0.34239378] # [theta, phi, alpha]
    #dV/dx = -1 -> t = 0 -> angle = negative_angle -> right bias
    #dV/dx = 1 -> t = 1 -> angle = positive_angle -> left bias
    # Angle = (1 - t) * negative_angle + t * positive_angle
    angles_x = [(1 - t_x) * negative_angles[i] + t_x * positive_angles[i] for i in range(3)]
    angles_y = [(1 - t_y) * negative_angles[i] + t_y * positive_angles[i] for i in range(3)]
    return  angles_x, angles_y

# Function that takes the angles and computes the shot result, which is a new bit string
def compute_shot_result(angles_x, angles_y):

    # SUBCIRCUIT 
    n = 4 # number of qubits
    qr = qk.QuantumRegister(n) # qubit register
    qc = qk.QuantumCircuit(qr) # quantum circuit acting on the register


    # Circuit architecture: Coin + C1,1(J) x Increment Gate x C1,1(J)
    # Coin operator
    qc.h(3)
    # C1,1(J) matrix
    [qc.cx(i+1, i) for i in np.arange(0, n-1)]
    [qc.cx(i, i-1) for i in np.arange(n-2, 0, -1)]
    # Increment Gate
    arr = list(np.arange(0, n))
    [qc.mcx(arr[0:i], i) for i in np.arange(n-2, 0, -1)]
    qc.x(0)
    # C1,1(J) matrix
    [qc.cx(i+1, i) for i in np.arange(0, n-1)]
    [qc.cx(i, i-1) for i in np.arange(n-2, 0, -1)]


    # CREATE MAIN CIRCUIT
    N = 8 # number of qubits
    main_circuit = qk.QuantumCircuit(N,N) # second N is for classical bits register
    midpoint = N // 2
    # Apply the subroutine to the first and last halves of the qubits
    for start in [0, midpoint]:
        qubits = list(range(start, start + midpoint))
        main_circuit.compose(qc, qubits=qubits, inplace=True)

    # PREPARATION OF COIN STATES USING THE ANGLES
    qr2 = qk.QuantumRegister(N) # qubit register
    qc2 = qk.QuantumCircuit(qr2) # quantum circuit acting on the register
    qc2.ry(angles_x[0], 3)
    qc2.rz(angles_x[1], 3)
    qc2.rx(angles_x[2], 3)
    qc2.ry(angles_y[0], 7)
    qc2.rz(angles_y[1], 7)
    qc2.rx(angles_y[2], 7)

    # SIMULATION EXPERIMENT
    # Create a new circuit by composing the initial circuit three times
    combined_circuit = qc2.compose(main_circuit).compose(main_circuit).compose(main_circuit)
    # Add measurement operations
    combined_circuit.measure(range(N), range(N))
    # Get the qasm simulator backend for measurement
    qasm_simulator = Aer.get_backend('qasm_simulator')
    # Transpile the circuit for the simulator
    transpiled_circuit = transpile(combined_circuit, qasm_simulator)
    # Run the circuit n_times and collect the statevectors
    vector = list(qasm_simulator.run(transpiled_circuit, shots=1).result().get_counts().keys())[0]
    #print(vector)

    return vector

# Function to convert bit string segments to x and y in the Hilbert space for position values using the specified formula
def bit_string_to_hilbert(bit_str):

    x_hilbert = (4 * int(bit_str[5]) + 2 * int(bit_str[6]) + int(bit_str[7]))
    y_hilbert = (4 * int(bit_str[1]) + 2 * int(bit_str[2]) + int(bit_str[3]))

    if x_hilbert > 4:
        x_hilbert = x_hilbert - 8
    if y_hilbert > 4:
        y_hilbert = y_hilbert - 8

    return x_hilbert, y_hilbert

# Function to convert x and y values in the Hilbert space to x and y values in the pixel space
def hilbert_to_pixel(x_hilbert, y_hilbert, ratio_pixel_to_hilbert = 10):
    x_pixel = ratio_pixel_to_hilbert * x_hilbert
    y_pixel = ratio_pixel_to_hilbert * y_hilbert
    return x_pixel, y_pixel

# Functions for the potentials
def linear_potential(x, y, k=0):
    return k * y

def gaussian_potential(x, y, A=100, sigma=100):
    return -A * np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Function to compute max and min derivatives of the potentials. Potential is a function of x and y. 
def compute_max_min_derivatives(potential, minx = -250, maxx = 250, miny = -250, maxy = 250):
    x = np.arange(minx, maxx, 1)
    y = np.arange(miny, maxy, 1)
    X, Y = np.meshgrid(x, y)
    V = potential(X, Y)
    dVdx = np.gradient(V, axis=1)
    dVdy = np.gradient(V, axis=0)
    max_dVdx = np.max(dVdx)
    min_dVdx = np.min(dVdx)
    max_dVdy = np.max(dVdy)
    min_dVdy = np.min(dVdy)
    return max_dVdx, min_dVdx, max_dVdy, min_dVdy

# Function that stores the path in a txt file
def path2txt(new_path, filename):
    with open(filename, 'w') as f:
        for item in new_path:
            f.write("%s\n" % str(item))

# Function that reads the path from a txt file
def txt2path(filename):
    new_path = []
    with open(filename, 'r') as f:
        for line in f:
            # each line is "(x, y)", where x and y are integers 
            values = line.split(",")
            x = int(float(values[0][1:]))
            y = int(float(values[1][:-2]))
            x_norm = (float(x) + 250.0) / 500.0
            y_norm = (float(y) + 250.0) / 500.0
            new_path.append((x_norm, y_norm))
    return new_path

    
