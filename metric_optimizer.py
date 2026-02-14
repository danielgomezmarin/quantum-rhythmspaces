# This script is a metric optimizer for quantum circuit preparation using gradient descent.

""" --- IMPORT LIBRARIES --- """
import numpy as np
import qiskit as qk
from qiskit import transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt

def cost_function(theta, phi, alpha):

    """ --- CREATE OPTIMIZED CIRCUIT --- """

    # SUBCIRCUIT 
    n = 4 # number of qubits
    qr = qk.QuantumRegister(n) # qubit register
    qc = qk.QuantumCircuit(qr) # quantum circuit acting on the register

    # Circuit architecture: Coin + C1,1(J) x Increment Gate x C1,1(J)
    # Coin operator
    qc.h(3)
    # C1,1(J) matrix
    for i in np.arange(0, n-1):
        qc.cx(i+1, i)
    for i in np.arange(n-2, 0, -1):
        qc.cx(i, i-1)
    # Increment Gate
    arr = list(np.arange(0, n))
    for i in np.arange(n-2, 0, -1):
        qc.mcx(arr[0:i], i)
    qc.x(0)
    # C1,1(J) matrix
    for i in np.arange(0, n-1):
        qc.cx(i+1, i)
    for i in np.arange(n-2, 0, -1):
        qc.cx(i, i-1)
    
    combined_circuit = qc.compose(qc).compose(qc)

    # CREATE CIRCUIT OF 8 QUBITS
    n = 4 # number of qubits
    qr = qk.QuantumRegister(n) # qubit register
    qc2 = qk.QuantumCircuit(qr) # quantum circuit acting on the register
    qc2.ry(theta, 3)
    qc2.rz(phi, 3)
    qc2.rx(alpha, 3)
    composed_circuit = qc2.compose(combined_circuit)

    """ --- METRIC OPTIMIZATION PROBLEM --- """
    # Initialize 4 qubits in the |0000> state and apply the qc
    # Adding additional functionality to simulate the circuit and retrieve statevector
    statevector_simulator = Aer.get_backend('statevector_simulator')
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(composed_circuit, statevector_simulator)
    # Execute the circuit on the statevector simulator
    job = statevector_simulator.run(compiled_circuit)
    # Grab results from the job
    result = job.result()
    # Returns the statevector
    statevector = result.get_statevector(compiled_circuit)

    # Sum the amplitudes for the 8 positions (3 position qubits)
    # The statevector indices correspond to binary representations of the qubit states
    # We need to sum over the coin qubit (qubit 3)
    # Initialize an array to hold the summed amplitudes for each position
    position_amplitudes = np.zeros(8, dtype=complex)
    # Loop through the statevector and sum the amplitudes for the 8 positions
    for i, amplitude in enumerate(statevector):
        position = i % 8  # Get the position by ignoring the coin qubit
        position_amplitudes[position] += abs(amplitude)**2

    # Create a for loop through position_amplitude that multiplies a constant per each amplitude and accumulates the sum. This is going to be a metric
    # that we are going to optimize
    # Initialize the metric to 0
    metric = 0
    for i in range(8):
        x = i
        if i>4:
            x-=8
        metric -= position_amplitudes[i] * x

    return np.real(metric)

def plot_probabilities(theta, phi, alpha):
    """ --- CREATE OPTIMIZED CIRCUIT --- """

    # SUBCIRCUIT 
    n = 4 # number of qubits
    qr = qk.QuantumRegister(n) # qubit register
    qc = qk.QuantumCircuit(qr) # quantum circuit acting on the register

    # Circuit architecture: Coin + C1,1(J) x Increment Gate x C1,1(J)
    # Coin operator
    qc.h(3)
    # C1,1(J) matrix
    for i in np.arange(0, n-1):
        qc.cx(i+1, i)
    for i in np.arange(n-2, 0, -1):
        qc.cx(i, i-1)
    # Increment Gate
    arr = list(np.arange(0, n))
    for i in np.arange(n-2, 0, -1):
        qc.mcx(arr[0:i], i)
    qc.x(0)
    # C1,1(J) matrix
    for i in np.arange(0, n-1):
        qc.cx(i+1, i)
    for i in np.arange(n-2, 0, -1):
        qc.cx(i, i-1)
    
    combined_circuit = qc.compose(qc).compose(qc)

    # CREATE CIRCUIT OF 8 QUBITS
    n = 4 # number of qubits
    qr = qk.QuantumRegister(n) # qubit register
    qc2 = qk.QuantumCircuit(qr) # quantum circuit acting on the register
    qc2.ry(theta, 3)
    qc2.rz(phi, 3)
    qc2.rx(alpha, 3)
    composed_circuit = qc2.compose(combined_circuit)

    """ --- METRIC OPTIMIZATION PROBLEM --- """
    # Initialize 4 qubits in the |0000> state and apply the qc
    # Adding additional functionality to simulate the circuit and retrieve statevector
    statevector_simulator = Aer.get_backend('statevector_simulator')
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(composed_circuit, statevector_simulator)
    # Execute the circuit on the statevector simulator
    job = statevector_simulator.run(compiled_circuit)
    # Grab results from the job
    result = job.result()
    # Returns the statevector
    statevector = result.get_statevector(compiled_circuit)

    # Sum the amplitudes for the 8 positions (3 position qubits)
    # The statevector indices correspond to binary representations of the qubit states
    # We need to sum over the coin qubit (qubit 3)
    # Initialize an array to hold the summed amplitudes for each position
    position_amplitudes = np.zeros(8, dtype=complex)
    # Loop through the statevector and sum the amplitudes for the 8 positions
    for i, amplitude in enumerate(statevector):
        position = i % 8  # Get the position by ignoring the coin qubit
        position_amplitudes[position] += abs(amplitude)**2

    xs = np.arange(8)
    for i in range(8):
        x = i
        if i>4:
            x-=8
        xs[i] = x
    print(xs, position_amplitudes)
    plt.bar(xs, position_amplitudes)
    plt.show()
    

initial_theta = 0.01
initial_phi = 0.3
initial_alpha = 0.5

def gradient_descent_step(theta, phi, alpha, learning_rate):
    # Calculate the cost function for the current values of theta, phi, and alpha
    cost = cost_function(theta, phi, alpha)
    # Calculate the partial derivative of the cost function with respect to theta
    dcost_dtheta = (cost_function(theta + 0.01, phi, alpha) - cost_function(theta - 0.01, phi, alpha)) / 0.02
    # Calculate the partial derivative of the cost function with respect to phi
    dcost_dphi = (cost_function(theta, phi + 0.01, alpha) - cost_function(theta, phi - 0.01, alpha)) / 0.02
    # Calculate the partial derivative of the cost function with respect to alpha
    dcost_dalpha = (cost_function(theta, phi, alpha + 0.01) - cost_function(theta, phi, alpha - 0.01)) / 0.02
    # Update the values of theta, phi, and alpha using the gradient descent update rule
    theta = theta - learning_rate * dcost_dtheta
    phi = phi - learning_rate * dcost_dphi
    alpha = alpha - learning_rate * dcost_dalpha
    return theta, phi, alpha

learning_rate = 0.015 # learning rate tha balances between a not too fast convergence but also advancing enough

def gradient_descent(theta, phi, alpha, learning_rate, steps):
    costs = []
    for i in range(steps):
        theta, phi, alpha = gradient_descent_step(theta, phi, alpha, learning_rate)
        cost = cost_function(theta, phi, alpha)
        costs.append(cost)
    plt.plot(costs)
    plt.show()
    return theta, phi, alpha

theta, phi, alpha = gradient_descent(initial_theta, initial_phi, initial_alpha, learning_rate, 200)

# plot the before and then the after
plot_probabilities(initial_theta, initial_phi, initial_alpha)
plot_probabilities(theta, phi, alpha)
print(theta, phi, alpha)

# Save the three angles as an array
negative_angles = np.array([theta, phi, alpha])
print(negative_angles)


