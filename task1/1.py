from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Estimator

# Инициализация симулятора
backend = AerSimulator()

# Гамильтониан элемента C в виде матрицы
Hamiltonian = [
    [-1.06365335, 0, 0, 0.1809312],
    [0, -1.83696799, 0.1809312, 0],
    [0, 0.1809312, -0.24521829, 0],
    [0.1809312, 0, 0, -1.06365335]
]

# Преобразуем гамильтониан в операторы Паули с использованием SparsePauliOp
H_op = SparsePauliOp.from_operator(Hamiltonian)

# Определение вариационного анзаца
ansatz = EfficientSU2(num_qubits=2,
                      entanglement='linear',
                      reps=5,
                      skip_final_rotation_layer=True)

# Генерация оптимизированного анзаца
ansatz_opt = ansatz

# Визуализация схемы анзаца как QuantumCircuit
ansatz_circuit = ansatz_opt.to_instruction()

# Создаем и рисуем схему
circuit = QuantumCircuit(2)
circuit.append(ansatz_circuit, [0, 1])
circuit.measure_all()  # Добавляем измерения для всех кубитов

# Рисуем схему
circuit.draw('mpl', filename='vqe_ansatz_circuit.png')

# Определение оптимизатора (COBYLA)
optimizer = COBYLA(maxiter=1500)

# Инициализация Estimator для вычислений
estimator = Estimator(
    run_options={"shots": 2048, "seed": 28},
)

# Инициализация точек для параметров анзаца
initial_point_values = 2 * np.pi * np.random.rand(ansatz_opt.num_parameters)

# Определение и запуск алгоритма VQE
vqe = VQE(
    estimator=estimator,
    ansatz=ansatz_opt,
    optimizer=optimizer,
    initial_point=initial_point_values
)

# Вычисление минимальной энергии с использованием гамильтониана
result = vqe.compute_minimum_eigenvalue(H_op)

# Нормализация минимальной энергии
normalized_energy = result.eigenvalue.real  # Просто без смещения для проверки

print("Минимальная энергия: ", normalized_energy)

# Сравнение с таблицей значений для элементов
element_energies = {
    "H2": -1.9,
    "Be": -14.7,
    "He": -2.9,
    "Li": -7.3
}

# Поиск элемента, который наиболее близок к нормализованной энергии
closest_element = min(element_energies, key=lambda x: abs(element_energies[x] - normalized_energy))
print("Ближайший элемент: ", closest_element)
print(H_op)