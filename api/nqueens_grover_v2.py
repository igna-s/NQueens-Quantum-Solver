"""
N-reinas cuántico vía algoritmo de Grover - VERSIÓN 2 (Arquitectura Optimizada)

Esta versión implementa las mejoras arquitectónicas del "Santo Grial" para escalar (ej. N=50):
1. Reemplaza los flags combinatorios O(N^2) por un único Acumulador Binario (Ahorro masivo de memoria).
2. Reemplaza el match_pattern O(N^4) por Restadores Cuánticos (Ahorro masivo de profundidad).
"""

import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations
from math import pi, ceil, log2, sqrt

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate

# ---------------------------------------------------------------------------
# Cajas Negras Aritméticas (Abstracciones de Hardware)
# ---------------------------------------------------------------------------

class QuantumSubtractorConflict(Gate):
    """
    Abstracción de un Restador Cuántico.
    Matemáticamente calcula |col_i - col_j| y evalúa si chocan en columna (==0)
    o en diagonal (== d_rows).
    Costo en hardware real: ~O(k) compuertas base. Usamos dummy gates para aproximarlo.
    """
    def __init__(self, k):
        # Entradas: k qubits (reina i) + k qubits (reina j) + 1 qubit (flag temporal)
        super().__init__("Subtractor_Conflict", 2 * k + 1, [])
        self.k = k

    def _define(self):
        qc = QuantumCircuit(2 * self.k + 1)
        # Dummy gates para simular la profundidad asintótica (O(k))
        # Esto le dirá al simulador que la caja negra tiene una profundidad proporcional a k
        for _ in range(self.k * 3):
            qc.cx(0, 1) 
        self.definition = qc

class QuantumAdder(Gate):
    """
    Abstracción de un Sumador Cuántico (+1).
    Suma el valor del flag temporal (0 o 1) al Registro Acumulador Binario.
    Costo en hardware real: ~O(k_acc) compuertas base.
    """
    def __init__(self, k_acc):
        # Entradas: 1 qubit (flag temporal) + k_acc qubits (registro acumulador)
        super().__init__("Adder_Accumulator", 1 + k_acc, [])
        self.k_acc = k_acc

    def _define(self):
        qc = QuantumCircuit(1 + self.k_acc)
        # Dummy gates para simular la profundidad de la suma (O(k_acc))
        for _ in range(self.k_acc * 2):
            qc.cx(0, 1)
        self.definition = qc

# ---------------------------------------------------------------------------
# Construcción del Oráculo V2
# ---------------------------------------------------------------------------

def build_arithmetic_oracle(N):
    """
    Oráculo V2: Utiliza aritmética y un registro contador global (Accumulator).
    """
    k = max(1, ceil(log2(N)))
    n_state = N * k
    
    # 1. Registro Contador Binario (Acumulador)
    # Debe poder contar hasta C(N,2) posibles colisiones en el peor caso.
    max_collisions = N * (N - 1) // 2
    k_acc = max(1, ceil(log2(max_collisions + 1)))
    
    # 2. Registros Cuánticos
    state_reg = QuantumRegister(n_state, "q")
    accumulator_reg = QuantumRegister(k_acc, "acc")
    scratchpad_flag = QuantumRegister(1, "scratch") # Memoria reciclable temporal
    
    regs = [state_reg, accumulator_reg, scratchpad_flag]
    
    needs_range = (2**k) > N
    range_flags = None
    if needs_range:
        range_flags = QuantumRegister(N, "rf")
        regs.append(range_flags)

    qc = QuantumCircuit(*regs, name="Arithmetic_Oracle")

    # Instanciar puertas lógicas abstractas
    subtractor_gate = QuantumSubtractorConflict(k)
    adder_gate = QuantumAdder(k_acc)
    subtractor_dg = subtractor_gate.inverse()
    adder_dg = adder_gate.inverse()
    
    # --- ETAPA 1: COMPUTE CONFLICTS ---
    for i, j in combinations(range(N), 2):
        q_i = [state_reg[i * k + b] for b in range(k)]
        q_j = [state_reg[j * k + b] for b in range(k)]
        
        # a) Restador: Detectar conflicto aritméticamente -> resultado va a scratchpad
        qc.append(subtractor_gate, q_i + q_j + [scratchpad_flag[0]])
        
        # b) Acumulador: Si hay conflicto, sumar +1 al contador global
        qc.append(adder_gate, [scratchpad_flag[0]] + list(accumulator_reg))
        
        # c) Uncompute Restador: Limpiar scratchpad para el próximo par (Reutilización de memoria)
        qc.append(subtractor_dg, q_i + q_j + [scratchpad_flag[0]])

    # Detección de rango usando una abstracción simplificada de costo constante para V2
    if needs_range:
        for i in range(N):
            qc.x(range_flags[i]) # Placeholder lógico 

    # --- ETAPA 2: PHASE FLIP (-1) ---
    # Si TODOS los contadores están en 0, es una solución válida.
    all_acc_range = list(accumulator_reg) + (list(range_flags) if needs_range else [])
    
    qc.x(all_acc_range) # 0 se vuelve 1 para que el MCZ dispare
    
    # Aplicar MCZ = H · MCX · H
    if len(all_acc_range) > 1:
        qc.h(all_acc_range[-1])
        qc.mcx(all_acc_range[:-1], all_acc_range[-1])
        qc.h(all_acc_range[-1])
    elif len(all_acc_range) == 1:
        qc.z(all_acc_range[0])
        
    qc.x(all_acc_range) # Restaurar los 1s a 0s

    # --- ETAPA 3: UNCOMPUTE CONFLICTS ---
    if needs_range:
        for i in reversed(range(N)):
            qc.x(range_flags[i])
            
    for i, j in reversed(list(combinations(range(N), 2))):
        q_i = [state_reg[i * k + b] for b in range(k)]
        q_j = [state_reg[j * k + b] for b in range(k)]
        
        qc.append(subtractor_gate, q_i + q_j + [scratchpad_flag[0]])
        qc.append(adder_dg, [scratchpad_flag[0]] + list(accumulator_reg))
        qc.append(subtractor_dg, q_i + q_j + [scratchpad_flag[0]])

    return qc, state_reg, accumulator_reg, range_flags

# ---------------------------------------------------------------------------
# Análisis y Métricas
# ---------------------------------------------------------------------------

def analyze_v2(N):
    oracle, state_reg, accumulator_reg, range_flags = build_arithmetic_oracle(N)
    
    # Transpile básico para desarmar los bloques abstractos en gates elementales
    qc_t = transpile(oracle, basis_gates=["u", "cx", "h", "x", "ccx"], optimization_level=1)
    
    k = max(1, ceil(log2(N)))
    info = {
        "N": N,
        "k": k,
        "n_state_qubits": len(state_reg),
        "n_accumulator_qubits": len(accumulator_reg),
        "n_range_flags": (len(range_flags) if range_flags is not None else 0),
        "n_scratchpad_qubits": 1,
        "total_qubits": oracle.num_qubits,
        "oracle_depth": qc_t.depth()
    }
    return info

def print_v2_metrics(N, info):
    print(f"\n{'='*60}")
    print(f"  N = {N} (Arquitectura V2 - Aritmética Cuántica)")
    print(f"{'='*60}")
    print(f"  Qubits estado        : {info['n_state_qubits']}  ({info['k']} bits/columna)")
    print(f"  Qubits contador      : {info['n_accumulator_qubits']} (Acumulador binario de colisiones)")
    print(f"  Qubits scratchpad    : {info['n_scratchpad_qubits']} (Reciclables)")
    print(f"  Flags rango          : {info['n_range_flags']}")
    print(f"  TOTAL QUBITS         : {info['total_qubits']}")
    print(f"  PROFUNDIDAD ORÁCULO  : {info['oracle_depth']} gates (Estimación con bloques O(k))")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Iniciando análisis de arquitectura V2 (Restador + Acumulador)...")
    
    info_4 = analyze_v2(4)
    print_v2_metrics(4, info_4)
    
    print("\n[NOTA DE ESCALABILIDAD]")
    print("Compara este output de V2 con el comportamiento de V1:")
    print(" - En V1, para N=4 la profundidad por iteración es de ~5.100 compuertas.")
    print(" - En V2, para N=4 la profundidad cae drásticamente gracias a la aritmética y el uso del contador.")
