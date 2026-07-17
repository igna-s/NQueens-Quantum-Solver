import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
from itertools import combinations, permutations
from math import pi, ceil, log2, asin, sqrt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

def classical_solutions(N):
    sols = []
    for perm in permutations(range(N)):
        ok = True
        for i, j in combinations(range(N), 2):
            if abs(perm[i] - perm[j]) == j - i:
                ok = False
                break
        if ok:
            sols.append(list(perm))
    return sols

def decode(bitstring, N, k):
    bits = bitstring[::-1]
    queens = []
    for i in range(N):
        col = 0
        for b in range(k):
            if bits[i * k + b] == "1":
                col |= 1 << b
        queens.append(col)
    return queens

def is_valid(queens, N):
    if any(c < 0 or c >= N for c in queens):
        return False
    for i, j in combinations(range(N), 2):
        if queens[i] == queens[j] or abs(queens[i] - queens[j]) == j - i:
            return False
    return True

def print_board(queens, N):
    for i in range(N):
        line = "    "
        for c in range(N):
            line += "Q " if queens[i] == c else ". "
        print(line)

def add_conflict_detection(qc, q_i, q_j, d, scratch):
    for A in range(4):
        for B in range(4):
            if A == B or abs(A - B) == d:
                x_flips = []
                if (A & 1) == 0: x_flips.append(q_i[0])
                if (A & 2) == 0: x_flips.append(q_i[1])
                if (B & 1) == 0: x_flips.append(q_j[0])
                if (B & 2) == 0: x_flips.append(q_j[1])
                
                if x_flips: qc.x(x_flips)
                qc.mcx(q_i + q_j, scratch)
                if x_flips: qc.x(x_flips)

def add_1_controlled(qc, flag, acc):
    qc.mcx([flag, acc[0], acc[1]], acc[2])
    qc.ccx(flag, acc[0], acc[1])
    qc.cx(flag, acc[0])

def uncompute_add_1_controlled(qc, flag, acc):
    qc.cx(flag, acc[0])
    qc.ccx(flag, acc[0], acc[1])
    qc.mcx([flag, acc[0], acc[1]], acc[2])

def build_arithmetic_oracle(N):
    k = 2
    n_state = N * k
    k_acc = 3
    
    state_reg = QuantumRegister(n_state, "q")
    acc_reg = QuantumRegister(k_acc, "acc")
    scratch = QuantumRegister(1, "scratch")
    
    qc = QuantumCircuit(state_reg, acc_reg, scratch, name="Arithmetic_Oracle")
    
    for i, j in combinations(range(N), 2):
        d = j - i
        q_i = [state_reg[i*k + b] for b in range(k)]
        q_j = [state_reg[j*k + b] for b in range(k)]
        add_conflict_detection(qc, q_i, q_j, d, scratch[0])
        add_1_controlled(qc, scratch[0], acc_reg)
        add_conflict_detection(qc, q_i, q_j, d, scratch[0])
        
    qc.x(acc_reg)
    qc.h(acc_reg[-1])
    qc.mcx(list(acc_reg[:-1]), acc_reg[-1])
    qc.h(acc_reg[-1])
    qc.x(acc_reg)
    
    for i, j in reversed(list(combinations(range(N), 2))):
        d = j - i
        q_i = [state_reg[i*k + b] for b in range(k)]
        q_j = [state_reg[j*k + b] for b in range(k)]
        add_conflict_detection(qc, q_i, q_j, d, scratch[0])
        uncompute_add_1_controlled(qc, scratch[0], acc_reg)
        add_conflict_detection(qc, q_i, q_j, d, scratch[0])
        
    return qc, state_reg, acc_reg, scratch

def diffuser(n_state):
    qc = QuantumCircuit(n_state, name="Diff")
    qc.h(range(n_state))
    qc.x(range(n_state))
    qc.h(n_state - 1)
    qc.mcx(list(range(n_state - 1)), n_state - 1)
    qc.h(n_state - 1)
    qc.x(range(n_state))
    qc.h(range(n_state))
    return qc

def run_grover_v2(N=4, shots=4096):
    k = 2
    n_state = N * k
    total_states = 2 ** n_state
    n_sols = len(classical_solutions(N))
    
    theta = 2 * asin(sqrt(n_sols / total_states))
    iters = max(1, int(round(pi / (2 * theta) - 0.5)))
    
    oracle, state_reg, acc_reg, scratch = build_arithmetic_oracle(N)
    creg = ClassicalRegister(n_state, "c")
    qc = QuantumCircuit(state_reg, acc_reg, scratch, creg)
    
    qc.h(state_reg)
    diff = diffuser(n_state)
    for _ in range(iters):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, qubits=state_reg, inplace=True)
    qc.measure(state_reg, creg)
    
    sim = AerSimulator(method="statevector")
    qc_t = transpile(qc, basis_gates=["u", "cx", "h", "x", "ccx"], optimization_level=1)
    
    info = {
        "N": N,
        "k": k,
        "n_state_qubits": n_state,
        "n_acc_qubits": len(acc_reg),
        "n_scratch_qubits": len(scratch),
        "total_states": total_states,
        "n_solutions": n_sols,
        "iterations": iters,
        "circuit_depth": qc_t.depth(),
        "total_qubits": qc.num_qubits,
    }
    
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()
    return counts, info, k

def pretty_report_v2(N, counts, info, k, top=15):
    print(f"\n{'='*60}")
    print(f"  N = {N} (Arquitectura V2 - Aritmética Cuántica)")
    print(f"{'='*60}")
    print(f"  qubits estado  : {info['n_state_qubits']}  ({info['k']} bits/columna)")
    print(f"  qubits contador: {info['n_acc_qubits']} (Acumulador)")
    print(f"  qubits scratch : {info['n_scratch_qubits']}")
    print(f"  qubits totales : {info['total_qubits']}")
    print(f"  espacio        : {info['total_states']}")
    print(f"  soluciones     : {info['n_solutions']}")
    print(f"  iteraciones    : {info['iterations']}")
    print(f"  profundidad    : {info['circuit_depth']}")

    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    total = sum(counts.values())
    valid_count = 0
    unique_sols = set()

    print(f"\n  Top {top} mediciones:")
    for bs, c in sorted_counts[:top]:
        queens = decode(bs, N, k)
        ok = is_valid(queens, N)
        mark = "OK " if ok else "X  "
        print(f"    {mark} cols={queens}  count={c}  p={c/total:.3f}")

    for bs, c in counts.items():
        q = decode(bs, N, k)
        if is_valid(q, N):
            valid_count += c
            unique_sols.add(tuple(q))

    print(f"\n  P(medir solución válida) = {valid_count/total:.1%}")
    print(f"  Soluciones únicas medidas: {len(unique_sols)} / {info['n_solutions']}")

    print("\n  Tableros (de mayor a menor frecuencia):")
    sol_freq = {}
    for bs, c in counts.items():
        q = tuple(decode(bs, N, k))
        if is_valid(list(q), N):
            sol_freq[q] = sol_freq.get(q, 0) + c
    for sol, c in sorted(sol_freq.items(), key=lambda x: -x[1]):
        print(f"\n    cols={list(sol)}  count={c}")
        print_board(list(sol), N)

if __name__ == "__main__":
    counts, info, k = run_grover_v2(4, shots=4096)
    pretty_report_v2(4, counts, info, k)
