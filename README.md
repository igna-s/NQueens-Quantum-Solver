# N-Queens Quantum Solver

This project implements and compares quantum and classical algorithms for solving the **N-Queens Problem**. It features a modern web interface for visualizing and running the different approaches.

🌍 **Live Demo:** [https://thankful-pond-0af99ab10.7.azurestaticapps.net/](https://thankful-pond-0af99ab10.7.azurestaticapps.net/)

## Overview

The repository explores how quantum computing can be applied to classical constraint satisfaction problems like N-Queens. It includes:
- **Grover's Algorithm:** A quantum search approach with amplitude amplification.
- **QAOA (Quantum Approximate Optimization Algorithm):** A variational quantum approach.
- **Classical Algorithm:** A highly optimized bitwise approach in C for a performance baseline.

## Architecture

The project is structured into several components:
- **`app/`**: The frontend web application providing an interactive UI to run the algorithms.
- **`api/`**: Serverless backend built with **Azure Functions**. Contains the Python implementations for Qiskit (Grover and QAOA) and the C classical implementation.
- **`Extra/`**: Additional scripts for circuit drawing and running jobs on **Azure Quantum** hardware/simulators.
- **`data/`**: A comprehensive LaTeX report detailing the mathematical formulation and implementation of the Grover's algorithm for N-Queens.

## Tech Stack
- **Frontend:** Vanilla HTML/CSS/JavaScript
- **Backend:** Azure Functions (Python, C)
- **Quantum Framework:** Qiskit, Azure Quantum
- **Hosting:** Azure Static Web Apps

## Running Locally

To run the backend API locally:
1. Install Azure Functions Core Tools.
2. Navigate to the `api/` directory.
3. Run `func start`.

To view the frontend, simply open `app/index.html` in your browser or serve it using a local static server.

## Author
Ignacio Andres Schwindt
