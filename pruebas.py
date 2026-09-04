"""
Verificacion automatica del nucleo de calculo.

Cada metodo se compara contra el valor exacto conocido o contra la solucion
de NumPy.  Ejecutar con:   python pruebas.py
"""

import numpy as np

from nucleo import parser, raices, sistemas

TOL = 1e-6
resultados = []


def revisar(nombre, obtenido, esperado, tolerancia=1e-6):
    obtenido = np.atleast_1d(np.array(obtenido, dtype=float))
    esperado = np.atleast_1d(np.array(esperado, dtype=float))
    error = float(np.max(np.abs(obtenido - esperado)))
    paso = error < tolerancia
    resultados.append(paso)
    print(f"[{'OK ' if paso else 'MAL'}] {nombre:<46} error = {error:.2e}")


print("\n--- RAICES ------------------------------------------------------")

f, expr = parser.funcion("x^3 - x - 1")
df, _ = parser.derivada(expr)
exacta = 1.3247179572447458          # raiz real de x^3 - x - 1

revisar("Biseccion        x^3-x-1 en [1,2]",
        raices.biseccion(f, 1, 2, TOL, 200).raiz, exacta, 1e-5)
revisar("Regula Falsi     x^3-x-1 en [1,2]",
        raices.regula_falsi(f, 1, 2, TOL, 200).raiz, exacta, 1e-5)
revisar("Newton-Raphson   x^3-x-1 desde x0=1",
        raices.newton_raphson(f, df, 1.0, 1e-12, 100).raiz, exacta)
revisar("Secante          x^3-x-1 desde 1 y 2",
        raices.secante(f, 1, 2, 1e-12, 100).raiz, exacta)

g, _ = parser.funcion("cos(x)")
revisar("Punto fijo       x = cos(x)",
        raices.punto_fijo(g, 0.5, 1e-10, 500).raiz, 0.7390851332151607, 1e-8)

f2, _ = parser.funcion("x^5 - 3x^2 + 4x - 1")
revisar("Biseccion        x^5-3x^2+4x-1 en [0,1]",
        raices.biseccion(f2, 0, 1, 1e-10, 200).raiz, 0.3313423853092938, 1e-8)

# la biseccion debe rechazar un intervalo sin cambio de signo
f3, _ = parser.funcion("x^2 + 1")
try:
    raices.biseccion(f3, 0, 1)
    resultados.append(False)
    print("[MAL] Biseccion deberia rechazar x^2+1 en [0,1]")
except raices.ErrorDeMetodo:
    resultados.append(True)
    print(f"[OK ] {'Biseccion rechaza intervalo sin cambio de signo':<46}")


print("\n--- SISTEMAS ----------------------------------------------------")

A, b, _ = parser.sistema_desde_texto(
    "2x + y + z = 3\n4x + 3y + 2z = 6\n-2x + 5y + 4z = 7")
exacto = np.linalg.solve(A, b)
revisar("Gauss-Jordan     sistema 3x3", sistemas.gauss_jordan(A, b).solucion, exacto)

A, b, _ = parser.sistema_desde_texto(
    "7x + 8y + 5z = 3\n3x + 4y - 10z = 6\n9x + 2y - z = -2")
exacto = np.linalg.solve(A, b)
revisar("Gauss-Seidel     sistema del cuaderno",
        sistemas.gauss_seidel(A, b, tol=1e-10, max_iter=500).solucion, exacto, 1e-8)
revisar("Gauss-Jacobi     sistema del cuaderno",
        sistemas.gauss_jacobi(A, b, tol=1e-10, max_iter=500).solucion, exacto, 1e-8)

A, b, _ = parser.sistema_desde_texto(
    "10x - y + 2z = 6\n-x + 11y - z + 3w = 25\n2x - y + 10z - w = -11\n3y - z + 8w = 15")
revisar("Gauss-Seidel     sistema 4x4 dominante",
        sistemas.gauss_seidel(A, b, tol=1e-12, max_iter=500).solucion,
        np.linalg.solve(A, b), 1e-8)

A, b, _ = parser.sistema_desde_texto(
    "3p - 2q = 1.0\np + 5q - 0.2r = 5.8\n4q + 7r = 11.0")
revisar("Thomas           sistema tridiagonal",
        sistemas.thomas(A, b).solucion, np.linalg.solve(A, b))

# Thomas debe rechazar una matriz llena
A, b, _ = parser.sistema_desde_texto("x + y + z = 1\nx + 2y + 3z = 2\nx + 4y + 9z = 3")
try:
    sistemas.thomas(A, b)
    resultados.append(False)
    print("[MAL] Thomas deberia rechazar una matriz no tridiagonal")
except sistemas.ErrorDeMetodo:
    resultados.append(True)
    print(f"[OK ] {'Thomas rechaza matriz no tridiagonal':<46}")


print("\n-----------------------------------------------------------------")
print(f"{sum(resultados)} de {len(resultados)} pruebas superadas.")
raise SystemExit(0 if all(resultados) else 1)
