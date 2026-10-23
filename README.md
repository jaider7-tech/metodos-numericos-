# Métodos Numéricos — aplicación web

Resuelve **cualquier ejercicio** de los temas del curso: se escribe la función, el sistema
o los datos, y la app calcula, muestra la tabla de iteraciones y grafica.

| Tema | Métodos |
|---|---|
| Raíces de ecuaciones | Bisección, Regula Falsi, Punto Fijo, Newton-Raphson, Secante |
| Sistemas lineales | Gauss-Jordan (paso a paso), Gauss-Seidel, Gauss-Jacobi, Algoritmo de Thomas |
| Interpolación | Lagrange, Newton (diferencias divididas), Neville, Hermite |
| Integración numérica | Trapecio, Simpson 1/3, Simpson 3/8, Boole, Cuadratura de Gauss-Legendre (1–20 puntos) |
| Ecuaciones diferenciales (PVI) | Euler, Cauchy-Euler (Heun), Punto medio (RK2), Runge-Kutta 4 |

Extras: detener por tolerancia **o** por número fijo de iteraciones (como en clase), error
absoluto o relativo %, comparación entre métodos, valor de referencia / solución exacta
para medir el error, y verificación contra `numpy`.

## Estructura

```
app.py                 interfaz web (solo pide datos y dibuja)
nucleo/
  parser.py            texto -> funciones, sistemas, listas de puntos
  raices.py            5 métodos de raíces (un solo motor de iteración)
  sistemas.py          Gauss-Jordan, Seidel/Jacobi, Thomas
  interpolacion.py     Lagrange, Newton, Neville, Hermite
  integracion.py       Newton-Cotes compuestas y Gauss-Legendre
  edo.py               Euler, Heun, RK2, RK4 (un solo motor de un paso)
pruebas.py             18 pruebas automáticas (python pruebas.py)
requirements.txt
```

## Cómo escribir

| Quieres | Escribe |
|---|---|
| x³ − x − 1 | `x^3 - x - 1` |
| 2x·sen(x) | `2x sen(x)` o `2xsen(x)` |
| eˣ, ln x | `exp(x)` o `e^x`, `ln(x)` |
| √(x+2), ∛(x⁵−2) | `sqrt(x+2)`, `cbrt(x^5-2)` |
| dy/dx = y − x² + 1 | `y - x^2 + 1` |
| Puntos | `(0,3)(2,3)(4,0)` |
| Hermite | una línea por punto: `x  f(x)  f'(x)` |
