"""
Verificacion automatica del nucleo (sin abrir la pagina):

    python pruebas.py

Cada prueba compara contra un valor de referencia independiente
(SymPy, numpy.linalg.solve, mpmath, solucion exacta) o contra los
ejercicios resueltos del cuaderno del curso.
"""

import numpy as np
import sympy as sp

from nucleo import edo, integracion as G, interpolacion as I, parser as P, raices as R, sistemas as S

pruebas = []


def prueba(nombre):
    def registrar(func):
        pruebas.append((nombre, func))
        return func
    return registrar


def cerca(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b}"


# ------------------------------------------------------------------ raices
@prueba("Biseccion x^5-3x^2+4x-1 en [0,1] (cuaderno)")
def _():
    _, f, _, _ = P.funcion_x("x^5 - 3x^2 + 4x - 1")
    r = R.biseccion(f, 0, 1, tol=0.01)
    cerca(r.raiz, 0.3359375, 1e-9)


@prueba("Punto fijo g=cos(x), 4 iteraciones (cuaderno)")
def _():
    _, g, _, _ = P.funcion_x("cos(x)")
    r = R.punto_fijo(g, 0.5, criterio="iteraciones", max_iter=4)
    x = 0.5
    for _ in range(4):
        x = np.cos(x)
    cerca(r.raiz, x, 1e-12)


@prueba("Newton x^3-3x-4, x0=2 contra raiz exacta de SymPy")
def _():
    e, f, df, _ = P.funcion_x("x^3 - 3x - 4")
    exacta = float(sp.nsolve(e, P.X, 2))
    for r in (R.newton_raphson(f, df, 2), R.secante(f, 2, 3), R.regula_falsi(f, 2, 3, tol=1e-10)):
        cerca(r.raiz, exacta, 1e-6)


@prueba("Biseccion rechaza intervalo sin cambio de signo")
def _():
    _, f, _, _ = P.funcion_x("x^2 + 1")
    try:
        R.biseccion(f, 0, 1)
    except R.ErrorDeMetodo:
        return
    raise AssertionError("debio fallar")


# ---------------------------------------------------------------- sistemas
@prueba("Gauss-Jordan (3 sistemas del cuaderno) vs numpy")
def _():
    for txt in ("2x+y+z=3\n4x+3y+2z=6\n-2x+5y+4z=7",
                "x+2y-z=8\n2x+3y+z=13\n3x+5y+2z=10",
                "x-y+3z=13\nx+y+z=11\n2x+2y-z=7"):
        A, b, n = P.sistema(txt)
        for piv in (True, False):
            assert np.allclose(S.gauss_jordan(A, b, n, piv).solucion, np.linalg.solve(A, b))


@prueba("Seidel y Jacobi reordenan y convergen (sistema del cuaderno)")
def _():
    A, b, n = P.sistema("7x+8y+5z=3\n3x+4y-10z=6\n9x+2y-z=-2")
    for seidel in (True, False):
        r = S.iterativo(A, b, n, seidel, tol=1e-10, tipo_error="absoluto")
        assert r.convergio and r.extra["rho_final"] < 1
        assert np.allclose(r.solucion, np.linalg.solve(A, b), atol=1e-8)


@prueba("Thomas 3x3 y 4x4 (cuaderno) vs numpy")
def _():
    for txt in ("3x1-2x2=1\nx1+5x2-0.2x3=5.8\n4x2+7x3=11",
                "0.5X1-2.7X2=-5.952\nX1-0.2X2-3X3=-1.35\n4.9X2-7.8X3+4X4=4.975\n2.8X3-3.4X4=-0.1"):
        A, b, n = P.sistema(txt)
        a, bd, c = S.diagonales(A)
        assert np.allclose(S.thomas(a, bd, c, b, n).solucion, np.linalg.solve(A, b))


@prueba("Thomas rechaza matriz no tridiagonal")
def _():
    A, _, _ = P.sistema("x+y+z=1\nx+2y+z=2\nx+y+3z=3")
    assert not S.es_tridiagonal(A)


# ------------------------------------------------------------ interpolacion
@prueba("Lagrange (0,3)(2,3)(4,0) = -3/8 x^2 + 3/4 x + 3")
def _():
    r = I.lagrange([0, 2, 4], [3, 3, 0])
    assert sp.simplify(r.polinomio - (-sp.Rational(3, 8) * P.X**2 + sp.Rational(3, 4) * P.X + 3)) == 0


@prueba("Newton = Lagrange en los 3 ejercicios del cuaderno")
def _():
    for xs, ys in (([1, 2, 3], [2, 3, 5]), ([1, 0, -3], [2, 4, -2]), ([2, 3, 5], [5, 10, 26])):
        assert sp.simplify(I.newton(xs, ys).polinomio - I.lagrange(xs, ys).polinomio) == 0


@prueba("Neville en X=2, 2.5, 3.2 coincide con el polinomio")
def _():
    for xs, ys, X0 in (([1, 3, 4], [2, 4, 1], 2), ([1, 2, 3], [2, 3, 5], 2.5), ([2, 3, 4], [1.5, 2.5, 4], 3.2)):
        cerca(I.neville(xs, ys, X0).valor, float(I.lagrange(xs, ys).polinomio.subs(P.X, X0)), 1e-12)


@prueba("Hermite cumple H y H' en los nodos (2 ejercicios del cuaderno)")
def _():
    r = I.hermite([[1, 2, 3], [2, 3, 1]])
    assert sp.expand(r.polinomio - (2 * P.X**3 - 10 * P.X**2 + 17 * P.X - 7)) == 0
    r = I.hermite([[-2, -12, 22], [1, 9, 10]])
    assert all(abs(v.Pedido - v.Obtenido) < 1e-12 for v in r.extra["verificacion"].itertuples())


# ------------------------------------------------------------- integracion
@prueba("Simpson 1/3 (cuaderno): x^2/2 [2,5] n=6 = 19.5 ; sqrt(x) [1,3] n=4")
def _():
    _, f, _, _ = P.funcion_x("x^2/2")
    cerca(G.newton_cotes(f, 2, 5, 6).valor, 19.5, 1e-12)
    e, f, _, _ = P.funcion_x("sqrt(x)")
    cerca(G.newton_cotes(f, 1, 3, 4).valor, G.integral_referencia(e, 1, 3, P.X), 1e-3)


@prueba("Trapecio, 3/8 y Boole convergen al valor de mpmath")
def _():
    e, f, _, _ = P.funcion_x("x/sqrt(x^3+1)")
    ref = G.integral_referencia(e, 0, 2, P.X)
    for regla, n in (("Trapecio", 600), ("Simpson 3/8", 99), ("Boole", 80)):
        cerca(G.newton_cotes(f, 0, 2, n, regla).valor, ref, 1e-5)


@prueba("Gauss-Legendre n=2 exacta hasta grado 3; x^4+x^2+1 da 2.8889 (cuaderno)")
def _():
    _, f, _, _ = P.funcion_x("2x^2+3x+1")
    cerca(G.gauss_legendre(f, 1, 3, 2).valor, 94 / 3, 1e-10)
    _, f, _, _ = P.funcion_x("x^4+x^2+1")
    cerca(G.gauss_legendre(f, -1, 1, 2).valor, 26 / 9, 1e-10)
    e, f, _, _ = P.funcion_x("sen(x^2)")
    cerca(G.gauss_legendre(f, 1, 2, 5).valor, G.integral_referencia(e, 1, 2, P.X), 1e-5)


# --------------------------------------------------------------------- EDO
@prueba("Euler y' = y - x^2 + 1, h=0.2 -> y(1)=2.4582 (cuaderno)")
def _():
    _, f = P.funcion_xy("y - x^2 + 1")
    cerca(edo.resolver(f, 0, 0.5, 0.2, 1, "Euler").ys[-1], 2.458176, 1e-6)


@prueba("RK4 y' = 2x - y hasta x=3 vs solucion exacta")
def _():
    e, f = P.funcion_xy("2x - y")
    expr, exacta = edo.solucion_exacta(e, 0, 1)
    assert expr is not None
    r = edo.resolver(f, 0, 1, 0.25, 3, "Runge-Kutta 4 (RK4)")
    cerca(r.ys[-1], exacta(3), 1e-4)


@prueba("Orden de precision: Euler > Heun > RK4 en error")
def _():
    e, f = P.funcion_xy("x + y")
    _, exacta = edo.solucion_exacta(e, 0, 1)
    errores = [abs(edo.resolver(f, 0, 1, 0.25, 1, m).ys[-1] - exacta(1))
               for m in ("Euler", "Cauchy-Euler (Heun / Euler mejorado)", "Runge-Kutta 4 (RK4)")]
    assert errores[0] > errores[1] > errores[2]


if __name__ == "__main__":
    fallas = 0
    for nombre, func in pruebas:
        try:
            func()
            print(f"  OK    {nombre}")
        except Exception as error:
            fallas += 1
            print(f"  FALLA {nombre}: {error}")
    print(f"\n{len(pruebas) - fallas}/{len(pruebas)} pruebas pasan.")
    raise SystemExit(1 if fallas else 0)
