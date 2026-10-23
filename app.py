"""
Metodos Numericos — aplicacion web (Streamlit).

Interfaz unicamente: pide datos, llama a nucleo/ y dibuja tablas y graficas.
Toda la matematica vive en la carpeta nucleo/.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from nucleo import edo, integracion, interpolacion, parser, raices, sistemas
from nucleo.parser import ErrorDeEntrada

st.set_page_config(page_title="Métodos Numéricos", page_icon="🧮", layout="wide")

ERRORES = (
    ErrorDeEntrada,
    raices.ErrorDeMetodo,
    sistemas.ErrorDeSistema,
    interpolacion.ErrorDeInterpolacion,
    integracion.ErrorDeIntegracion,
    edo.ErrorDeEDO,
)


# =================================================================== utilidades

def decimales():
    return st.session_state.get("decimales", 6)


def tabla(df, altura=None):
    if df is None or len(df) == 0:
        return
    d = decimales()
    df = df.copy()
    for c in df.columns:                       # columnas con None -> NaN para que salgan vacias
        if df[c].dtype == object and df[c].map(lambda v: v is None or isinstance(v, (int, float, np.number))).all():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    numericas = df.select_dtypes(include="number").columns
    formato = {c: f"{{:.{d}f}}" for c in numericas if c not in ("Iter", "i", "n")}
    opciones = {"hide_index": True}
    if altura:
        opciones["height"] = altura
    st.dataframe(df.style.format(formato, na_rep=""), **opciones)


def num(texto, etiqueta):
    try:
        return parser.numero(texto)
    except ErrorDeEntrada:
        raise ErrorDeEntrada(f"{etiqueta}: «{texto}» no es un número válido.")


def fmt(v):
    return "—" if v is None else f"{v:.{decimales()}f}"


def figura(ancho=8, alto=4.2):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.axhline(0, color="black", linewidth=0.8)
    return fig, ax


def mostrar(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def criterio_parada(clave, tol_defecto="0.000001", tipo_defecto=0):
    c1, c2, c3 = st.columns(3)
    criterio = c1.radio("Detener por", ["tolerancia", "iteraciones"], key=f"crit_{clave}",
                        horizontal=True,
                        help="«iteraciones» hace exactamente N iteraciones, como en los ejercicios de clase.")
    if criterio == "tolerancia":
        tol = num(c2.text_input("Tolerancia", tol_defecto, key=f"tol_{clave}"), "Tolerancia")
        max_iter = c3.number_input("Máx. iteraciones", 1, 10000, 100, key=f"max_{clave}")
    else:
        tol = 0.0
        c2.write("")
        max_iter = c3.number_input("Número de iteraciones", 1, 10000, 4, key=f"n_{clave}")
    tipo = st.radio("Tipo de error", ["absoluto", "relativo %"], index=tipo_defecto,
                    key=f"tipo_{clave}", horizontal=True,
                    help="absoluto = |x_nuevo − x_anterior|   ·   relativo % = |x_nuevo − x_anterior| / |x_nuevo| · 100")
    return criterio, tol, int(max_iter), tipo


def latex_expr(prefijo, expr):
    st.latex(f"{prefijo} = {sp.latex(expr)}")


def latex_decimal(prefijo, poli):
    """Muestra la version decimal solo si hay coeficientes fraccionarios."""
    try:
        coefs = sp.Poly(poli, interpolacion.X).coeffs()
    except Exception:
        return
    if any(not c.is_Integer for c in coefs):
        st.latex(f"{prefijo} \\approx {sp.latex(interpolacion.decimal(poli))}")


# =================================================================== RAÍCES

METODOS_RAICES = ["Bisección", "Regula Falsi (falsa posición)", "Punto Fijo", "Newton-Raphson", "Secante"]


def seccion_raices():
    metodo = st.sidebar.selectbox("Método", METODOS_RAICES)
    st.header(f"Raíces de ecuaciones — {metodo}")

    if metodo == "Punto Fijo":
        st.info("Escribe **g(x)** ya despejada de f(x) = 0 en la forma x = g(x). "
                "Ej.: para f(x) = cos(x) − x usa g(x) = cos(x); para x³ − x − 2 usa g(x) = cbrt(x + 2) o sqrt(x+2) si viene así.")
        texto = st.text_input("g(x) =", "cos(x)")
    else:
        texto = st.text_input("f(x) =", "x^3 - x - 1")

    expr, f, df, dexpr = parser.funcion_x(texto)
    latex_expr("g(x)" if metodo == "Punto Fijo" else "f(x)", expr)

    c1, c2, c3 = st.columns(3)
    extra = {}
    a = b = x0 = x1 = None
    if metodo in ("Bisección", "Regula Falsi (falsa posición)"):
        a = num(c1.text_input("a (extremo izquierdo)", "1"), "a")
        b = num(c2.text_input("b (extremo derecho)", "2"), "b")
        with c3:
            st.caption("¿No sabes el intervalo?")
            if st.button("Buscar cambios de signo en [−10, 10]"):
                encontrados = raices.buscar_intervalos(f, -10, 10, 400)
                st.write(", ".join(f"[{p:g}, {q:g}]" for p, q in encontrados) or "No hay cambios de signo en [−10, 10].")
    elif metodo == "Secante":
        x0 = num(c1.text_input("x₀", "1"), "x₀")
        x1 = num(c2.text_input("x₁", "2"), "x₁")
    else:
        x0 = num(c1.text_input("Valor inicial x₀", "0.5" if metodo == "Punto Fijo" else "1"), "x₀")
        if metodo == "Newton-Raphson":
            with c2:
                st.latex(f"f'(x) = {sp.latex(dexpr)}")
                st.caption("La derivada la calcula el programa.")
        if metodo == "Punto Fijo":
            exacta = c2.text_input("Raíz exacta (opcional, para |error real|)", "")
            if exacta.strip():
                extra["raiz_exacta"] = num(exacta, "Raíz exacta")
            dg = df(x0)
            if np.isfinite(dg):
                c3.metric("|g'(x₀)|", f"{abs(dg):.4f}",
                          "converge (<1)" if abs(dg) < 1 else "puede diverger (≥1)",
                          delta_color="normal" if abs(dg) < 1 else "inverse")

    criterio, tol, max_iter, tipo = criterio_parada(
        "raiz", tipo_defecto=1 if metodo == "Newton-Raphson" else 0)
    comparar = metodo != "Punto Fijo" and st.checkbox(
        "Comparar también con los otros métodos (usa los mismos datos: intervalo [a,b] o x₀)")

    if not st.button("Resolver", type="primary"):
        return

    kw = dict(tol=tol, max_iter=max_iter, criterio=criterio, tipo_error=tipo)
    if metodo == "Bisección":
        r = raices.biseccion(f, a, b, **kw)
    elif metodo.startswith("Regula"):
        r = raices.regula_falsi(f, a, b, **kw)
    elif metodo == "Punto Fijo":
        r = raices.punto_fijo(f, x0, **kw, **extra)
    elif metodo == "Newton-Raphson":
        r = raices.newton_raphson(f, df, x0, **kw)
    else:
        r = raices.secante(f, x0, x1, **kw)

    (st.success if r.convergio else st.warning)(r.mensaje)
    if r.raiz is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raíz aproximada" if metodo != "Punto Fijo" else "Punto fijo x*", fmt(r.raiz))
        c2.metric("Iteraciones", r.iteraciones)
        if metodo == "Punto Fijo":
            c3.metric("|g(x*) − x*|", f"{abs(f(r.raiz) - r.raiz):.3e}")
        else:
            c3.metric("f(raíz)", f"{f(r.raiz):.3e}")
    st.subheader("Tabla de iteraciones")
    tabla(r.tabla)

    # ---- gráfica
    if r.historial:
        pts = list(r.historial) + [v for v in (a, b, x0, x1) if v is not None]
        lo, hi = min(pts), max(pts)
        margen = max(1.0, (hi - lo) * 0.4)
        xs = np.linspace(lo - margen, hi + margen, 600)
        fig, ax = figura()
        if metodo == "Punto Fijo":
            ax.plot(xs, f(xs), label=f"g(x) = {texto}", linewidth=2)
            ax.plot(xs, xs, "--", color="gray", label="y = x")
            xk = x0
            for xn in r.historial:                       # telaraña
                ax.plot([xk, xk], [xk, xn], color="tab:red", linewidth=0.9)
                ax.plot([xk, xn], [xn, xn], color="tab:red", linewidth=0.9)
                xk = xn
            ax.scatter([r.raiz], [r.raiz], color="red", zorder=5, s=60, label=f"x* ≈ {r.raiz:.6f}")
            ax.set_title("Punto fijo: intersección de y = g(x) con y = x")
        else:
            ys = f(xs)
            ax.plot(xs, ys, label=f"f(x) = {texto}", linewidth=2)
            ax.scatter(r.historial, [f(v) for v in r.historial], color="orange", s=25, zorder=4,
                       label="Aproximaciones")
            if r.raiz is not None:
                ax.scatter([r.raiz], [0], color="red", s=70, zorder=5, label=f"Raíz ≈ {r.raiz:.6f}")
            finitos = ys[np.isfinite(ys)]
            if len(finitos):
                q1, q9 = np.percentile(finitos, [2, 98])
                ax.set_ylim(q1 - 0.2 * abs(q9 - q1) - 0.5, q9 + 0.2 * abs(q9 - q1) + 0.5)
            ax.set_title(f"Método de {metodo}")
        ax.set_xlabel("x")
        ax.legend()
        mostrar(fig)

        if len(r.tabla) > 1:
            col = "Error abs" if tipo == "absoluto" else "Error rel %"
            errores = r.tabla[col].replace(0, np.nan)
            if errores.notna().any() and (errores > 0).any():
                fig, ax = figura(8, 3)
                ax.semilogy(r.tabla["Iter"], errores, "o-")
                ax.set_xlabel("Iteración")
                ax.set_ylabel(col)
                ax.set_title("Evolución del error (escala logarítmica)")
                mostrar(fig)

    if comparar:
        st.subheader("Comparación de métodos")
        filas = []
        if a is not None:
            ref_a, ref_b, ref_x0 = a, b, (a + b) / 2
        else:
            ref_a, ref_b, ref_x0 = x0, (x1 if x1 is not None else x0 + 1), x0
        corridas = {
            "Bisección": lambda: raices.biseccion(f, ref_a, ref_b, **kw),
            "Regula Falsi": lambda: raices.regula_falsi(f, ref_a, ref_b, **kw),
            "Newton-Raphson": lambda: raices.newton_raphson(f, df, ref_x0, **kw),
            "Secante": lambda: raices.secante(f, ref_a, ref_b, **kw),
        }
        for nombre, correr in corridas.items():
            try:
                rr = correr()
                filas.append({"Método": nombre, "Raíz": rr.raiz, "Iteraciones": rr.iteraciones,
                              "Convergió": "sí" if rr.convergio else "no", "Detalle": rr.mensaje})
            except ERRORES as e:
                filas.append({"Método": nombre, "Raíz": None, "Iteraciones": None,
                              "Convergió": "no aplica", "Detalle": str(e)})
        tabla(pd.DataFrame(filas))


# =================================================================== SISTEMAS

METODOS_SISTEMAS = ["Gauss-Jordan", "Gauss-Seidel", "Gauss-Jacobi", "Algoritmo de Thomas"]


def seccion_sistemas():
    metodo = st.sidebar.selectbox("Método", METODOS_SISTEMAS)
    st.header(f"Sistemas de ecuaciones lineales — {metodo}")

    if metodo == "Algoritmo de Thomas":
        modo = st.radio("Cómo quieres ingresar el sistema", ["Ecuaciones", "Diagonales a, b, c, d"], horizontal=True)
        if modo == "Diagonales a, b, c, d":
            st.caption("a = subdiagonal (empieza en 0), b = diagonal principal, c = superdiagonal (termina en 0), "
                       "d = lado derecho. Separa con comas o espacios (si usas coma decimal, separa con «;»).")
            c1, c2 = st.columns(2)
            a = parser.lista_numeros(c1.text_input("a (subdiagonal)", "0, 1, 4"))
            bd = parser.lista_numeros(c2.text_input("b (diagonal)", "3, 5, 7"))
            c = parser.lista_numeros(c1.text_input("c (superdiagonal)", "-2, -0.2, 0"))
            d = parser.lista_numeros(c2.text_input("d (lado derecho)", "1, 5.8, 11"))
            nombres = [f"x{i+1}" for i in range(len(d))]
            if st.button("Resolver", type="primary"):
                _mostrar_thomas(sistemas.thomas(a, bd, c, d, nombres), nombres)
            return
        defecto = "3x1 - 2x2 = 1\nx1 + 5x2 - 0.2x3 = 5.8\n4x2 + 7x3 = 11"
    elif metodo in ("Gauss-Seidel", "Gauss-Jacobi"):
        defecto = "7x + 8y + 5z = 3\n3x + 4y - 10z = 6\n9x + 2y - z = -2"
    else:
        defecto = "2x + y + z = 3\n4x + 3y + 2z = 6\n-2x + 5y + 4z = 7"

    texto = st.text_area("Escribe una ecuación por línea (las incógnitas pueden llamarse como quieras)",
                         defecto, height=140)
    A, b, nombres = parser.sistema(texto)
    with st.expander("Matriz A y vector b detectados"):
        st.dataframe(pd.DataFrame(np.column_stack([A, b]), columns=nombres + ["| b"]), hide_index=True)

    if metodo == "Gauss-Jordan":
        pivoteo = st.checkbox("Pivoteo parcial (más estable; desmárcalo para reproducir el procedimiento a mano)", True)
        if st.button("Resolver", type="primary"):
            r = sistemas.gauss_jordan(A, b, nombres, pivoteo)
            st.success(r.mensaje)
            _metricas_solucion(r.solucion, nombres)
            with st.expander("Paso a paso de las operaciones de fila", expanded=True):
                for titulo, M in r.pasos:
                    st.markdown(f"**{titulo}**")
                    tabla(M)
            _verificar(A, b, r.solucion, nombres)

    elif metodo in ("Gauss-Seidel", "Gauss-Jacobi"):
        seidel = metodo == "Gauss-Seidel"
        c1, c2 = st.columns(2)
        x0txt = c1.text_input("Vector inicial x⁽⁰⁾", ", ".join(["0"] * len(b)))
        reordenar = c2.checkbox("Reordenar ecuaciones automáticamente si así converge", True)
        criterio, tol, max_iter, tipo = criterio_parada("sis", "0.0001", 1)
        if st.button("Resolver", type="primary"):
            x0 = parser.lista_numeros(x0txt)
            if len(x0) != len(b):
                raise ErrorDeEntrada(f"El vector inicial debe tener {len(b)} valores.")
            r = sistemas.iterativo(A, b, nombres, seidel, x0, tol, max_iter, criterio, tipo, reordenar)
            ex = r.extra
            if ex.get("reordenado"):
                st.info(f"Se reordenaron las ecuaciones (orden {[i + 1 for i in ex['orden']]}). "
                        f"Radio espectral de la matriz de iteración: {ex['rho_original']:.4f} → {ex['rho_final']:.4f}. "
                        "El método converge solo si ρ < 1.")
            rho = ex.get("rho_final", np.nan)
            dominante = sistemas.es_diagonal_dominante(ex["A"])
            c1, c2 = st.columns(2)
            c1.metric("Radio espectral ρ", f"{rho:.4f}", "converge" if rho < 1 else "diverge",
                      delta_color="normal" if rho < 1 else "inverse")
            c2.metric("¿Diagonalmente dominante?", "sí" if dominante else "no")
            st.markdown("**Despejes usados:**")
            st.code("\n".join(ex["despejes"]))
            (st.success if r.convergio else st.warning)(r.mensaje)
            _metricas_solucion(r.solucion, nombres)
            st.subheader("Tabla de iteraciones")
            tabla(r.tabla)
            fig, ax = figura(8, 3.6)
            for v in nombres:
                ax.plot(r.tabla["Iter"], r.tabla[v], "o-", label=v)
            ax.set_xlabel("Iteración")
            ax.set_title(f"Convergencia de {metodo}")
            ax.legend()
            mostrar(fig)
            _verificar(A, b, r.solucion, nombres)

    else:  # Thomas desde ecuaciones
        if not sistemas.es_tridiagonal(A):
            st.error("La matriz no es tridiagonal: Thomas solo sirve para sistemas tridiagonales. Usa Gauss-Jordan.")
            return
        a, bd, c = sistemas.diagonales(A)
        st.write(pd.DataFrame({"a (sub)": a, "b (diag)": bd, "c (super)": c, "d": b}))
        if st.button("Resolver", type="primary"):
            _mostrar_thomas(sistemas.thomas(a, bd, c, b, nombres), nombres)
            _verificar(A, b, sistemas.thomas(a, bd, c, b, nombres).solucion, nombres)


def _mostrar_thomas(r, nombres):
    st.success(r.mensaje)
    _metricas_solucion(r.solucion, nombres)
    st.subheader("Barrido hacia adelante")
    tabla(r.pasos[0][1])
    e = r.extra
    idx = np.arange(1, len(nombres) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    axs[0].plot(idx, e["b"], "bo--", label="antes")
    axs[0].plot(idx, e["b_mod"], "ro-", label="después")
    axs[0].set_title("Diagonal b")
    axs[1].plot(idx, e["d"], "go--", label="antes")
    axs[1].plot(idx, e["d_mod"], "mo-", label="después")
    axs[1].set_title("Lado derecho d")
    axs[2].bar(nombres, r.solucion, color="skyblue")
    axs[2].set_title("Solución")
    for ax in axs:
        ax.grid(True, alpha=0.4)
    axs[0].legend()
    axs[1].legend()
    mostrar(fig)


def _metricas_solucion(x, nombres):
    cols = st.columns(min(len(nombres), 6))
    for i, (v, nombre) in enumerate(zip(x, nombres)):
        cols[i % len(cols)].metric(nombre, fmt(v))


def _verificar(A, b, x, nombres):
    with st.expander("Verificación"):
        residuo = A @ x - b
        try:
            ref = np.linalg.solve(A, b)
            st.dataframe(pd.DataFrame({"Incógnita": nombres, "Este método": x, "numpy.linalg.solve": ref,
                                       "Diferencia": np.abs(x - ref)}), hide_index=True)
        except np.linalg.LinAlgError:
            pass
        st.write(f"‖A·x − b‖ = {np.linalg.norm(residuo):.3e}")


# =================================================================== INTERPOLACIÓN

METODOS_INTERP = ["Lagrange", "Newton (diferencias divididas)", "Neville", "Hermite"]


def seccion_interpolacion():
    metodo = st.sidebar.selectbox("Método", METODOS_INTERP)
    st.header(f"Interpolación — {metodo}")

    if metodo == "Hermite":
        st.caption("Una línea por punto: **x  f(x)  f'(x)**  (puedes agregar f''(x), f'''(x)... en la misma línea). "
                   "Separa con espacios, comas o «;».")
        texto = st.text_area("Datos", "-2  -12  22\n1  9  10", height=110)
        datos = []
        for linea in texto.splitlines():
            if linea.strip():
                datos.append(parser.lista_numeros(linea))
        Xtxt = st.text_input("Evaluar en x = (opcional)", "")
        X0 = num(Xtxt, "x") if Xtxt.strip() else None
        if not st.button("Resolver", type="primary"):
            return
        r = interpolacion.hermite(datos, X0)
        st.subheader("Polinomio de Hermite")
        latex_expr("H(x)", r.polinomio)
        latex_decimal("H(x)", r.polinomio)
        latex_expr("H'(x)", r.extra["derivada"])
        if r.valor is not None:
            st.metric(f"H({X0:g})", fmt(r.valor))
        st.subheader("Tabla de diferencias divididas (nodos repetidos)")
        tabla(r.tabla)
        st.markdown("**Forma de Newton:**")
        st.latex(f"H(x) = {sp.latex(r.forma)}")
        with st.expander("Verificación de las condiciones"):
            tabla(r.extra["verificacion"])
        nodos = r.extra["nodos"]
        xs_n = [n for n, _ in nodos]
        lo, hi = min(xs_n), max(xs_n)
        m = max(1.0, (hi - lo) * 0.3)
        xx = np.linspace(lo - m, hi + m, 400)
        H = sp.lambdify(interpolacion.X, r.polinomio, "numpy")
        dH = sp.lambdify(interpolacion.X, r.extra["derivada"], "numpy")
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(xx, H(xx) * np.ones_like(xx), label="H(x)", linewidth=2)
        for xn, vals in nodos:
            axs[0].scatter([xn], [vals[0]], color="red", zorder=5)
            if len(vals) > 1:
                tramo = np.linspace(xn - m / 2, xn + m / 2, 10)
                axs[0].plot(tramo, vals[0] + vals[1] * (tramo - xn), "--", color="gray")
        axs[0].set_title("H(x) y sus tangentes en los nodos")
        axs[1].plot(xx, dH(xx) * np.ones_like(xx), color="tab:red", label="H'(x)")
        axs[1].scatter([xn for xn, v in nodos if len(v) > 1], [v[1] for _, v in nodos if len(v) > 1],
                       color="purple", zorder=5, label="f'(x) dadas")
        axs[1].set_title("Derivada H'(x)")
        for ax in axs:
            ax.grid(True, alpha=0.4)
            ax.legend()
        mostrar(fig)
        return

    st.caption("Escribe los puntos como (x, y)(x, y)… o una pareja «x y» por línea. "
               "Si te dan una función, escríbela abajo y solo pon los x (los y se calculan).")
    usar_f = st.checkbox("Los y salen de una función f(x)")
    if usar_f:
        c1, c2 = st.columns(2)
        ftxt = c1.text_input("f(x) =", "log(x)")
        xs = parser.lista_numeros(c2.text_input("Valores de x", "1, 2, 4"))
        fexpr, fnum, _, _ = parser.funcion_x(ftxt)
        ys = [fnum(v) for v in xs]
    else:
        defecto = {"Lagrange": "(0,3)(2,3)(4,0)", "Newton (diferencias divididas)": "(1,2)(2,3)(3,5)",
                   "Neville": "(1,2)(3,4)(4,1)"}[metodo]
        xs, ys = parser.puntos(st.text_area("Puntos", defecto, height=80))
        fnum = None

    Xtxt = st.text_input("Interpolar en x =" + ("" if metodo == "Neville" else " (opcional)"),
                         "2" if metodo == "Neville" else "")
    X0 = num(Xtxt, "x") if Xtxt.strip() else None

    if not st.button("Resolver", type="primary"):
        return

    if metodo == "Lagrange":
        r = interpolacion.lagrange(xs, ys, X0)
    elif metodo.startswith("Newton"):
        r = interpolacion.newton(xs, ys, X0)
    else:
        r = interpolacion.neville(xs, ys, X0)

    st.subheader("Polinomio interpolante")
    latex_expr("P(x)", r.polinomio)
    latex_decimal("P(x)", r.polinomio)
    if r.valor is not None:
        c1, c2 = st.columns(2)
        c1.metric(f"P({X0:g})", fmt(r.valor))
        if fnum is not None:
            real = fnum(X0)
            c2.metric(f"f({X0:g}) real · error", fmt(real), f"error = {abs(real - r.valor):.3e}",
                      delta_color="off")

    if metodo == "Lagrange":
        st.markdown("**Polinomios base:**")
        for nombre, L in r.bases:
            st.latex(f"{nombre.replace('(x)', '')}(x) = {sp.latex(L)}")
        st.latex(f"P(x) = {sp.latex(r.forma)}")
        tabla(r.tabla)
    elif metodo.startswith("Newton"):
        st.subheader("Tabla de diferencias divididas")
        tabla(r.tabla)
        st.markdown("**Forma de Newton** (coeficientes = primera fila de la tabla):")
        st.latex(f"P(x) = {sp.latex(r.forma)}")
    else:
        st.subheader(f"Tabla de Neville evaluada en x = {X0:g}")
        tabla(r.tabla)
        st.caption("Cada columna combina dos valores de la anterior: "
                   "P = [(X − xᵢ)·P_{inferior} − (X − x_{i+j})·P_{superior}] / (x_{i+j} − xᵢ). "
                   "El resultado está en la esquina superior derecha.")

    lo, hi = min(xs + ([X0] if X0 is not None else [])), max(xs + ([X0] if X0 is not None else []))
    m = max(0.5, (hi - lo) * 0.2)
    xx = np.linspace(lo - m, hi + m, 400)
    if metodo == "Lagrange":
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for nombre, L in r.bases:
            Lf = sp.lambdify(interpolacion.X, L, "numpy")
            axs[0].plot(xx, Lf(xx) * np.ones_like(xx), "--", label=nombre)
        axs[0].set_title("Funciones base de Lagrange")
        axs[0].legend()
        axs[0].grid(True, alpha=0.4)
        ax = axs[1]
    else:
        fig, ax = figura()
    ax.plot(xx, r.evaluar(xx), linewidth=2, label="P(x)")
    if fnum is not None:
        ax.plot(xx, fnum(xx), ":", color="gray", label=f"f(x) = {ftxt}")
    ax.scatter(xs, ys, color="red", s=60, zorder=5, label="Puntos dados")
    if r.valor is not None:
        ax.scatter([X0], [r.valor], color="green", s=90, zorder=6, label=f"P({X0:g}) = {r.valor:.4f}")
        ax.axvline(X0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Interpolación de {metodo.split(' ')[0]}")
    ax.grid(True, alpha=0.4)
    ax.legend()
    mostrar(fig)


# =================================================================== INTEGRACIÓN

METODOS_INTEG = ["Simpson 1/3", "Simpson 3/8", "Trapecio", "Boole", "Cuadratura de Gauss-Legendre"]


def seccion_integracion():
    metodo = st.sidebar.selectbox("Método", METODOS_INTEG)
    st.header(f"Integración numérica — {metodo}")
    st.caption("Newton-Cotes: Trapecio, Simpson 1/3, Simpson 3/8 y Boole (fórmulas compuestas). "
               "Gauss-Legendre: n puntos óptimos, exacta para polinomios de grado ≤ 2n − 1.")

    modo = "Función"
    if metodo != "Cuadratura de Gauss-Legendre":
        modo = st.radio("Datos", ["Función", "Tabla de valores (x equiespaciados)"], horizontal=True)

    if modo.startswith("Tabla"):
        xs, ys = parser.puntos(st.text_area("Puntos (x, y)", "(0,0)(1,1)(2,4)(3,9)(4,16)", height=80))
        if not st.button("Resolver", type="primary"):
            return
        r = integracion.newton_cotes_tabla(xs, ys, metodo)
        st.metric("Integral aproximada", fmt(r.valor))
        st.code(r.formula)
        tabla(r.tabla)
        fig, ax = figura()
        ax.plot(r.nodos_x, r.nodos_y, "o-", color="tab:blue")
        ax.fill_between(r.nodos_x, r.nodos_y, alpha=0.2)
        ax.set_title(f"{metodo} con datos de tabla")
        mostrar(fig)
        return

    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    texto = c1.text_input("f(x) =", "x^2/2" if metodo != "Cuadratura de Gauss-Legendre" else "sin(x^2)")
    a = num(c2.text_input("a", "2" if metodo != "Cuadratura de Gauss-Legendre" else "1"), "a")
    b = num(c3.text_input("b", "5" if metodo != "Cuadratura de Gauss-Legendre" else "2"), "b")
    if metodo == "Cuadratura de Gauss-Legendre":
        n = c4.number_input("Puntos n", 1, 20, 2)
    else:
        paso = {"Simpson 1/3": 2, "Simpson 3/8": 3, "Trapecio": 1, "Boole": 4}[metodo]
        n = c4.number_input("Subintervalos n", paso, 10000, 6 if paso <= 3 else 8, step=paso)
    expr, f, _, _ = parser.funcion_x(texto)
    st.latex(f"\\int_{{{a:g}}}^{{{b:g}}} {sp.latex(expr)}\\,dx")
    comparar = st.checkbox("Comparar con todos los métodos")

    if not st.button("Resolver", type="primary"):
        return

    if metodo == "Cuadratura de Gauss-Legendre":
        r = integracion.gauss_legendre(f, a, b, n)
    else:
        r = integracion.newton_cotes(f, a, b, n, metodo)

    ref = integracion.integral_referencia(expr, a, b, parser.X)
    c1, c2, c3 = st.columns(3)
    c1.metric("Integral aproximada", fmt(r.valor))
    if ref is not None:
        c2.metric("Valor de referencia (alta precisión)", fmt(ref))
        err = abs(ref - r.valor)
        c3.metric("Error absoluto", f"{err:.3e}",
                  f"{err / abs(ref) * 100:.4f} %" if abs(ref) > 1e-14 else None, delta_color="off")
    if expr.is_polynomial(parser.X):
        F, exacto = parser.con_tiempo_limite(integracion.integral_simbolica, expr, a, b, parser.X,
                                             segundos=5, por_defecto=(None, None))
        if F is not None:
            st.latex(f"\\text{{Exacta: }} F(x) = {sp.latex(F)},\\quad "
                     f"F({b:g}) - F({a:g}) = {sp.latex(exacto)} \\approx {float(exacto):.10g}")
    st.code(r.formula)
    if metodo == "Cuadratura de Gauss-Legendre":
        st.caption(f"Con n = {n} la regla es exacta para polinomios de grado ≤ {r.extra['exacto_hasta_grado']}.")
    tabla(r.tabla)

    m = (b - a) * 0.05
    xx = np.linspace(min(a, b) - m, max(a, b) + m, 600)
    yy = f(xx)
    fig, ax = figura()
    ax.plot(xx, yy, linewidth=2, label=f"f(x) = {texto}")
    dentro = np.linspace(min(a, b), max(a, b), 600)
    ax.fill_between(dentro, f(dentro), alpha=0.18, color="tab:blue")
    if metodo == "Trapecio":
        ax.plot(r.nodos_x, r.nodos_y, "-", color="tab:orange", linewidth=1.2, label="Trapecios")
    elif metodo.startswith("Simpson") or metodo == "Boole":
        grado = {"Simpson 1/3": 2, "Simpson 3/8": 3, "Boole": 4}[metodo]
        for k in range(0, len(r.nodos_x) - 1, grado):
            px, py = r.nodos_x[k:k + grado + 1], r.nodos_y[k:k + grado + 1]
            coef = np.polyfit(px, py, grado)
            t = np.linspace(px[0], px[-1], 60)
            ax.plot(t, np.polyval(coef, t), "--", color="tab:orange", linewidth=1.1)
        ax.plot([], [], "--", color="tab:orange", label="Polinomios locales")
    for xc in r.nodos_x:
        ax.axvline(xc, color="gray", linestyle=":", linewidth=0.7)
    ax.scatter(r.nodos_x, r.nodos_y, color="red", s=45, zorder=5,
               label="Nodos de Gauss" if metodo.startswith("Cuadratura") else "Nodos")
    ax.set_title(f"{metodo}  ≈  {r.valor:.6f}")
    ax.legend()
    mostrar(fig)

    if comparar:
        filas = []
        for nombre, paso in (("Trapecio", 1), ("Simpson 1/3", 2), ("Simpson 3/8", 3), ("Boole", 4)):
            nn = int(np.ceil(n / paso) * paso) if metodo != "Cuadratura de Gauss-Legendre" else 12
            try:
                v = integracion.newton_cotes(f, a, b, nn, nombre).valor
                filas.append({"Método": f"{nombre} (n={nn})", "Valor": v,
                              "Error abs": abs(ref - v) if ref is not None else None})
            except ERRORES as e:
                filas.append({"Método": nombre, "Valor": None, "Error abs": None})
        for k in (2, 3, 4, 5, 6):
            try:
                v = integracion.gauss_legendre(f, a, b, k).valor
                filas.append({"Método": f"Gauss-Legendre (n={k})", "Valor": v,
                              "Error abs": abs(ref - v) if ref is not None else None})
            except ERRORES:
                pass
        tabla(pd.DataFrame(filas))


# =================================================================== EDO / PVI

def seccion_edo():
    metodo = st.sidebar.selectbox("Método", list(edo.METODOS))
    st.header(f"Problemas de valor inicial — {metodo}")
    st.latex(r"\frac{dy}{dx} = f(x, y), \qquad y(x_0) = y_0")

    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
    texto = c1.text_input("f(x, y) =", "y - x^2 + 1")
    x0 = num(c2.text_input("x₀", "0"), "x₀")
    y0 = num(c3.text_input("y₀ = y(x₀)", "0.5"), "y₀")
    h = num(c4.text_input("Paso h", "0.2"), "h")
    xf = num(c5.text_input("Hallar y en x =", "1"), "x final")
    expr, f = parser.funcion_xy(texto)
    st.latex(f"y' = {sp.latex(expr)},\\quad y({x0:g}) = {y0:g},\\quad h = {h:g}")
    c1, c2 = st.columns(2)
    buscar_exacta = c1.checkbox("Buscar la solución exacta para comparar", True)
    comparar = c2.checkbox("Comparar los 4 métodos")

    if not st.button("Resolver", type="primary"):
        return

    exacta_expr, exacta = (None, None)
    if buscar_exacta:
        with st.spinner("Buscando solución exacta…"):
            exacta_expr, exacta = parser.con_tiempo_limite(edo.solucion_exacta, expr, x0, y0,
                                                           segundos=8, por_defecto=(None, None))
        if exacta_expr is not None:
            st.latex(f"\\text{{Solución exacta: }} y(x) = {sp.latex(exacta_expr)}")
        else:
            st.caption("No se encontró solución exacta en forma cerrada; se muestra solo la aproximación.")

    r = edo.resolver(f, x0, y0, h, xf, metodo, exacta)
    c1, c2 = st.columns(2)
    c1.metric(f"y({xf:g}) ≈", fmt(r.ys[-1]))
    if exacta:
        c2.metric(f"y({xf:g}) exacta", fmt(exacta(r.xs[-1])), f"error = {abs(exacta(r.xs[-1]) - r.ys[-1]):.3e}",
                  delta_color="off")
    st.caption(r.mensaje)
    tabla(r.tabla)

    fig, ax = figura()
    if exacta:
        xx = np.linspace(min(r.xs), max(r.xs), 400)
        ax.plot(xx, [exacta(v) for v in xx], linewidth=2, color="black", label="Solución exacta")
    ax.plot(r.xs, r.ys, "o--", color="red", label=metodo)
    if comparar:
        for nombre in edo.METODOS:
            if nombre != metodo:
                rr = edo.resolver(f, x0, y0, h, xf, nombre)
                ax.plot(rr.xs, rr.ys, ".-", alpha=0.7, label=nombre)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"y' = {texto}")
    ax.legend()
    mostrar(fig)

    if comparar:
        filas = []
        for nombre in edo.METODOS:
            rr = edo.resolver(f, x0, y0, h, xf, nombre)
            fila = {"Método": nombre, f"y({xf:g})": rr.ys[-1]}
            if exacta:
                fila["Error abs"] = abs(exacta(rr.xs[-1]) - rr.ys[-1])
            filas.append(fila)
        tabla(pd.DataFrame(filas))


# =================================================================== principal

SECCIONES = {
    "Raíces de ecuaciones": seccion_raices,
    "Sistemas de ecuaciones lineales": seccion_sistemas,
    "Interpolación": seccion_interpolacion,
    "Integración numérica": seccion_integracion,
    "Ecuaciones diferenciales (PVI)": seccion_edo,
}


def main():
    st.title("🧮 Métodos Numéricos")
    st.caption("Escribe cualquier ejercicio: la app calcula, muestra la tabla de iteraciones y grafica. "
               "Elige el tema y el método en el panel de la izquierda.")
    st.sidebar.title("Menú")
    tema = st.sidebar.radio("Tema", list(SECCIONES))
    st.sidebar.divider()
    try:
        SECCIONES[tema]()
    except ERRORES as error:
        st.error(str(error))
    st.sidebar.divider()
    st.sidebar.number_input("Decimales en las tablas", 2, 15, 6, key="decimales")
    st.sidebar.caption(
        "**Cómo escribir:**\n\n"
        "- potencias: `x^3` o `x**3`\n"
        "- multiplicación implícita: `2x`, `3x^2`, `2xsen(x)`\n"
        "- funciones: `sen`/`sin`, `cos`, `tan`, `exp`, `ln`/`log`, `sqrt`, `cbrt` (raíz cúbica)\n"
        "- constantes: `pi`, `e`\n"
        "- decimales con punto: `0.5`"
    )


if __name__ == "__main__":
    main()
