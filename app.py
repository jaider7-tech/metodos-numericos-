"""
Metodos Numericos - aplicacion web
==================================
Interfaz construida con Streamlit.  Todo el calculo vive en el paquete
`nucleo`, asi que esta archivo solo se encarga de pedir datos y dibujar.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from nucleo import parser, raices, sistemas

st.set_page_config(page_title="Métodos Numéricos", page_icon="∑", layout="wide")

METODOS_RAICES = {
    "Bisección": "Divide el intervalo por la mitad conservando siempre el cambio de signo. Lento pero **nunca falla** si f(a) y f(b) tienen signos opuestos.",
    "Punto fijo": "Reescribe f(x)=0 como x=g(x) e itera. Converge si |g'(x)| < 1 cerca de la raíz.",
    "Regula Falsi": "Igual que bisección, pero en vez de partir por la mitad corta con la recta que une los extremos. Suele necesitar menos iteraciones.",
    "Newton-Raphson": "Avanza por la recta tangente. Es el más rápido (convergencia cuadrática), pero necesita la derivada y un buen punto inicial.",
    "Secante": "Newton sin derivada: aproxima la tangente con la recta que pasa por los dos últimos puntos.",
    "Comparar los 5 métodos": "Ejecuta todos los métodos sobre la misma función y compara cuántas iteraciones necesita cada uno.",
}

METODOS_SISTEMAS = {
    "Gauss-Jordan": "Método **directo**: reduce la matriz aumentada hasta la identidad. Da la solución exacta en un número fijo de operaciones.",
    "Gauss-Seidel": "Método **iterativo**: cada incógnita usa de inmediato los valores recién calculados en el mismo barrido.",
    "Gauss-Jacobi": "Método **iterativo**: todas las incógnitas se actualizan con los valores de la iteración anterior.",
    "Algoritmo de Thomas": "Solo para matrices **tridiagonales**. Resuelve en O(n) operaciones en lugar de O(n³).",
}

EJEMPLO_SISTEMA = "2x + y + z = 3\n4x + 3y + 2z = 6\n-2x + 5y + 4z = 7"


# --------------------------------------------------------------- utilidades

def columnas_aumentada(nombres):
    """Evita que la columna del termino independiente choque con una incognita
    que el usuario haya llamado justamente 'b'."""
    etiqueta = "b"
    while etiqueta in nombres:
        etiqueta += "'"
    return list(nombres) + [etiqueta]


def mostrar_tabla(tabla):
    if tabla is None or tabla.empty:
        return
    st.dataframe(tabla.style.format(precision=8), width="stretch", hide_index=True)


def grafica_funcion(f, a, b, raiz=None, titulo="f(x)"):
    xs = np.linspace(a, b, 600)
    ys = parser.evaluar(f, xs)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xs, ys, linewidth=2, label="f(x)")
    ax.axhline(0, color="black", linewidth=0.8)
    if raiz is not None and np.isfinite(raiz):
        ax.plot([raiz], [0], "o", markersize=9, color="crimson", zorder=5,
                label=f"raíz ≈ {raiz:.6f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def grafica_convergencia(series, titulo="Convergencia del error"):
    """series: dict nombre -> lista de errores por iteración."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hubo_datos = False
    for nombre, errores in series.items():
        errores = [e for e in errores if e is not None and np.isfinite(e) and e > 0]
        if errores:
            ax.semilogy(range(1, len(errores) + 1), errores, marker="o",
                        markersize=4, label=nombre)
            hubo_datos = True
    if not hubo_datos:
        plt.close(fig)
        return None
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error (escala logarítmica)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    return fig


def resumen(resultado, etiqueta="Raíz aproximada"):
    columnas = st.columns(3)
    valor = resultado.raiz if hasattr(resultado, "raiz") else None
    columnas[0].metric(etiqueta, f"{valor:.8f}" if valor is not None else "—")
    columnas[1].metric("Iteraciones", resultado.iteraciones)
    columnas[2].metric("¿Convergió?", "Sí" if resultado.convergio else "No")
    (st.success if resultado.convergio else st.warning)(resultado.mensaje)


# ------------------------------------------------------------------ raíces

def seccion_raices():
    metodo = st.sidebar.selectbox("Método", list(METODOS_RAICES))
    st.header(metodo)
    st.caption(METODOS_RAICES[metodo])

    es_punto_fijo = metodo == "Punto fijo"
    etiqueta = "g(x) — la ecuación reescrita como x = g(x)" if es_punto_fijo else "f(x) — se busca f(x) = 0"
    valor_inicial = "cos(x)" if es_punto_fijo else "x^3 - x - 1"

    texto = st.text_input(etiqueta, valor_inicial,
                          help="Puedes escribir x^2, 2x, sin(x), exp(x), sqrt(x), log(x), pi...")

    try:
        f, expresion = parser.funcion(texto)
    except parser.ErrorDeEntrada as error:
        st.error(str(error))
        return

    st.latex(("g(x) = " if es_punto_fijo else "f(x) = ") + parser.bonito(expresion))

    usa_intervalo = metodo in ("Bisección", "Regula Falsi", "Comparar los 5 métodos")
    columnas = st.columns(4)

    if usa_intervalo:
        a = columnas[0].number_input("a (extremo izquierdo)", value=0.0, format="%.6f")
        b = columnas[1].number_input("b (extremo derecho)", value=2.0, format="%.6f")
    elif metodo == "Secante":
        a = columnas[0].number_input("x₀", value=1.0, format="%.6f")
        b = columnas[1].number_input("x₁", value=2.0, format="%.6f")
    else:
        a = columnas[0].number_input("x₀ (valor inicial)", value=1.0, format="%.6f")
        b = a + 1

    tolerancia = columnas[2].number_input("Tolerancia", value=1e-6, format="%.10f", min_value=0.0)
    max_iter = columnas[3].number_input("Máx. iteraciones", value=50, min_value=1, max_value=1000, step=1)

    if not st.button("Resolver", type="primary", width="stretch"):
        return

    # ---- comparación de los cinco métodos
    if metodo == "Comparar los 5 métodos":
        comparar(f, expresion, a, b, tolerancia, max_iter)
        return

    # ---- un solo método
    try:
        if metodo == "Bisección":
            resultado = raices.biseccion(f, a, b, tolerancia, max_iter)
        elif metodo == "Regula Falsi":
            resultado = raices.regula_falsi(f, a, b, tolerancia, max_iter)
        elif metodo == "Punto fijo":
            resultado = raices.punto_fijo(f, a, tolerancia, max_iter)
        elif metodo == "Secante":
            resultado = raices.secante(f, a, b, tolerancia, max_iter)
        else:
            df, derivada = parser.derivada(expresion)
            st.latex("f'(x) = " + parser.bonito(derivada) + r"\quad\text{(derivada calculada automáticamente)}")
            resultado = raices.newton_raphson(f, df, a, tolerancia, max_iter)
    except raices.ErrorDeMetodo as error:
        st.error(str(error))
        return

    resumen(resultado)
    st.subheader("Tabla de iteraciones")
    mostrar_tabla(resultado.tabla)

    izquierda, derecha = st.columns(2)
    with izquierda:
        centro = resultado.raiz if resultado.raiz is not None else a
        inicio, fin = (a, b) if usa_intervalo else (centro - 3, centro + 3)
        titulo = "g(x) y la recta y = x" if es_punto_fijo else "Ubicación de la raíz"
        figura = grafica_funcion(f, inicio, fin, None if es_punto_fijo else resultado.raiz, titulo)
        if es_punto_fijo:
            eje = figura.axes[0]
            xs = np.linspace(inicio, fin, 200)
            eje.plot(xs, xs, "--", color="gray", linewidth=1.2, label="y = x")
            if resultado.raiz is not None:
                eje.plot([resultado.raiz], [resultado.raiz], "o", color="crimson",
                         markersize=9, label=f"punto fijo ≈ {resultado.raiz:.6f}")
            eje.legend()
        st.pyplot(figura)
    with derecha:
        figura = grafica_convergencia({metodo: list(resultado.tabla.get("Error", []))})
        if figura:
            st.pyplot(figura)

    st.download_button("Descargar la tabla en CSV",
                       resultado.tabla.to_csv(index=False).encode("utf-8"),
                       file_name="iteraciones.csv", mime="text/csv")


def comparar(f, expresion, a, b, tolerancia, max_iter):
    df, _ = parser.derivada(expresion)
    medio = (a + b) / 2
    intentos = {
        "Bisección": lambda: raices.biseccion(f, a, b, tolerancia, max_iter),
        "Regula Falsi": lambda: raices.regula_falsi(f, a, b, tolerancia, max_iter),
        "Newton-Raphson": lambda: raices.newton_raphson(f, df, medio, tolerancia, max_iter),
        "Secante": lambda: raices.secante(f, a, b, tolerancia, max_iter),
        "Punto fijo (g = x - f(x))": lambda: raices.punto_fijo(
            lambda x: x - f(x), medio, tolerancia, max_iter),
    }

    filas, series = [], {}
    for nombre, ejecutar in intentos.items():
        try:
            resultado = ejecutar()
            filas.append({
                "Método": nombre,
                "Raíz": resultado.raiz,
                "Iteraciones": resultado.iteraciones,
                "¿Convergió?": "Sí" if resultado.convergio else "No",
                "Error final": resultado.tabla["Error"].iloc[-1] if not resultado.tabla.empty else np.nan,
            })
            if resultado.convergio:
                series[nombre] = list(resultado.tabla["Error"])
        except raices.ErrorDeMetodo as error:
            filas.append({"Método": nombre, "Raíz": None, "Iteraciones": 0,
                          "¿Convergió?": "No aplica", "Error final": np.nan})
            st.info(f"**{nombre}**: {error}")

    tabla = pd.DataFrame(filas)
    st.subheader("Comparación")
    mostrar_tabla(tabla)

    validos = tabla[tabla["¿Convergió?"] == "Sí"]
    if not validos.empty:
        ganador = validos.loc[validos["Iteraciones"].idxmin()]
        st.success(
            f"El método más rápido fue **{ganador['Método']}** "
            f"con {int(ganador['Iteraciones'])} iteraciones."
        )

    izquierda, derecha = st.columns(2)
    with izquierda:
        raiz = validos["Raíz"].iloc[0] if not validos.empty else None
        st.pyplot(grafica_funcion(f, a, b, raiz, "Función analizada"))
    with derecha:
        figura = grafica_convergencia(series, "Velocidad de convergencia de cada método")
        if figura:
            st.pyplot(figura)
        st.caption("Entre más inclinada la recta, más rápido converge el método.")


# ---------------------------------------------------------------- sistemas

def leer_sistema():
    """Devuelve (A, b, nombres) desde texto o desde una tabla editable."""
    modo = st.radio("¿Cómo quieres ingresar el sistema?",
                    ["Escribir las ecuaciones", "Llenar la tabla de coeficientes"],
                    horizontal=True)

    if modo == "Escribir las ecuaciones":
        texto = st.text_area("Una ecuación por línea", EJEMPLO_SISTEMA, height=140,
                             help="Ejemplo: 2x + y - 3z = 5. Las incógnitas pueden llamarse como quieras.")
        return parser.sistema_desde_texto(texto)

    n = st.slider("Número de ecuaciones e incógnitas", 2, 8, 3)
    nombres = [f"x{i + 1}" for i in range(n)]
    base = pd.DataFrame(np.zeros((n, n + 1)), columns=nombres + ["b"])
    for i in range(n):
        base.iloc[i, i] = 1.0
    editada = st.data_editor(base, num_rows="fixed", width="stretch",
                             key="tabla_sistema")
    valores = editada.to_numpy(dtype=float)
    return valores[:, :n], valores[:, n], nombres


def seccion_sistemas():
    metodo = st.sidebar.selectbox("Método", list(METODOS_SISTEMAS))
    st.header(metodo)
    st.caption(METODOS_SISTEMAS[metodo])

    try:
        A, b, nombres = leer_sistema()
    except parser.ErrorDeEntrada as error:
        st.error(str(error))
        return

    with st.expander("Sistema interpretado por el programa"):
        st.code(parser.sistema_a_texto(A, b, nombres), language="text")
        st.dataframe(pd.DataFrame(np.column_stack([A, b]), columns=columnas_aumentada(nombres)),
                     width="stretch", hide_index=True)

    iterativo = metodo in ("Gauss-Seidel", "Gauss-Jacobi")
    if iterativo:
        columnas = st.columns(3)
        inicial = columnas[0].text_input("Valores iniciales (separados por coma)",
                                         ", ".join("0" for _ in nombres))
        tolerancia = columnas[1].number_input("Tolerancia", value=1e-6, format="%.10f", min_value=0.0)
        max_iter = columnas[2].number_input("Máx. iteraciones", value=100, min_value=1,
                                            max_value=2000, step=1)
        try:
            x0 = [float(v) for v in inicial.replace(";", ",").split(",")]
            if len(x0) != len(nombres):
                raise ValueError
        except ValueError:
            st.error(f"Debes escribir exactamente {len(nombres)} valores iniciales.")
            return
    else:
        x0, tolerancia, max_iter = None, 1e-6, 100

    if not st.button("Resolver", type="primary", width="stretch"):
        return

    try:
        if metodo == "Gauss-Jordan":
            resultado = sistemas.gauss_jordan(A, b)
        elif metodo == "Gauss-Seidel":
            resultado = sistemas.gauss_seidel(A, b, x0, tolerancia, max_iter)
        elif metodo == "Gauss-Jacobi":
            resultado = sistemas.gauss_jacobi(A, b, x0, tolerancia, max_iter)
        else:
            resultado = sistemas.thomas(A, b)
    except sistemas.ErrorDeMetodo as error:
        st.error(str(error))
        return

    for aviso in resultado.avisos:
        st.info(aviso)

    if resultado.solucion is not None and resultado.convergio:
        st.subheader("Solución")
        for columna, nombre, valor in zip(st.columns(len(nombres)), nombres, resultado.solucion):
            columna.metric(nombre, f"{valor:.8f}")
        st.success(resultado.mensaje)
    else:
        st.warning(resultado.mensaje)

    if not resultado.tabla.empty:
        st.subheader("Proceso" if metodo == "Algoritmo de Thomas" else "Tabla de iteraciones")
        mostrar_tabla(resultado.tabla)
        st.download_button("Descargar la tabla en CSV",
                           resultado.tabla.to_csv(index=False).encode("utf-8"),
                           file_name="proceso.csv", mime="text/csv")

    if resultado.historial:
        figura = grafica_convergencia({metodo: resultado.historial})
        if figura:
            st.pyplot(figura)

    if resultado.pasos:
        st.subheader("Operaciones elementales paso a paso")
        for titulo, matriz in resultado.pasos:
            st.markdown(f"**{titulo}**")
            st.dataframe(pd.DataFrame(matriz, columns=columnas_aumentada(nombres)).style.format(precision=6),
                         width="stretch", hide_index=True)


# --------------------------------------------------------------------- app

def main():
    st.title("∑  Métodos Numéricos")
    st.markdown(
        "Resuelve **cualquier** ecuación o sistema lineal que escribas. "
        "Elige el tema y el método en el panel de la izquierda."
    )
    st.sidebar.title("Menú")
    tema = st.sidebar.radio("Tema", ["Raíces de ecuaciones", "Sistemas de ecuaciones lineales"])
    st.sidebar.divider()

    if tema == "Raíces de ecuaciones":
        seccion_raices()
    else:
        seccion_sistemas()

    st.sidebar.caption(
        "Cómo escribir:\n\n"
        "- potencias: `x^3` o `x**3`\n"
        "- multiplicación implícita: `2x`, `3x^2`\n"
        "- funciones: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`\n"
        "- constantes: `pi`, `E`"
    )


if __name__ == "__main__":
    main()
