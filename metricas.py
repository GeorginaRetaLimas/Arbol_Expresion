"""
metricas.py
-----------
Modulo REUTILIZABLE para:
1. Generar y cargar el archivo CSV con los datos de la funcion:
       y = 6.78 + 11 * cos(7.23 * x^3)
   en el rango [-10.0, 10.0] con paso de 0.001.
2. Evaluar un ArbolExpresion contra puntos de prueba y calcular:
   - Media de y (ÿ o Y_barra): (1/n) * sum(y_i)
   - MSE:                      (1/n) * sum((y_i - y_hat_i)^2)
   - RMSE:                     sqrt(MSE)
   - NMSE / NUSE:              sum((y_i - y_hat_i)^2) / sum(y_i^2)
   - R^2:                      1 - sum((y_i - y_hat_i)^2) / sum((y_i - y_barra)^2)

No depende de ninguna interfaz grafica.
"""
import os
import numpy as np

# Expresion analitica objetivo en formato entendible por ArbolExpresion
EXPRESION_OBJETIVO = "6.78 + 11 * cos(7.23 * (x ^ 3))"
ARCHIVO_CSV_DEFAULT = "datos_funcion.csv"


def funcion_objetivo(x, c=6.78):
    """Calcula y = c + 11 * cos(7.23 * x^3) de forma vectorizada."""
    x_arr = np.asarray(x, dtype=float)
    return c + 11.0 * np.cos(7.23 * (x_arr ** 3))


def generar_csv(ruta=ARCHIVO_CSV_DEFAULT, a=-10.0, b=10.0, paso=0.001, c=6.78):
    """
    Genera el archivo CSV con los valores de la funcion:
        y = c + 11 * cos(7.23 * x^3)
    en el rango [a, b] en pasos de 'paso' (por defecto 0.001).
    """
    # Determinamos el numero de puntos con precision numerica
    num_puntos = int(round((b - a) / paso)) + 1
    x = np.round(np.linspace(a, b, num_puntos), 4)
    y = funcion_objetivo(x, c=c)

    # Guardar en formato CSV: columnas x, y
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("x,y\n")
        for xi, yi in zip(x, y):
            f.write(f"{xi:.3f},{yi:.8f}\n")

    return ruta, num_puntos


def cargar_csv(ruta=ARCHIVO_CSV_DEFAULT):
    """
    Carga los puntos (x, y) desde un archivo CSV.
    Si no existe la ruta especificada, la genera automaticamente.
    """
    if not os.path.exists(ruta):
        generar_csv(ruta)

    # Cargar saltando la cabecera
    datos = np.loadtxt(ruta, delimiter=",", skiprows=1)
    x = datos[:, 0]
    y = datos[:, 1]
    return x, y


def calcular_metricas(y_real, y_pred):
    """
    Calcula la diferencia de puntos entre y_real e y_pred siguiendo
    las formulas del pizarron:
      - ÿ (y_barra): media de y_real
      - MSE:         (1/n) * sum((y - y_hat)^2)
      - RMSE:        sqrt(MSE)
      - NMSE / NUSE: sum((y - y_hat)^2) / sum(y^2)
      - R^2:         1 - sum((y - y_hat)^2) / sum((y - y_barra)^2)

    Retorna un diccionario con los valores numericos calculados.
    """
    y = np.asarray(y_real, dtype=float)
    y_hat = np.asarray(y_pred, dtype=float)

    # Filtrar pares donde ambos sean finitos (ignora infinitos o indeterminaciones)
    mascara = np.isfinite(y) & np.isfinite(y_hat)
    n_total = len(y)
    n_validos = int(np.sum(mascara))

    if n_validos == 0:
        return {
            "n_total": n_total,
            "n_validos": 0,
            "y_barra": float(np.mean(y[np.isfinite(y)])) if np.any(np.isfinite(y)) else np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "nmse": np.nan,
            "nuse": np.nan,
            "r2": np.nan,
            "valido": False,
        }

    y_val = y[mascara]
    y_hat_val = y_hat[mascara]

    # Media de y (ÿ o y_barra)
    y_barra = float(np.mean(y_val))

    # Diferencias cuadraticas
    diff = y_val - y_hat_val
    sse = float(np.sum(diff ** 2))

    # MSE y RMSE
    mse = float(sse / n_validos)
    rmse = float(np.sqrt(mse))

    # NMSE / NUSE = sum((y - y_hat)^2) / sum(y^2)
    sum_y2 = float(np.sum(y_val ** 2))
    nmse = float(sse / sum_y2) if sum_y2 != 0 else np.nan

    # R^2 = 1 - sum((y - y_hat)^2) / sum((y - y_barra)^2)
    sst = float(np.sum((y_val - y_barra) ** 2))
    if sst != 0:
        r2 = float(1.0 - (sse / sst))
    else:
        r2 = 1.0 if sse == 0 else 0.0

    return {
        "n_total": n_total,
        "n_validos": n_validos,
        "y_barra": y_barra,
        "mse": mse,
        "rmse": rmse,
        "nmse": nmse,
        "nuse": nmse,  # En el pizarron se escribe NMSE (donde M parece U/W)
        "r2": r2,
        "valido": True,
    }


def evaluar_arbol_contra_datos(arbol, x_datos, y_datos):
    """
    Evalua el ArbolExpresion sobre el vector x_datos y calcula las metricas
    contra y_datos.
    """
    with np.errstate(all="ignore"):
        y_pred = np.asarray(arbol.evaluar(x_datos), dtype=float) * np.ones_like(x_datos)
    # Suprimir valores absurdamente desbordados
    y_pred[np.abs(y_pred) > 1e12] = np.nan
    metricas = calcular_metricas(y_datos, y_pred)
    return y_pred, metricas


if __name__ == "__main__":
    print("Generando CSV de prueba...")
    ruta, n = generar_csv()
    print(f"CSV generado en '{ruta}' con {n} puntos.")

    x, y = cargar_csv()
    print(f"Puntos cargados: {len(x)}. Rango x: [{x[0]}, {x[-1]}]")

    # Prueba con la expresion identica
    import arbol_expresion
    import biblioteca_operadores

    arbol = arbol_expresion.ArbolExpresion(EXPRESION_OBJETIVO)
    y_pred, res = evaluar_arbol_contra_datos(arbol, x, y)
    print("\n--- Metricas evaluando la expresion objetivo ---")
    print(f"ÿ (Media de y):   {res['y_barra']:.6f}")
    print(f"MSE:              {res['mse']:.8f}")
    print(f"RMSE:             {res['rmse']:.8f}")
    print(f"NMSE / NUSE:      {res['nmse']:.8f}")
    print(f"R^2:              {res['r2']:.6f}")
