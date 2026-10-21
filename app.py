"""
app.py  -  Interfaz gráfica (Tkinter) que integra todos los módulos.
Ejecutar:  python app.py
Instalar:  pip install numpy matplotlib

Paleta:    Rosita bonita, suave, elegante y digerible (no chillona).
Árbol:     Visualización binaria pura (hijos izq/der, centrado, funciones unarias verticales).
Evaluación:Muestra métricas de error (RMSE, MSE, NUSE/NMSE, R², ÿ) ÚNICAMENTE
           al presionar el botón 'Evaluar con CSV'.
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from arbol_expresion import ArbolExpresion
from generador_aleatorio import generar_graficable, altura
from visualizacion import (
    ROSA, ROSA_CLARO, ROSA_MEDIO, ROSA_BORDE, BLANCO, NEGRO, TEXTO_MUTED,
    dibujar_arbol, dibujar_funcion
)
from metricas import (
    EXPRESION_OBJETIVO, ARCHIVO_CSV_DEFAULT,
    cargar_csv, evaluar_arbol_contra_datos, generar_csv
)

# Paleta estética rosita suave y descansada
COLOR_FONDO_APP = "#faf3f7"         # Fondo suave aperlado
COLOR_TARJETA = "#ffffff"           # Tarjetas blancas limpias
COLOR_BORDE_TARJETA = "#edd4e2"     # Borde pastel suave
COLOR_HEADER = "#f8e1ec"            # Banner superior rosa pastel
COLOR_TITULO = "#381a2c"            # Ciruela profundo para títulos principales
COLOR_SUBTITULO = "#754f66"         # Ciruela medio para subtítulos y etiquetas
COLOR_BTN_PRIMARIO = "#d94a7a"      # Rosa frambuesa principal
COLOR_BTN_PRIMARIO_HOVER = "#c53b6b"# Hover rosa intenso
COLOR_BTN_SECUNDARIO = "#faebf2"    # Botón rosa pastel claro
COLOR_BTN_SECUNDARIO_TX = "#a83260" # Texto de botón secundario
COLOR_BTN_EVALUAR = "#993d6b"       # Ciruela/frambuesa oscuro para botón de evaluación
COLOR_BTN_EVALUAR_HOVER = "#852e59" # Hover botón evaluar
COLOR_METRICA_BG = "#fcf2f6"        # Fondo para cajitas de métricas
COLOR_METRICA_BORDE = "#ebd0df"     # Borde para cajitas de métricas


class App:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("🌸 Árbol de Expresión Binario • Interfaz Rosa & Métricas")
        self.ventana.geometry("1240x840")
        self.ventana.minsize(1050, 720)
        self.ventana.configure(bg=COLOR_FONDO_APP)

        # Estado del CSV y evaluación
        self.ruta_csv = ARCHIVO_CSV_DEFAULT
        self.x_csv = None
        self.y_csv = None
        self._cargar_datos_csv()

        # Bandera: solo se muestra la evaluación de error cuando se evalúa con el CSV
        self.evaluado_con_csv = False
        self.arbol_actual = None

        self._crear_interfaz()
        self.graficar()

    def _cargar_datos_csv(self):
        """Carga o genera los datos del CSV para el cálculo de métricas."""
        try:
            if not os.path.exists(self.ruta_csv):
                generar_csv(self.ruta_csv)
            self.x_csv, self.y_csv = cargar_csv(self.ruta_csv)
        except Exception as e:
            self.x_csv, self.y_csv = None, None
            print(f"Aviso al cargar CSV: {e}")

    def _crear_interfaz(self):
        # =====================================================================
        # 1. ENCABEZADO ESTÉTICO
        # =====================================================================
        header = tk.Frame(self.ventana, bg=COLOR_HEADER, pady=10, padx=16,
                          highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        header.pack(fill="x", padx=12, pady=(10, 6))

        titulo_box = tk.Frame(header, bg=COLOR_HEADER)
        titulo_box.pack(side="left")

        tk.Label(titulo_box, text="🌸 Árbol de Expresión Binario",
                 bg=COLOR_HEADER, fg=COLOR_TITULO,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(titulo_box,
                 text="Modelado simbólico jerárquico, visualización binaria y evaluación de error con CSV",
                 bg=COLOR_HEADER, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9)).pack(anchor="w")

        # Botones rápidos en el encabezado
        hdr_actions = tk.Frame(header, bg=COLOR_HEADER)
        hdr_actions.pack(side="right")

        btn_obj = tk.Button(hdr_actions, text="🎯 Cargar Función Objetivo (CSV)",
                            bg=COLOR_TARJETA, fg=COLOR_BTN_PRIMARIO,
                            activebackground="#faebf2", activeforeground=COLOR_BTN_PRIMARIO,
                            font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                            highlightbackground=COLOR_BTN_PRIMARIO, highlightthickness=1,
                            padx=10, pady=4, command=self.cargar_expresion_objetivo)
        btn_obj.pack(side="left", padx=4)

        # =====================================================================
        # 2. PANEL DE CONTROL (ENTRADAS Y BOTONES)
        # =====================================================================
        panel_control = tk.Frame(self.ventana, bg=COLOR_TARJETA, padx=16, pady=10,
                                 highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        panel_control.pack(fill="x", padx=12, pady=4)

        # Fila 1: Entrada de expresión
        f1 = tk.Frame(panel_control, bg=COLOR_TARJETA)
        f1.pack(fill="x", pady=(2, 6))

        tk.Label(f1, text="Expresión f(x):", bg=COLOR_TARJETA, fg=COLOR_TITULO,
                 font=("Segoe UI", 10, "bold"), width=13, anchor="w").pack(side="left")

        self.entrada_expr = tk.Entry(f1, font=("Consolas", 11),
                                     bg="#fffafc", fg=NEGRO,
                                     highlightbackground=COLOR_BORDE_TARJETA,
                                     highlightcolor=COLOR_BTN_PRIMARIO,
                                     highlightthickness=1, relief="flat")
        self.entrada_expr.insert(0, EXPRESION_OBJETIVO)
        self.entrada_expr.pack(side="left", fill="x", expand=True, padx=(4, 10), ipady=4)

        # Fila 2: Parámetros y botones
        f2 = tk.Frame(panel_control, bg=COLOR_TARJETA)
        f2.pack(fill="x", pady=2)

        tk.Label(f2, text="Intervalo [a, b]:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_int = tk.Entry(f2, width=12, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_int.insert(0, "-10, 10")
        self.entrada_int.pack(side="left", padx=(6, 16), ipady=2)

        tk.Label(f2, text="Altura máx:", bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        self.entrada_alt = tk.Entry(f2, width=5, font=("Segoe UI", 10),
                                    bg="#fffafc", fg=NEGRO, justify="center",
                                    highlightbackground=COLOR_BORDE_TARJETA,
                                    highlightthickness=1, relief="flat")
        self.entrada_alt.insert(0, "4")
        self.entrada_alt.pack(side="left", padx=(6, 20), ipady=2)

        # Botones principales
        btn_prim = dict(bg=COLOR_BTN_PRIMARIO, fg=BLANCO,
                        activebackground=COLOR_BTN_PRIMARIO_HOVER, activeforeground=BLANCO,
                        font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=14, pady=4)

        btn_sec = dict(bg=COLOR_BTN_SECUNDARIO, fg=COLOR_BTN_SECUNDARIO_TX,
                       activebackground="#f3d3e2", activeforeground=COLOR_BTN_SECUNDARIO_TX,
                       font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2",
                       highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1,
                       padx=12, pady=4)

        btn_eval = dict(bg=COLOR_BTN_EVALUAR, fg=BLANCO,
                        activebackground=COLOR_BTN_EVALUAR_HOVER, activeforeground=BLANCO,
                        font=("Segoe UI", 9, "bold"), relief="flat", cursor="hand2", padx=14, pady=4)

        tk.Button(f2, text="📈 Graficar", command=self.graficar, **btn_prim).pack(side="left", padx=4)
        tk.Button(f2, text="🎲 Aleatoria", command=self.aleatoria, **btn_sec).pack(side="left", padx=4)

        # Separador visual sutil entre graficar y evaluación
        sep = tk.Frame(f2, bg=COLOR_BORDE_TARJETA, width=1, height=22)
        sep.pack(side="left", padx=8, fill="y")

        # BOTÓN DE EVALUACIÓN (solicitado expresamente por el usuario)
        tk.Button(f2, text="📊 Evaluar con CSV", command=self.evaluar_con_csv, **btn_eval).pack(side="left", padx=4)
        tk.Button(f2, text="📁 Seleccionar CSV...", command=self.seleccionar_csv, **btn_sec).pack(side="left", padx=4)

        # =====================================================================
        # 3. CARD DE INFORMACIÓN DEL ÁRBOL
        # =====================================================================
        self.card_info = tk.Frame(self.ventana, bg=COLOR_TARJETA, padx=14, pady=8,
                                  highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        self.card_info.pack(fill="x", padx=12, pady=4)

        # Info del árbol: Altura y Preorden
        info_top = tk.Frame(self.card_info, bg=COLOR_TARJETA)
        info_top.pack(fill="x")

        self.lbl_altura = tk.Label(info_top, text="Altura: -",
                                   bg=COLOR_TARJETA, fg=COLOR_TITULO,
                                   font=("Segoe UI", 10, "bold"))
        self.lbl_altura.pack(side="left")

        self.lbl_preorden = tk.Label(info_top, text="Preorden: -",
                                     bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
                                     font=("Consolas", 9), anchor="w")
        self.lbl_preorden.pack(side="left", padx=(20, 0), fill="x", expand=True)

        # Estado informativo de evaluación (cuando aún no se evalúa)
        self.lbl_estado_eval = tk.Label(
            info_top,
            text="💡 Presiona '📊 Evaluar con CSV' para calcular el error (RMSE, MSE, NUSE, R², ÿ)",
            bg=COLOR_TARJETA, fg=COLOR_SUBTITULO, font=("Segoe UI", 8, "italic")
        )
        self.lbl_estado_eval.pack(side="right")

        # =====================================================================
        # 4. PANEL DE MÉTRICAS (SOLO SE MUESTRA CUANDO SE EVALÚA CON CSV)
        # =====================================================================
        self.frame_metricas_container = tk.Frame(self.card_info, bg=COLOR_TARJETA)
        # Por defecto NO se empaqueta hasta que el usuario presione "Evaluar con CSV"

        # Encabezado del panel de métricas con botón para ocultarlo
        hdr_met = tk.Frame(self.frame_metricas_container, bg=COLOR_TARJETA)
        hdr_met.pack(fill="x", pady=(6, 4))

        self.lbl_titulo_metricas = tk.Label(
            hdr_met,
            text="📊 Evaluación de Error contra Dataset CSV:",
            bg=COLOR_TARJETA, fg=COLOR_TITULO, font=("Segoe UI", 9, "bold")
        )
        self.lbl_titulo_metricas.pack(side="left")

        btn_cerrar_met = tk.Button(
            hdr_met, text="✕ Ocultar evaluación",
            bg=COLOR_TARJETA, fg=COLOR_SUBTITULO,
            activebackground="#faebf2", activeforeground=COLOR_TITULO,
            font=("Segoe UI", 8), relief="flat", cursor="hand2",
            command=self.ocultar_evaluacion
        )
        btn_cerrar_met.pack(side="right")

        # Cajitas individuales para cada métrica
        self.frame_cajas = tk.Frame(self.frame_metricas_container, bg=COLOR_TARJETA)
        self.frame_cajas.pack(fill="x")

        self.widget_metricas = {}
        metricas_def = [
            ("ÿ (Media de y)", "lbl_y_barra", "Media de los datos reales y"),
            ("MSE",            "lbl_mse",     "Error Cuadrático Medio"),
            ("RMSE",           "lbl_rmse",    "Raíz del Error Cuadrático Medio"),
            ("NUSE / NMSE",    "lbl_nmse",    "Error Cuadrático Normalizado (sum|y-y_hat|^2 / sum|y|^2)"),
            ("R² (Ajuste)",    "lbl_r2",      "Coeficiente de Determinación"),
            ("Puntos CSV",     "lbl_puntos",  "Puntos evaluados del dataset"),
        ]

        for nombre, clave, desc in metricas_def:
            caja = tk.Frame(self.frame_cajas, bg=COLOR_METRICA_BG, padx=8, pady=4,
                            highlightbackground=COLOR_METRICA_BORDE, highlightthickness=1)
            caja.pack(side="left", fill="both", expand=True, padx=3)

            tk.Label(caja, text=nombre, bg=COLOR_METRICA_BG, fg=COLOR_SUBTITULO,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w")
            lbl_val = tk.Label(caja, text="---", bg=COLOR_METRICA_BG, fg=COLOR_TITULO,
                               font=("Consolas", 10, "bold"))
            lbl_val.pack(anchor="w")
            self.widget_metricas[clave] = lbl_val

        # =====================================================================
        # 5. ÁREA DE GRÁFICAS MATPLOTLIB
        # =====================================================================
        frame_canvas = tk.Frame(self.ventana, bg=COLOR_TARJETA,
                                highlightbackground=COLOR_BORDE_TARJETA, highlightthickness=1)
        frame_canvas.pack(fill="both", expand=True, padx=12, pady=(4, 10))

        self.fig = Figure(figsize=(12, 5.5), facecolor=BLANCO)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_canvas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=2, pady=2)

    # =========================================================================
    # LECTURA DE ENTRADAS Y ACCIONES
    # =========================================================================
    def _intervalo(self):
        txt = self.entrada_int.get().strip()
        partes = txt.split(",")
        if len(partes) != 2:
            raise ValueError("El intervalo debe tener el formato a, b (ejemplo: -10, 10)")
        a, b = float(partes[0]), float(partes[1])
        if a >= b:
            raise ValueError("En el intervalo [a, b], 'a' debe ser menor que 'b'")
        return a, b

    def _altura_max(self):
        val = int(self.entrada_alt.get().strip())
        if val < 0 or val > 12:
            raise ValueError("La altura máxima debe estar entre 0 y 12")
        return val

    def cargar_expresion_objetivo(self):
        """Carga la expresion objetivo del CSV y restaura el intervalo [-10, 10]."""
        self.entrada_expr.delete(0, "end")
        self.entrada_expr.insert(0, EXPRESION_OBJETIVO)
        self.entrada_int.delete(0, "end")
        self.entrada_int.insert(0, "-10, 10")
        self.evaluado_con_csv = False
        self.frame_metricas_container.pack_forget()
        self.lbl_estado_eval.pack(side="right")
        self.graficar()

    def seleccionar_csv(self):
        """Permite al usuario seleccionar cualquier archivo CSV compatible."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar archivo CSV con datos (x, y)",
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if ruta:
            try:
                x, y = cargar_csv(ruta)
                self.ruta_csv = ruta
                self.x_csv, self.y_csv = x, y
                messagebox.showinfo("CSV cargado",
                                    f"Se cargaron exitosamente {len(x):,} puntos desde:\n{os.path.basename(ruta)}")
                # Si ya estaba evaluado, reevalúa con el nuevo CSV
                if self.evaluado_con_csv:
                    self.evaluar_con_csv()
            except Exception as e:
                messagebox.showerror("Error al cargar CSV", f"No se pudo leer el archivo:\n{e}")

    def ocultar_evaluacion(self):
        """Oculta el panel de métricas y quita la curva de referencia de la gráfica."""
        self.evaluado_con_csv = False
        self.frame_metricas_container.pack_forget()
        self.lbl_estado_eval.pack(side="right")
        if self.arbol_actual:
            self.mostrar(self.arbol_actual)

    def graficar(self):
        """
        Grafica la expresión actual.
        Al graficar una nueva expresión sin evaluar, la evaluación de error se oculta
        (cumpliendo la regla: solo mostrar la evaluación al evaluar con el CSV).
        """
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión matemática")
            self.arbol_actual = ArbolExpresion(expr)
            self.evaluado_con_csv = False
            self.frame_metricas_container.pack_forget()
            self.lbl_estado_eval.pack(side="right")
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error de Expresión", f"Revisa tus datos:\n{error}")

    def aleatoria(self):
        """Genera una función aleatoria y la grafica (sin evaluación de error previa)."""
        try:
            a, b = self._intervalo()
            self.arbol_actual = generar_graficable(self._altura_max(), a, b)
            self.entrada_expr.delete(0, "end")
            self.entrada_expr.insert(0, self.arbol_actual.expresion)
            self.evaluado_con_csv = False
            self.frame_metricas_container.pack_forget()
            self.lbl_estado_eval.pack(side="right")
            self.mostrar(self.arbol_actual)
        except Exception as error:
            messagebox.showerror("Error", f"Revisa tus datos:\n{error}")

    def evaluar_con_csv(self):
        """
        SOLO al presionar este botón se activa y muestra la evaluación de error:
        Calcula RMSE, MSE, NUSE, R^2 y ÿ contra los datos del CSV y muestra la tarjeta.
        """
        try:
            expr = self.entrada_expr.get().strip()
            if not expr:
                raise ValueError("Por favor escribe una expresión matemática antes de evaluar")

            self.arbol_actual = ArbolExpresion(expr)

            if self.x_csv is None or self.y_csv is None:
                self._cargar_datos_csv()
                if self.x_csv is None:
                    raise ValueError("No se pudo cargar el archivo CSV para evaluar")

            # 1. Calcular métricas usando el módulo modular metricas.py
            _, met = evaluar_arbol_contra_datos(self.arbol_actual, self.x_csv, self.y_csv)

            if met["valido"]:
                self.widget_metricas["lbl_y_barra"].config(text=f"{met['y_barra']:.4f}")
                self.widget_metricas["lbl_mse"].config(text=f"{met['mse']:.6f}")
                self.widget_metricas["lbl_rmse"].config(text=f"{met['rmse']:.6f}")
                self.widget_metricas["lbl_nmse"].config(text=f"{met['nmse']:.6f}")
                self.widget_metricas["lbl_r2"].config(text=f"{met['r2']:.4f}")
                self.widget_metricas["lbl_puntos"].config(text=f"{met['n_validos']:,} / {met['n_total']:,}")
            else:
                for key in self.widget_metricas:
                    self.widget_metricas[key].config(text="Indefinido")

            # 2. Actualizar título del panel con el nombre del archivo
            nom_csv = os.path.basename(self.ruta_csv)
            self.lbl_titulo_metricas.config(
                text=f"📊 Evaluación de Error contra '{nom_csv}' ({len(self.x_csv):,} puntos):"
            )

            # 3. Mostrar el panel de métricas en la interfaz
            self.lbl_estado_eval.pack_forget()
            self.frame_metricas_container.pack(fill="x", pady=(4, 0))

            # 4. Activar bandera de comparación en gráfica y redibujar
            self.evaluado_con_csv = True
            self.mostrar(self.arbol_actual)

        except Exception as error:
            messagebox.showerror("Error al Evaluar con CSV", f"No se pudo completar la evaluación:\n{error}")

    # =========================================================================
    # DIBUJO
    # =========================================================================
    def mostrar(self, arbol):
        a, b = self._intervalo()
        h, h_max = altura(arbol.raiz), self._altura_max()

        # 1. Limpiar figura de matplotlib
        self.fig.clear()
        ax_arbol = self.fig.add_subplot(1, 2, 1)
        ax_func = self.fig.add_subplot(1, 2, 2)

        # 2. Dibujar árbol binario mejorado
        dibujar_arbol(ax_arbol, arbol)

        # 3. Datos de referencia para f(x): SOLO si se ha evaluado con el CSV
        datos_ref_plot = None
        if self.evaluado_con_csv and self.x_csv is not None and self.y_csv is not None:
            datos_ref_plot = (self.x_csv, self.y_csv)

        dibujar_funcion(ax_func, arbol, a, b, datos_ref=datos_ref_plot)

        self.fig.tight_layout()
        self.canvas.draw()

        # 4. Actualizar etiquetas de altura y preorden
        aviso = "  ⚠️ (excede altura máx)" if h > h_max else ""
        self.lbl_altura.config(text=f"Altura: {h} / {h_max}{aviso}")
        tokens_pre = "  ".join(arbol.preorden())
        if len(tokens_pre) > 95:
            tokens_pre = tokens_pre[:92] + "..."
        self.lbl_preorden.config(text=f"Preorden: {tokens_pre}")


if __name__ == "__main__":
    ventana = tk.Tk()
    app = App(ventana)
    ventana.mainloop()
