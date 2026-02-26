import os
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import fsolve
from scipy.integrate import simpson, quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from functools import partial

# ==============================================================================
# CLASE 1: BASE DE DATOS AERODINAMICOS DE ANALISIS 2D 
# ==============================================================================
class BaseDatosAerodinamica:
    def __init__(self, carpeta, configuracion_archivos):
        self.carpeta = carpeta
        self.archivos = configuracion_archivos
        self.interpolador_cl = None
        self.interpolador_cd = None
        self.interpolador_cm = None
        self.limite_min = 0.0
        self.limite_max = 0.0
        self._cargar_y_procesar_datos()

    def _procesar_polar_individual(self, ruta_archivo, re_val, ang_val):
        try:
            df = pd.read_csv(
                ruta_archivo, sep=r'\s+', header=None, skiprows=11, 
                names=["alpha", "CL", "CD", "Cm", "Xcp", "CDp"], 
                usecols=range(6), engine='python', on_bad_lines='skip'
            ).apply(pd.to_numeric, errors='coerce').dropna()

            return np.column_stack((
                np.full(len(df), re_val), df["alpha"].to_numpy(), np.full(len(df), ang_val),
                df["CL"].to_numpy(), df["CD"].to_numpy(), df["Cm"].to_numpy()
            )).tolist()
        except Exception as e:
            print(f"error critico procesando {os.path.basename(ruta_archivo)}: {e}")
            return []

    def _cargar_y_procesar_datos(self):
        print("iniciando carga de polares aerodinamicas")
        matriz_datos_global = []
        for nombre, valores in self.archivos.items():
            ruta_full = os.path.join(self.carpeta, nombre)
            if os.path.exists(ruta_full):
                matriz_datos_global.extend(self._procesar_polar_individual(ruta_full, valores[0], valores[1]))
            else:
                print(f"archivo no encontrado: {ruta_full}")

        if not matriz_datos_global:
            raise ValueError("no se ha podido cargar ningun dato.")

        datos_array = np.array(matriz_datos_global)
        puntos_entrada = datos_array[:, [1, 2]] 
        valores_alpha = datos_array[:, 1]

        self.limite_min = np.min(valores_alpha)
        self.limite_max = np.max(valores_alpha)
        
        self.interpolador_cl = LinearNDInterpolator(puntos_entrada, datos_array[:, 3], rescale=True)
        self.interpolador_cd = LinearNDInterpolator(puntos_entrada, datos_array[:, 4], rescale=True)
        self.interpolador_cm = LinearNDInterpolator(puntos_entrada, datos_array[:, 5], rescale=True)
        print(f"base de datos lista (AOA Min {self.limite_min}º | Max {self.limite_max}º)\n")

    def consultar_coeficientes(self, alpha_deg, flap_deg):
        alpha_seguro = np.clip(alpha_deg, self.limite_min, self.limite_max)
        cl = np.nan_to_num(self.interpolador_cl(alpha_seguro, flap_deg), nan=0.0)
        cd = np.nan_to_num(self.interpolador_cd(alpha_seguro, flap_deg), nan=0.0)
        cm = np.nan_to_num(self.interpolador_cm(alpha_seguro, flap_deg), nan=0.0)
        return cl, cd, cm


# ==============================================================================
# CLASE 2: ENTORNO Y FLUIDO
# ==============================================================================
class EntornoFluido:
    def __init__(self, rho=1.225, V=5.14444):
        self.rho = rho
        self.V = V
        self.presion_dinamica = 0.5 * rho * (V**2)


# ==============================================================================
# CLASE 3: GEOMETRIA DE LA VELA
# ==============================================================================
class GeometriaVela:
    def __init__(self, span, cuerda_base, cuerda_punta, tipo_vela, opcion_bordes, 
                 semi_eje_menor, semi_eje_mayor, n_estaciones=21):
        
        self.span = span
        self.cuerda_base = cuerda_base
        self.cuerda_punta = cuerda_punta
        self.tipo_vela = tipo_vela
        self.opcion_bordes = opcion_bordes
        self.semi_eje_menor = semi_eje_menor
        self.semi_eje_mayor = semi_eje_mayor
        self.n = n_estaciones

        # filtro de seguridad
        if self.tipo_vela == 1:
            # comprobacion de limites fisicos
            if self.span > self.semi_eje_mayor:
                print(f"ALERTA GEOMETRICA: el span ({self.span}m) supera la elipse teorica ({self.semi_eje_mayor}m).")
                print(f"ampliando automaticamente el semi_eje_mayor a {self.span}m para evitar colapsos matematicos.")
                self.semi_eje_mayor = self.span  # auto-correccion 
            
            cuerda_real_punta = self.semi_eje_menor * np.sqrt(1 - (self.span / self.semi_eje_mayor)**2)
            
            print(f"AVISO: Vela eliptica. 'cuerda_en_punta' recalculada a: {cuerda_real_punta:.3f}m")
            self.cuerda_punta = cuerda_real_punta

        # inicia los vectores
        self.angle = np.zeros(self.n)
        self.y = np.zeros(self.n)
        
        # llama a los cslculos al nacer el objeto
        self._calcular_superficie_y_ar()
        self._generar_mallado()
    
    def _generar_mallado(self):
        """genera los vectores de estaciones (y) y distribuciones"""
        
        # genera estaciones (el bucle FOR se queda solo para esto)
        for i in range(self.n):
            self.angle[i] = (i+1) * np.pi / (self.n+1)
            self.y[i] = self.span * 0.5 * np.cos(self.angle[i])

        # distribucion de cuerda (mismo estilo que el x_LE)
        formulas_cuerda = {
            # opción 1: eliptica
            1: lambda y_vec: self.semi_eje_menor * np.sqrt(np.maximum(0, 1 - ((y_vec + (self.span/2)) / self.semi_eje_mayor)**2)),
            # opción 2: trapezoidal
            2: lambda y_vec: ((self.cuerda_punta - self.cuerda_base) / self.span) * (y_vec + self.span/2) + self.cuerda_base
        }
        # ejecuta la formula elegida pasandole el vector self.y
        self.distribucion_cuerda = formulas_cuerda.get(self.tipo_vela, formulas_cuerda[2])(self.y)

        # el borde de ataque (x_LE)
        geometrias_xle = {
            1: lambda: np.zeros_like(self.distribucion_cuerda),
            2: lambda: self.cuerda_base - self.distribucion_cuerda
        }
        self.x_LE = geometrias_xle.get(self.opcion_bordes, geometrias_xle[1])()
        
        # centro aerodinamico global
        self.centro_aero_global = self.x_LE + (0.25 * self.distribucion_cuerda)

    # CALCULO DE SUPERFICIE Y ASPECT RATIO
    def _calcular_superficie_y_ar(self):
        """integra la funcion matematica pura de la cuerda para obtener la superficie y el AR"""
        
        # diccionario  
        # la variable x es la altura desde la base (va de 0 hasta el span).
        funciones_forma = {
            # opcion 1: eliptica (x = 0 es la base, x = span es la punta)
            1: lambda x: self.semi_eje_menor * np.sqrt(np.maximum(0, 1 - (x / self.semi_eje_mayor)**2)),
            # opcion 2: trapezoidal (ecuacion de la recta: y = mx + n)
            2: lambda x: np.maximum(0, ((self.cuerda_punta - self.cuerda_base) / self.span) * x + self.cuerda_base)
            # añadir una rectangular 
            # 3: lambda x: self.cuerda_base
        }

        # extraemos la funcion elegida (por defecto la 2 si hay error)
        funcion_cuerda = funciones_forma.get(self.tipo_vela, funciones_forma[2])
        
        # integra usando quad desde la base (0) hasta la punta (self.span)
        self.S, self.error_S = quad(funcion_cuerda, 0, self.span)
        
        # guarda los resultados
        self.AR = (self.span**2) / self.S 
        # print(f"geometria calculada: superficie = {self.S:.4f} m2 | AR = {self.AR:.4f}") #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ==============================================================================
# CLASE 4: SIMULADOR (motor mates LLT)
# ==============================================================================
class SimuladorLLT:
    def __init__(self, vela, fluido, base_datos, termino=1):
        self.vela = vela
        self.fluido = fluido
        self.bd = base_datos
        self.termino = termino
        
        # matriz de fourier (se calcula una sola vez)
        self.Matriz_Seno = np.zeros((vela.n, vela.n))
        for i in range(vela.n):
            for j in range(vela.n):
                l = (self.termino * j) + 1
                self.Matriz_Seno[i, j] = np.sin(l * vela.angle[i])

    def resolver(self, aoa_global_deg, flap_deg=0):
        self.aoa_global_deg = aoa_global_deg
        self.flap_deg = flap_deg
        alpha_geom_rad = np.full(self.vela.n, np.deg2rad(aoa_global_deg))

        # funcion objetivo para fsolve
        def error_aerodinamico(alpha_eff_guess):
            alpha_eff_deg = np.rad2deg(alpha_eff_guess)
            flaps_deg = np.full(self.vela.n, flap_deg)
            
            # se usa el metodo de la Clase 1
            cl_2d, _, _ = self.bd.consultar_coeficientes(alpha_eff_deg, flaps_deg)
            
            RHS = (self.vela.distribucion_cuerda / (4 * self.vela.span)) * cl_2d
            A_1D_temp = np.linalg.solve(self.Matriz_Seno, RHS)
            
            alpha_w_nuevo = np.zeros(self.vela.n)
            for i in range(self.vela.n):
                suma_downwash = sum([((self.termino * j) + 1) * A_1D_temp[j] * (np.sin(((self.termino * j) + 1) * self.vela.angle[i]) / np.sin(self.vela.angle[i])) for j in range(self.vela.n)])
                alpha_w_nuevo[i] = suma_downwash
            return (alpha_geom_rad - alpha_w_nuevo) - alpha_eff_guess

        # resuelve el LLT
        # print(f"resolviendo LLT para AOA={aoa_global_deg}º...") #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        alpha_inicial_rad = np.copy(alpha_geom_rad)
        self.alpha_eff_rad_final, info_dict, ier, mesg = fsolve(error_aerodinamico, alpha_inicial_rad, full_output=True)
        
        # if ier == 1: print("convergencia alcanzada con exito") #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # else: print("aviso del solver:", mesg) #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        vector_errores = info_dict["fvec"]
        error_maximo = np.max(np.abs(vector_errores))
        # print(f"error maximo de fsolve: {error_maximo:.8e} radianes") #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        # variables finales guardadas en "self" para usarlas en otros metodos deespues del NL-LLT
        self.alpha_eff_deg = np.rad2deg(self.alpha_eff_rad_final)
        self.cl_2d, self.cd_2d, self.cm_2d = self.bd.consultar_coeficientes(self.alpha_eff_deg, np.full(self.vela.n, flap_deg))
        
        self.alpha_eff_deg = np.rad2deg(self.alpha_eff_rad_final)
        self.cl_2d, self.cd_2d, self.cm_2d = self.bd.consultar_coeficientes(self.alpha_eff_deg, np.full(self.vela.n, flap_deg))        
        RHS_final = (self.vela.distribucion_cuerda / (4 * self.vela.span)) * self.cl_2d
        
        self.A_1D = np.linalg.solve(self.Matriz_Seno, RHS_final)
        self.A = self.A_1D.reshape((self.vela.n, 1))

        alpha_geom_rad = np.full(self.vela.n, np.deg2rad(self.aoa_global_deg))
        self.alpha_w = alpha_geom_rad - self.alpha_eff_rad_final
        
        # imprimimos el chequeo de seguridad
        print(f"rango de angulo efectivo final: min {np.min(self.alpha_eff_deg):.2f}º, max {np.max(self.alpha_eff_deg):.2f}º") #+++++++++++++++++++++++++++++++++++
    
    # METODO DE CALCULO AERODINAMICO 
    def calcular_fuerzas_aerodinamicas(self):
        """calcula distribuciones locales (Gamma), Lift, Drag Inducido y Viscoso"""
        
        # calculo de CL y CD totales analiticos
        self.CL_global = self.A[0, 0] * np.pi * self.vela.AR

        self.CD_inducido_global = 0
        for i in range(self.vela.n):
            self.CD_inducido_global += np.pi * self.vela.AR * (self.termino * i + 1) * (self.A[i, 0] ** 2)

        # distribución de carga local (Gamma) y Cl local de fourier
        self.gamma = np.zeros(self.vela.n)
        self.cl_local_fourier = np.zeros(self.vela.n)
        
        for i in range(self.vela.n):
            suma_gamma = 0.0
            for j in range(self.vela.n):
                suma_gamma += 2 * self.A[j, 0] * np.sin((self.termino * j + 1) * self.vela.angle[i])
            self.gamma[i] = suma_gamma * self.vela.span
            
            # coef  de sustentacion local de fourier
            self.cl_local_fourier[i] = 2 * self.gamma[i] / self.vela.distribucion_cuerda[i]

        # coeficiente de resistencia inducida local (Cdi = Cl * downwash)
        self.cdi_local = self.cl_local_fourier * self.alpha_w

        # INTEGRA EL LIFT Y DRAG 
        # variables cortas para simplificar la lectura de las formulas
        theta = self.vela.angle
        cuerda = self.vela.distribucion_cuerda
        q = self.fluido.presion_dinamica # 0.5 * rho * V^2

        # funciones integrando
        self.integrando_cl = cuerda * self.cl_2d * q * (self.vela.span/2) * np.sin(theta)
        self.integrando_cd = cuerda * self.cd_2d * q * (self.vela.span/2) * np.sin(theta)

        # integración numerica 
        self.L_trapz = np.trapezoid(self.integrando_cl, theta)
        self.L_simp = simpson(y=self.integrando_cl, x=theta)
        spline_L = CubicSpline(theta, self.integrando_cl)
        self.L_spline = spline_L.integrate(theta[0], theta[-1])
        
        # lift analitico LLT
        self.L_LLT = q * self.vela.S * self.CL_global

        # calculo de los Drags 
        self.D_inducido = q * self.vela.S * self.CD_inducido_global
        self.D_viscoso = np.trapezoid(self.integrando_cd, theta)
        self.Drag_Total = self.D_inducido + self.D_viscoso

        # error local de sustentación entre iterativo y XFOIL
        # Nota: uso np.divide para evitar avisos por division entre cero
        self.error_cl = np.divide((self.cl_local_fourier - self.cl_2d), self.cl_2d, 
                                  out=np.zeros_like(self.cl_2d), where=self.cl_2d!=0)


        # print(f"resistencia inducida (LLT):   {self.D_inducido:.3f} N")
        # print(f"resistencia viscosa (XFOIL):  {self.D_viscoso:.3f} N")
        print(f"RESISTENCIA TOTAL DE VELA:    {self.Drag_Total:.3f} N")
        # print("COMPARATIVA DE MÉTODOS DE INTEGRACIÓN LIFT:")
        # print(f"LIFT Trapecios:  {self.L_trapz:.3f} N")
        # print(f"LIFT Simpson:    {self.L_simp:.3f} N")
        # print(f"LIFT Spline:     {self.L_spline:.3f} N")
        print(f"LIFT Analitico:  {self.L_LLT:.3f} N")

    def calcular_esfuerzos_3d(self, posicion_mastil_base):
        # Brazos de palanca
        posicion_mastil_global = np.full_like(self.vela.y, posicion_mastil_base)
        brazo_horizontal = posicion_mastil_global - self.vela.centro_aero_global
        brazo_vertical = self.vela.y + (self.vela.span/2)
        
        # Proyeccion de Fuerzas
        integrando_cl = self.vela.distribucion_cuerda * self.cl_2d * self.fluido.presion_dinamica * (self.vela.span/2) * np.sin(self.vela.angle)
        integrando_cd = self.vela.distribucion_cuerda * self.cd_2d * self.fluido.presion_dinamica * (self.vela.span/2) * np.sin(self.vela.angle)
        
        integrando_Normal = integrando_cl * np.cos(self.alpha_eff_rad_final) + integrando_cd * np.sin(self.alpha_eff_rad_final)
        integrando_Axial = integrando_cd * np.cos(self.alpha_eff_rad_final) - integrando_cl * np.sin(self.alpha_eff_rad_final)

        # Momentos
        integrando_Cm_puro = self.fluido.presion_dinamica * (self.vela.distribucion_cuerda**2) * self.cm_2d * (self.vela.span/2) * np.sin(self.vela.angle)
        
        self.integrando_Torsion = integrando_Cm_puro + (integrando_Normal * brazo_horizontal)
        self.integrando_Escora = integrando_Normal * brazo_vertical
        self.integrando_Cabeceo = integrando_Axial * brazo_vertical

        # Integracion total
        momento_Torsion = simpson(self.integrando_Torsion, x=self.vela.angle)
        momento_Escora = simpson(self.integrando_Escora, x=self.vela.angle)
        momento_Cabeceo = simpson(self.integrando_Cabeceo, x=self.vela.angle)

        # print("\n=== ANÁLISIS ESTRUCTURAL 3D ===")
        print(f"Torsión (Mastil): {momento_Torsion:.2f} Nm")
        print(f"Escora (Base):    {momento_Escora:.2f} Nm")
        print(f"Cabeceo (Proa):   {momento_Cabeceo:.2f} Nm")
        
        return momento_Torsion, momento_Escora, momento_Cabeceo

# ==============================================================================
# CLASE 5:  GRAFICAS (generacion de graficas y resultados)
# ==============================================================================
class VisualizadorResultados:
    def __init__(self, simulador):
        # esta clase solo necesita saber quién es el simulador para extraerle los datos
        self.sim = simulador
        self.vela = simulador.vela

    def graficar_forma_planta(self, posicion_mastil_base=0.0):
        """dibuja la geometria fisica de la vela (plano de diseño 2D X-Y)."""
                
        # calcula la linea del borde de Salida (TE = Nariz + Cuerda)
        x_TE = self.vela.x_LE + self.vela.distribucion_cuerda
        
        plt.figure(figsize=(8, 10)) # formato vertical
        
        # dibuja las lineas de los bordes
        plt.plot(self.vela.x_LE, self.vela.y, 'k-', linewidth=2, label='Borde de Ataque (Nariz)')
        plt.plot(x_TE, self.vela.y, 'r--', linewidth=2, label='Borde de Salida (Cola)')
        
        # colorea el area de la vela
        plt.fill_betweenx(self.vela.y, self.vela.x_LE, x_TE, color='skyblue', alpha=0.4, label='Área Vela')
        
        # eje del mástil físico en su posición real
        plt.axvline(posicion_mastil_base, color='gray', linestyle=':', linewidth=2, label=f'Eje Mástil (X={posicion_mastil_base}m)')

        # detalles esteticos y escalas
        plt.title(f'Geometría de Diseño (Forma en Planta)\nSuperficie: {self.vela.S:.2f} m2', fontsize=14, fontweight='bold')
        plt.xlabel('Eje Longitudinal (X) [m] (Proa -> Popa)', fontsize=12)
        plt.ylabel('Posición en la vela (y) [m] (Base -> Punta)', fontsize=12)
        
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        # saca fuera del area de dibujo la leyenda
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, shadow=True)
        
        # forzamos escala 1:1 para que no se deforme la geometria
        plt.axis('equal') 
        plt.tight_layout()
        plt.show()

    def graficar_distribucion_fuerzas_seccionales(self):
        """Muestra el Lift y Drag local rodaja a rodaja (N/m) contra el Span."""
        
        print("-> Generando gráfica de distribución de fuerzas locales...")
        
        # reconstruimos las densidades fisicas de fuerza (N/m)
        q = self.sim.fluido.presion_dinamica
        c = self.vela.distribucion_cuerda
        
        dl_dy = q * c * self.sim.cl_2d
        dd_dy_viscoso = q * c * self.sim.cd_2d
        dd_dy_inducido = q * c * self.sim.cdi_local 
        dd_dy_total = dd_dy_viscoso + dd_dy_inducido

        # configuracion de subplots
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # eje 1: LIFT (dL/dy) ---
        color_lift = 'tab:blue'
        ax1.set_xlabel('Posición de envergadura (y) [m] (Base -> Punta)', fontsize=12)
        ax1.set_ylabel('Lift local (dL/dy) [N/m]', color=color_lift, fontsize=12)
        ax1.plot(self.vela.y, dl_dy, color=color_lift, linewidth=2.5, label='Lift (Sustentación)')
        ax1.tick_params(axis='y', labelcolor=color_lift)
        ax1.fill_between(self.vela.y, dl_dy, color=color_lift, alpha=0.2)

        # eje 2: DRAG (dD/dy) ---
        ax2 = ax1.twinx()  
        color_drag = 'tab:red'
        ax2.set_ylabel('Drag local (dD/dy) [N/m]', color=color_drag, fontsize=12)
        
        ax2.plot(self.vela.y, dd_dy_total, color=color_drag, linewidth=2.5, label='Drag Total')
        ax2.plot(self.vela.y, dd_dy_viscoso, color='gray', linestyle=':', linewidth=1.5, label='Drag Viscoso')
        ax2.plot(self.vela.y, dd_dy_inducido, color='gray', linestyle='--', linewidth=1.5, label='Drag Inducido')
        
        ax2.tick_params(axis='y', labelcolor=color_drag)

        # titulos y Leyendas
        plt.title('Distribución de Cargas Aerodinámicas a lo largo del Span', fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.7)
        
        # sacamos la leyenda conjunta debajo de la grafica
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=2)
        
        fig.tight_layout()
        plt.show()

    def graficar_cargas_estructurales(self):
        """dibuja la distribucion local de esfuerzos a lo largo del mastil"""
        
        # quito el Jacobiano para obtener la distribucion real (Nm/m) en el eje fisico (y)
        jacobiano = (self.vela.span/2) * np.sin(self.vela.angle)

        # uso np.divide para evitar dividir por cero en la base y la punta
        dM_dy_Torsion = np.divide(self.sim.integrando_Torsion, jacobiano, out=np.zeros_like(self.sim.integrando_Torsion), where=jacobiano!=0)
        dM_dy_Escora  = np.divide(self.sim.integrando_Escora, jacobiano, out=np.zeros_like(self.sim.integrando_Escora), where=jacobiano!=0)
        dM_dy_Cabeceo = np.divide(self.sim.integrando_Cabeceo, jacobiano, out=np.zeros_like(self.sim.integrando_Cabeceo), where=jacobiano!=0)

        # configuracion de la figura
        plt.figure(figsize=(10, 6))

        # dibuja las tres curvas (y va desde -span/2 en la base hasta +span/2 en la punta)
        plt.plot(self.vela.y, dM_dy_Escora, color='blue', linestyle='-', linewidth=2.5, 
                 label='Escora (Contribución al vuelco lateral)')
        plt.plot(self.vela.y, dM_dy_Cabeceo, color='green', linestyle='--', linewidth=2.5, 
                 label='Cabeceo (Contribución a la flexión trasera)')
        plt.plot(self.vela.y, dM_dy_Torsion, color='red', linestyle='-.', linewidth=2.5, 
                 label='Torsión (Contribución al giro en su eje)')

        # detalles esteticos
        plt.title('Distribución de Cargas Aerodinámicas en el Mástil', fontsize=15, fontweight='bold')
        plt.xlabel(f'Posición en la vela (y) [m]   ({-self.vela.span/2:.1f} = Base, {self.vela.span/2:.1f} = Punta)', fontsize=12)
        plt.ylabel('Carga Local aportada a la base (dM/dy) [Nm/m]', fontsize=12)

        # lineas de referencia (ejes neutros)
        plt.axhline(0, color='black', linewidth=1.5) 
        plt.axvline(0, color='gray', linestyle=':', alpha=0.7) 

        # mostrar y ajustar
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend(loc='upper right', fontsize=10, shadow=True)
        plt.tight_layout()
        plt.show()

    def encontrar_mastil_optimo(self):
            """busca matematicamente el punto exacto y dibuja la grafica de torque vs posicion"""
            
            print("\nbuscando posicion optima del mastil (torque cero)")
            
            # reconstruimos la fisica necesaria
            integrando_Normal = self.sim.integrando_cl * np.cos(self.sim.alpha_eff_rad_final) + \
                                self.sim.integrando_cd * np.sin(self.sim.alpha_eff_rad_final)
                                
            integrando_Cm_puro = self.sim.fluido.presion_dinamica * (self.vela.distribucion_cuerda**2) * \
                                self.sim.cm_2d * (self.vela.span/2) * np.sin(self.vela.angle)

            # función objetivo: calcula la torsion para una posicion 'pos' cualquiera
            def calcular_torque_residual(pos):
                posicion_prueba = np.full_like(self.vela.y, pos)
                brazo_prueba = posicion_prueba - self.vela.centro_aero_global
                integrando_prueba = integrando_Cm_puro + (integrando_Normal * brazo_prueba)
                return simpson(integrando_prueba, x=self.vela.angle)

            # fsolve encuentra la raíz (el cero)
            posicion_optima = fsolve(calcular_torque_residual, x0=0.5)[0]
            print(f"encontrado posicion optima del mastil: {posicion_optima:.3f} metros desde el borde de ataque")

            # ====================================================================
            # GENERACION DE DATOS Y GRAFICA 
            # ====================================================================
            
            # genera los datos evaluando la funcion que esta creada arriba
            max_cuerda = max(self.vela.cuerda_base, self.vela.cuerda_punta)
            posiciones_grafica = np.linspace(0.0, max_cuerda, 30)
            torques_grafica = [calcular_torque_residual(p) for p in posiciones_grafica]

            # dibujamos la grafica
            plt.figure(figsize=(7, 4))
            plt.plot(posiciones_grafica, torques_grafica, 'b-', linewidth=2, label='esfuerzo del mastil')

            # marca la linea del cero y el punto optimo
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5) # eje neutro (0 Nm)
            plt.plot(posicion_optima, 0, 'ro', markersize=8, label=f'optimo: {posicion_optima:.3f} m')

            # estetica de la grafica ejes y tal
            plt.title('torque vs posicion del mastil', fontsize=12, fontweight='bold')
            plt.xlabel('distancia del mastil desde el borde de ataque[m]')
            plt.ylabel('torque de torsion [Nm]')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.show()

# ==============================================================================
# MAIN: INTERFAZ DEL USUARIO 
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Base de datos
    CARPETA = r"C:\Users\aamal\OneDrive - Universidad Politécnica de Madrid\Escritorio\pruebas_tfg\xflr5_2D_prubas"
    ARCHIVOS = {
        "NACA 0020_Re0.462_flap_0.txt": [404000, 0],
        "NACA 0020_Re0.520_flap_5.txt": [520000, 5],    
        "NACA 0020_Re0.520_flap_10.txt": [520000, 10],
        "NACA 0020_Re0.520_flap_15.txt": [520000, 15],
        "NACA 0020_Re0.520_flap_20.txt": [520000, 20],
        "NACA 0020_Re0.520_flap_25.txt": [520000, 25],
        "NACA 0020_Re0.520_flap_30.txt": [520000, 30],
    }
    mi_bd = BaseDatosAerodinamica(CARPETA, ARCHIVOS)
    
    # 2. Entorno
    mi_fluido = EntornoFluido(rho=1.225, V=5.14444)
    
    # 3. Diseño de la Vela
    mi_vela = GeometriaVela(
        span=5, 
        cuerda_base=2, 
        cuerda_punta=1.4, 
        tipo_vela=1,           # 1=Eliptica, 2=Trapezoidal
        opcion_bordes=1,       # 1=Ataque Recto, 2=Salida Recta
        semi_eje_menor=2, 
        semi_eje_mayor=7,
        n_estaciones=21        # mejor dejar en numero IMPAR, no poner menos de 5 ni mas de 40 y en caso de fallo bajar el numero 
    )
    
    # 4. Inicializar motor de simulacion
    mi_simulador = SimuladorLLT(vela=mi_vela, fluido=mi_fluido, base_datos=mi_bd, termino=1)

    angulos_a_probar = [3, 5, 8, 10]
    angulos_aprobar_flap= [0, 5, 10, 15, 20, 25, 30]
    print("\nINICIANDO BARRIDO DE ÁNGULOS")
    for aoa in angulos_a_probar:
        for aof in angulos_aprobar_flap:
            print(f"\nProbando AOA = {aoa}º")
            print(f"\nProbando AOF = {aof}º")
            mi_simulador.resolver(aoa_global_deg=aoa, flap_deg=aof)
            mi_simulador.calcular_fuerzas_aerodinamicas()
            mi_simulador.calcular_esfuerzos_3d(posicion_mastil_base=0.5)
                    
            # lift = mi_simulador.L_LLT
            # drag = mi_simulador.Drag_Total
            # print(f"Resultado rápido -> Lift: {lift:.1f} N | Drag: {drag:.1f} N")
    
    # --- EJECUCION DE CALCULOS ---
    # mi_simulador.resolver(aoa_global_deg=5, flap_deg=0)
    # mi_simulador.calcular_fuerzas_aerodinamicas()
    # mi_simulador.calcular_esfuerzos_3d(posicion_mastil_base=0.5)
    
    # 5. generacion de graficas y optimizacion
    # mi_visualizador = VisualizadorResultados(mi_simulador)
    # mi_visualizador.graficar_forma_planta(posicion_mastil_base=0.5)
    # mi_visualizador.graficar_distribucion_fuerzas_seccionales()
    # mi_visualizador.graficar_cargas_estructurales()
    # mi_visualizador.encontrar_mastil_optimo()

