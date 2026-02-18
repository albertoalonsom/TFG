import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.integrate import quad
from functools import partial
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

# ==============================================================================
# 1. CONFIGURACIÓN DE USUARIO (Modifica esto primero SIGUIENDO INDICACIONES)
# ==============================================================================

# Escribe aquí la ruta de la carpeta donde están tus archivos .txt
CARPETA_DATOS = r"C:\Users\aamal\OneDrive - Universidad Politécnica de Madrid\Escritorio\pruebas_tfg\xflr5_2D_prubas"

# Define aquí los archivos que quieres procesar y sus valores asociados:
# Formato: "Nombre_del_Archivo.txt": [Valor_Reynolds, Valor_Angulo]
CONFIGURACION_ARCHIVOS = {
    "NACA 0020_Re0.462_flap_0.txt": [404000, 0],
    "NACA 0020_Re0.520_flap_5.txt": [520000, 5],

    # Añade más archivos aquí siguiendo el mismo formato
}
#============================================================================================
# ENTRADAS DEL USUARIO A MODIFICAR PARA CODIGO LIFTING LINE
#============================================================================================
cuerda_en_base = 2 # cuerda en la base de la vela [m]
cuerda_en_punta = 1.4 # cuerda en la punta de la vela [m]
span = 5 # span total de la vela [m]
rho = 1.225  # densidad del aire [kg/m^3]
V = 5.14444      # velocidad [m/s]

valor_angulo_ataque = 5 # VALORES EN GRADOS [º]
valor_angulo_flap = 0   # VALRES EN GRADOS [º]


#valores para la distribucion de cuerda de una vela semieliptica, es decir un cuarto de elipse siendo sus valores de los semi ejes 
semiancho = 2
semilargo= 7

# de momento solo funciona con vela eliptica poner 1
modelo_seleccionado = 1 #poner 1 para vela con distribucion de un cuartode elipse y poner 2 para trapezoidal
opcion_usuario = 1

# twistroot = 0 # valor Root twist angle in degrees    en nuestros casos simpre 0
# twisttip = 0 # valor tip twist angle in degrees      en nuestros casos simpre 0


# ==============================================================================
# PROCESADO DE DATOS TXT  (No tocar a menos que quieras cambiar la estructura para incluir mas datos)
# ==============================================================================

def procesar_polar(ruta_archivo, re_val, ang_val):
    """Lee el TXT y devuelve una matriz con el formato [Re, Alpha, Angulo, CL, CD]"""
    try:
        # Lectura y limpieza automática de datos numéricos
        df = pd.read_csv(
            ruta_archivo, sep='\s+', header=None, skiprows=11, 
            names=["alpha", "CL", "CD", "Cm", "Xcp", "CDp"], 
            usecols=range(6), engine='python', on_bad_lines='skip'
        ).apply(pd.to_numeric, errors='coerce').dropna()

        # Creación de la matriz por columnas y conversión a lista de filas
        matriz_np = np.column_stack((
            np.full(len(df), re_val),   # Columna Reynolds
            df["alpha"].to_numpy(),     # Columna Angulo de ataque 
            np.full(len(df), ang_val),  # Columna Angulo de flap
            df["CL"].to_numpy(),        # Columna CL
            df["CD"].to_numpy()         # Columna CD
        ))
        
        return matriz_np.tolist()
    
    except Exception as e:
        print(f"--- Error al procesar: {os.path.basename(ruta_archivo)} ---")
        print(f"Causa: {e}\n")
        return []

# --- EJECUCIÓN PRINCIPAL ---

matriz_DATOS2D = []

for nombre, valores in CONFIGURACION_ARCHIVOS.items():
    ruta_full = os.path.join(CARPETA_DATOS, nombre)
    
    if os.path.exists(ruta_full):
        filas = procesar_polar(ruta_full, valores[0], valores[1])
        matriz_DATOS2D.extend(filas)
    else:
        print(f"Archivo no encontrado: {nombre}")

# --- RESULTADO FINAL DE LA MATRIZ CON LOS DATOS DESEADOS---

# Imprime por pantalla la matriz resultante por filas
# for fila in matriz_DATOS2D:
#     print(fila)



#=====================================================================================
#para el interpolador 
#=====================================================================================

if len(matriz_DATOS2D) > 0:
    #para poder manipular columnas
    datos_array = np.array(matriz_DATOS2D)

    # para extraemos los "Puntos" (Inputs): agulo ataque y angulo de flap
    puntos_entrada = datos_array[:, [1, 2]]  
    #para extraer los "Valores" (Output): AOA
    valores_alpha = datos_array[:, 1]
    #para extraer los "Valores" (Output): CL
    valores_cl = datos_array[:, 3]
    #para extraer los "Valores" (Output): CD
    valores_cd = datos_array[:, 4]

    # creacion el interpolador
    # rescale=True creo que es necesario aquí porque Alpha (ej. -5 a 15) y Flap (ej. 0 a 40) tienen escalas parecidas,
    # pero si usaro el Reynolds, las escalas serían muy distintas
    interpolador_cl = LinearNDInterpolator(puntos_entrada, valores_cl, rescale=True)
    interpolador_cd = LinearNDInterpolator(puntos_entrada, valores_cd, rescale=True)
   
    # ejemplo para verificar si esta correcto
    # si quiero predecir CL para: Alpha = 5.9 grados, Flap = 5.6 grados
    
    cl_interpolado1 = interpolador_cl(valor_angulo_ataque, valor_angulo_flap)
    
else:
    print("No se cargaron datos, revisa la configuración de archivos.")


#======================================================
# CODIGO DEL LIFTING LINE 
#=========================================================
# Entradas del usuario
#a0tip = np.pi*2 #float(input("valor lift curve slope at the tip in units/radian: "))  #corregir con el 2D 
#alpha0tip = 0 # float(input("valor zero lift angle at the tip: "))     #mirar en 2D 

#terminos pares serie 
#termino = 2 # ala simetrica elimina terminos pares de la serie q son nulos 
termino = 1 # vela no es simetrica y hay q tener en cuenta los terminos pares


# Conversión de grados a radianes
deg2rad = np.pi / 180
valor_angulo_ataque *= deg2rad
# twistroot *= deg2rad
# twisttip *= deg2rad
# alpha0tip *= deg2rad


n = 20 # número de estaciones dejar mejor un un numero impar 
# Inicialización de vectores para resolver
angle = np.zeros(n)
y = np.zeros(n)
distribucion_cuerda = np.zeros(n)

# Definición de propiedades en estaciones (Geometría)
for i in range(n):
    angle[i] = (i+1) * np.pi / (n+1) # Evita 0 y pi
    y[i] = span * 0.5 * np.cos(angle[i])
    
    match opcion_usuario:
        case 0:
            distribucion_cuerda[i] = semiancho * (np.sqrt(1 - ((y[i]) / semilargo)**2))
        case 1:
            distribucion_cuerda[i] = semiancho * (np.sqrt(1 - ((y[i] + 2.5) / semilargo)**2))
        case 2:
            distribucion_cuerda[i] = ((cuerda_en_punta - cuerda_en_base) / span) * (y[i] + span/2) + cuerda_en_base


# ==============================================================================
# CREACIÓN DE LA MATRIZ (Fuera del bucle, solo se calcula una vez)
# ==============================================================================
Matriz_Seno = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        l = (termino * j) + 1
        Matriz_Seno[i, j] = np.sin(l * angle[i])

# ==============================================================================
# BUCLE NLLT  (Usando scipy.optimize.fsolve)
# ==============================================================================

LIMITE_MIN_DEG = -15.0 
LIMITE_MAX_DEG = 15.0

alpha_geom_rad = np.full(n, valor_angulo_ataque)

def calcular_error_aerodinamico(alpha_eff_guess):
    """
    Esta función: Toma un ángulo propuesto, calcula el downwash
    y devuelve la diferencia entre lo que debería ser y lo que propuso.
    ¡Fsolve buscará que esta diferencia sea CERO!
    """
    # 1. Escudo de grados
    alpha_eff_deg = np.rad2deg(alpha_eff_guess)
    alpha_eff_deg_seguro = np.clip(alpha_eff_deg, LIMITE_MIN_DEG, LIMITE_MAX_DEG)
    flaps_deg = np.full(n, valor_angulo_flap)
    
    # 2. Leemos la polar
    cl_2d = interpolador_cl(alpha_eff_deg_seguro, flaps_deg)
    cl_2d = np.nan_to_num(cl_2d, nan=0.0)
    
    # 3. Resolvemos la matriz temporalmente
    RHS = (distribucion_cuerda / (4 * span)) * cl_2d
    A_1D_temp = np.linalg.solve(Matriz_Seno, RHS)
    
    # 4. Calculamos el downwash
    alpha_w_nuevo = np.zeros(n)
    for i in range(n):
        suma_downwash = 0
        for j in range(n):
            l = (termino * j) + 1
            suma_downwash += l * A_1D_temp[j] * (np.sin(l * angle[i]) / np.sin(angle[i]))
        alpha_w_nuevo[i] = suma_downwash
        
    # 5. Ángulo que dicta la física vs Ángulo que propusimos
    alpha_eff_calc = alpha_geom_rad - alpha_w_nuevo
    return alpha_eff_calc - alpha_eff_guess

print("\nIniciando convergencia NLLT con fsolve (Autopiloto)...")

# Le damos el punto de partida (El ángulo geométrico inicial)
alpha_inicial_rad = np.copy(alpha_geom_rad)

# 
alpha_eff_rad_final, info_dict, ier, mesg = fsolve(calcular_error_aerodinamico, alpha_inicial_rad, full_output=True)

if ier == 1:
    print("¡Convergencia NLLT alcanzada con éxito por SciPy!")
else:
    print("Aviso del solver:", mesg)

# Extraemos el vector de errores (residuos) y buscamos el más grande
vector_errores = info_dict["fvec"]
error_maximo = np.max(np.abs(vector_errores))

print(f"Error máximo de fsolve: {error_maximo:.8e} radianes")

# ==============================================================================
# RECONSTRUCCIÓN FINAL DE VARIABLES PARA EL RESTO DEL CÓDIGO
# ==============================================================================
# Ahora que ya sabemos el ángulo efectivo perfecto, hacemos el cálculo una última vez
# para guardar los coeficientes A_1D y el cl_2d y pasárselos a tus integrales
alpha_eff_deg = np.rad2deg(alpha_eff_rad_final)
cl_2d = np.nan_to_num(interpolador_cl(np.clip(alpha_eff_deg, LIMITE_MIN_DEG, LIMITE_MAX_DEG), np.full(n, valor_angulo_flap)), nan=0.0)

RHS_final = (distribucion_cuerda / (4 * span)) * cl_2d
A_1D = np.linalg.solve(Matriz_Seno, RHS_final)

# Creamos las variables exactas que usas en el resto de tu código
alpha_w_nuevo = alpha_geom_rad - alpha_eff_rad_final
A = A_1D.reshape((n, 1))

print(f"Rango de ángulo efectivo final: min {np.min(alpha_eff_deg):.2f}º, max {np.max(alpha_eff_deg):.2f}º")

# ==========================================
# INTEGRACIÓN para el calculo de la superficie y AR
# ==========================================
def calcular_aerodinamica(span_real, funcion_cuerda):
    """
    Integra la función hasta el span_real físico del ala en funcion de la distrubcion de cuerda elegida a traves de los perfiles_truncados
    """
    area_media, error = quad(funcion_cuerda, 0, span) # Integramos desde 0 hasta el span del ala
    S = area_media # superficie del ala 
    AR = span**2 / S # AR basado en la envergadura física
    return S, AR, error

# ==========================================
# DEFINICIÓN DE FORMAS DEL ALA OJO NO MODIFICAR NUNCA, MODIFICARLO EN PERFIL TRUNCADO
# ==========================================

def forma_eliptica_general(x, root_chord, span_teorico):
    """
    Calcula la cuerda basándose en una elipse de tamaño 'span_teorico'.
    """
    term = 1 - (x / span_teorico)**2 # Ecuación de la elipse
    return root_chord * np.sqrt(np.maximum(0, term)) # Si x supera el semilargo teórico, devolvemos 0 para evitar errores matemáticos

def forma_trapezoidal_general(x, root_chord, tip_chord, span_teorico):
    """
    Calcula la cuerda basándose en un trapecio rectangulo en base a base mayor(cuerda en la base), base menor(cuerda en la punta) y altura(span)
    """
    pendiente = (tip_chord - root_chord) / span_teorico # para los calculos se usa en el eje X el span y en el eje Y la cuerda
    return np.maximum(0, pendiente * x + root_chord) 

# ==========================================
# EJECUCIÓN DEL PROBLEMA Configuramos la curvatura del ala usando los datos teoricos de la ecuacion matematica de la que partimos
# ==========================================

def opciones_selccion_modelo(opcion):
    perfil_truncado_elipse = partial(forma_eliptica_general, 
                          root_chord=cuerda_en_base, 
                          span_teorico=semilargo)

    perfil_truncado_trapezoidal = partial(forma_trapezoidal_general,
                        root_chord=cuerda_en_base,
                        tip_chord=cuerda_en_punta,
                        span_teorico=span)
    match opcion:
        case 1:
            return perfil_truncado_elipse
        case 2:
            return perfil_truncado_trapezoidal

modelo_seleccionado_para_calculo = opciones_selccion_modelo(modelo_seleccionado)

# Calculamos la INTEGRAL usando los datos REALES (los límites de integración 0 a span)
S, AR, error = calcular_aerodinamica(span, modelo_seleccionado_para_calculo)
print(f"Superficie vela: {S:.4f}")
print(f"Aspect Ratio vela: {AR:.4f}")
print(f"error vela: {error:.4f}")

# Cálculo de CL total analítico
CL = A[0, 0] * np.pi * AR

# Cálculo de CD total (Coeficiente de Resistencia Inducida Global)
CD = 0
for i in range(n):
    CD += np.pi * AR * (termino * i + 1) * A[i, 0] ** 2

# Reciclamos el downwash que ya calculó el bucle iterativo NLLT
alpha_w = alpha_w_nuevo 

# Cálculo de distribución de carga (gamma) y cl local
gamma = np.zeros(n)
cl = np.zeros(n)
for i in range(n):
    gamma[i] = 0.0
    for j in range(n):
        gamma[i] += 2 * A[j, 0] * np.sin((termino * j + 1) * angle[i])
    gamma[i] *= span
    cl[i] = 2 * gamma[i] / distribucion_cuerda[i] # coef de sustentacion local de Fourier

# Coeficiente de resistencia inducida local (seccional)
# Es simplemente la sustentación local multiplicada por el downwash (en radianes)
cdi = cl * alpha_w

# Mostrar resultados
# print("\n" + "="*40)
# print("VALORES GLOBALES LLT (Fourier)")
# print("="*40)
# print(f"CL Total = {CL:.5f}")
# print(f"CD Inducido Total = {CD:.5f}")
# print("vector alphadownwash (rad):", alpha_w)
# print("vector cl local:", cl)

# ==============================================================================
# OBTENCIÓN DEL CD VISCOSO FINAL (El Cl ya convergió en el bucle NLLT)
# ==============================================================================

# Usamos directamente el ángulo efectivo en grados que salió del bucle 'while'
flaps_grados = np.full_like(alpha_eff_deg, valor_angulo_flap)

# Interpolar SOLO el Cd, porque el Cl ya obligamos a que fuera igual en el bucle
raw_cd = interpolador_cd(alpha_eff_deg, flaps_grados)
resultados_cdxfoil = np.nan_to_num(raw_cd, nan=0.0)

# Para la integral, el Cl de XFOIL es simplemente el que usamos en la última iteración del bucle
resultados_clxfoil = cl_2d 

# print("\nVector de CD viscoso (XFOIL):")
# print(resultados_cdxfoil)

#========================================================
# INTEGRACION DEL LIFT Y DRAG
#=======================================================

# datos 
theta =np.array(angle)  
C_integra= np.array(distribucion_cuerda)                    
Cl_integra = np.array(resultados_clxfoil) 
Cd_integra = np.array(resultados_cdxfoil)                    

# Función integrando
integrando_cl = C_integra * Cl_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)
integrando_cd = C_integra * Cd_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)

# Integración con distintas reglas para verificar y ver errores
L_trapz = np.trapezoid(integrando_cl, theta)
L_simp = simpson(y=integrando_cl, x=theta)
spline_L = CubicSpline(theta, integrando_cl)
L_spline = spline_L.integrate(theta[0], theta[-1])
L_LLT = 0.5 * rho * V**2 * S * CL

D_inducido = 0.5 * rho * V**2 * S * CD  # Fuerza de Drag Inducido (Newtons)
D_viscoso = np.trapezoid(integrando_cd, theta) # Fuerza de Drag Viscoso (Newtons)
Drag_Total = D_inducido + D_viscoso

print("Resistencia Inducida (LLT) = ", D_inducido)
print("Resistencia Viscosa (XFOIL) = ", D_viscoso)
print("RESISTENCIA TOTAL DE LA VELA = ", Drag_Total)

print("LIFT trapecios= ", L_trapz)
print("LIFT simpson= ", L_simp)
print("LIFT spline= ", L_spline)
print("L_llt = ", L_LLT)

error = (cl - resultados_clxfoil)/resultados_clxfoil
print("error")
print(error)

# #graficas 
plt.plot(y, cl)
plt.xlabel("span")
plt.ylabel("cl")
plt.title("titulo")
plt.grid(True)
plt.show()