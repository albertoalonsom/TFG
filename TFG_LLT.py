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

# ==============================================================================
# 1. CONFIGURACIÓN DE USUARIO (Modifica esto primero SIGUIENDO INDICACIONES)
# ==============================================================================

# Escribe aquí la ruta de la carpeta donde están tus archivos .txt
CARPETA_DATOS = r"C:\Users\aamal\OneDrive - Universidad Politécnica de Madrid\Escritorio\pruebas_tfg\xflr5_2D_prubas"

# Define aquí los archivos que quieres procesar y sus valores asociados:
# Formato: "Nombre_del_Archivo.txt": [Valor_Reynolds, Valor_Angulo]
CONFIGURACION_ARCHIVOS = {
    "NACA 0020_Re0.462.txt": [404000, 5.0],
    "NACA 0020_Re0.578.txt": [600000, 7.5],
    # Añade más archivos aquí siguiendo el mismo formato
}
#============================================================================================
# ENTRADAS DEL USUARIO A MODIFICAR PARA CODIGO LIFTING LINE
#============================================================================================
cuerda_en_base = 2 # cuerda en la base de la vela [m]
cuerda_en_punta = 1.4 # cuerda en la punta de la vela [m]
span = 5 # span total de la vela [m]
rho = 1.225  # densidad del aire [kg/m^3]
V = 7       # velocidad [m/s]

valor_angulo_ataque = 7 # VALORES EN GRADOS [º]
valor_angulo_flap = 5    # VALRES EN GRADOS [º]

valor_angulo_ataque2_calculo_pendiente = valor_angulo_ataque - 0.5 # VALOR DEL ANGULO DE ATAQUE 2 PARA CALCULO DE LA PENDIENTE

cl_cero_2D = 0 # Dejar en cero

#valores para la distribucion de cuerda de una vela semieliptica, es decir un cuarto de elipse siendo sus valores de los semi ejes 
semiancho = 2
semilargo= 7

# de momento solo funciona con vela eliptica poner 1
modelo_seleccionado = 1 #poner 1 para vela con distribucion de un cuartode elipse y poner 2 para trapezoidal


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
for fila in matriz_DATOS2D:
    print(fila)



#=====================================================================================
#para el interpolador 
#=====================================================================================

if len(matriz_DATOS2D) > 0:
    #para poder manipular columnas
    datos_array = np.array(matriz_DATOS2D)

    # para extraemos los "Puntos" (Inputs): agulo ataque y angulo de flap
    puntos_entrada = datos_array[:, [1, 2]]  
    puntos_entrada_inv = datos_array[:, [3, 2]] 
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
    interpolador_alpha_L0 = LinearNDInterpolator(puntos_entrada_inv, valores_alpha, rescale=True)
   
    # ejemplo para verificar si esta correcto
    # si quiero predecir CL para: Alpha = 5.9 grados, Flap = 5.6 grados
    
    cl_interpolado1 = interpolador_cl(valor_angulo_ataque, valor_angulo_flap)
    cl_interpolado2 = interpolador_cl(valor_angulo_ataque2_calculo_pendiente, valor_angulo_flap)
    angulo_para_cl_cero = interpolador_alpha_L0(cl_cero_2D, valor_angulo_flap)
    pendiente = (cl_interpolado2 - cl_interpolado1) / np.deg2rad(valor_angulo_ataque2_calculo_pendiente - valor_angulo_ataque) if valor_angulo_ataque2_calculo_pendiente != valor_angulo_ataque else 0
    
    print(f"Para Alpha={valor_angulo_ataque} y Flap={valor_angulo_flap}, el CL estimado es: {cl_interpolado1}")
    print(f"Para Alpha={valor_angulo_ataque2_calculo_pendiente} y Flap={valor_angulo_flap}, el CL estimado es: {cl_interpolado2}")
    print(pendiente)
    print(angulo_para_cl_cero)
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
angulo_para_cl_cero *= deg2rad
# twistroot *= deg2rad
# twisttip *= deg2rad
# alpha0tip *= deg2rad


n = 20 # número de estaciones dejar mejor un un numero impar 

# Inicialización de vectores para resolver 
angle = np.zeros(n)
y = np.zeros(n)
distribucion_cuerda = np.zeros(n)
pendiente2D = np.zeros(n)
b=np.zeros(n)
ecuacionLLT = np.zeros((n, 1))
ecuacionLLT_2 = np.zeros((n, n))

# Definición de propiedades en estaciones
for i in range(n):
    angle[i] = (i+1) * np.pi / (n+1) #va desde 0 hasta pi sin incluir 0 y pi ya que conduce a una matriz singular q no se puede resolver 
    y[i] = span * 0.5 * np.cos(angle[i]) # posicion de la envergadura de la vela
    # aqui podria meter un if y dependiendo de lo que el usuario elija tener distinta distribucion de cuerda para no siempre ser la eliptica
    #c[i] = semiancho * (np.sqrt(1-((y[i])/semilargo)**2))                 # para hacerla simetrica y validar con xflr5
    distribucion_cuerda[i] = semiancho * (np.sqrt(1-((y[i]+2.5)/semilargo)**2))              # debido al distribucion eliptica del ala y desplazada por situar el eje el en medio de la vela
    pendiente2D[i] = pendiente        # pendiente curva 2D
    b[i] = cl_interpolado1     # este valor no deja de ser la pendiente multiplicada por (angulo de ataque - angulo a cl=0)

# Sistema de ecuaciones lineales
for j in range(n):
    ecuacionLLT[j, 0] = ((b[j] * distribucion_cuerda[j]) / (4 * span)) * np.sin(angle[j]) # En las ecuaciones generales se multiplica por 8  el span porque considera solo medio span 
    for i in range(n):                                                                    # es decir el span que muestro en mi codigo es el total, mientras que en las ecuaciones del LLT
        l = (termino *i) + 1                                                              # no ponen el span sino el semispan, ya que consideran alas simetricas; que no es mi caso
        ecuacionLLT_2[j, i] = (np.sin(l * angle[j])) * (np.sin(angle[j]) + (l * pendiente2D[j] * distribucion_cuerda[j]) / (4 * span))

# Resolver el sistema lineal
A = np.linalg.solve(ecuacionLLT_2, ecuacionLLT) # A sera un vector con los terminos de la solucion del sistema de ecuaciones 

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

# Cálculo de CL
CL = A[0, 0] * np.pi * AR

# Cálculo de CD y angulo de downwash
CD=0
cdi = np.zeros(n) # se puede eliminar es para comprobar
alpha_w = np.zeros(angle.size) # El angulo de downwash
for i in range(n):
    CD += np.pi * AR * (termino * i + 1) * A[i, 0] ** 2
    cdi[i] = np.pi * AR * (termino * i + 1) * A[i, 0] ** 2
    alpha_w += ((termino * i + 1) * A[i, 0] * (np.sin((termino * i + 1) * angle))) / (np.sin(angle))


# Cálculo de distribución de carga (gamma)
gamma = np.zeros(n)
cl = np.zeros(n)
for i in range(n):
    gamma[i] = 0.0
    for j in range(n):
        gamma[i] += 2 * A[j, 0] * np.sin((termino * j + 1) * angle[i])
    gamma[i] *= span
    cl[i] = 2 * gamma[i] / (distribucion_cuerda[i]) # coef de sustentacion de una seccion

# Mostrar resultados
# print("\n" + "="*40)
print("valores LLT")
print("="*40)
print("vector alphadownwash", alpha_w)
print("CL =", CL)
print("CD =", CD)
print("vector cl =", cl)

#verificaciones para cl(theta) = clxfoil evaluado en (alpha inf - alpha w)##############################################################################################
# ------------------------- cálculo de clxfoil -------------------------
punto_evaluado = np.rad2deg(valor_angulo_ataque - alpha_w) # hay que convertirlo a grados 

# ==============================================================================
# BUCLE UNIFICADO: OBTENCIÓN DE CL Y CD
# ==============================================================================

# Como el flap es fijo para todos los puntos, creamos un vector lleno de ese valor
# Si alphas_grados tiene 100 puntos, flaps_grados tendrá 100 veces el valor 5.0
flaps_grados = np.full_like(punto_evaluado, valor_angulo_flap)

# Interpolar (Scipy acepta arrays directamente)
# Pasamos los dos vectores: (alphas, flaps)
raw_cl = interpolador_cl(punto_evaluado, flaps_grados)
raw_cd = interpolador_cd(punto_evaluado, flaps_grados)

# Limpieza de NaNs (Equivalente a un if
# np.nan_to_num convierte los NaN a 0.0 automáticamente
resultados_clxfoil = np.nan_to_num(raw_cl, nan=0.0)
resultados_cdxfoil = np.nan_to_num(raw_cd, nan=0.0)

# Mostrar resultados
print("\nVector de CL xfoil:")
print(resultados_clxfoil)
print("\nVector de CD xfoil:")
print(resultados_cdxfoil)

# Convertir a array de numpy 
resultados_cdxfoil = np.array(resultados_cdxfoil)
# print("\n" + "="*40)
# print("valores cdxfoil verificacion")
# print("="*40)
print("\nVector de cd:")
print(cdi)

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

L_trapz2 = np.trapezoid(integrando_cd, theta)

L_LLT = 0.5 * rho * V**2 * S * CL
D_LLT = 0.5 * rho * V**2 * S * CD


print("LIFT trapecios= ", L_trapz)
print("LIFT simpson= ", L_simp)
print("LIFT spline= ", L_spline)

print("L_llt = ", L_LLT)
print("D_llt = ", D_LLT)
print("Drag = ", L_trapz2)

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