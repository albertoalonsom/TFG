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

#============================================================
# ENTRADAS DEL USUARIO A MODIFICAR PARA CODIGO LIFTING LINE
#==========================================================
###########################################################
# parametros del fluido 
###########################################################
rho = 1.225  # densidad del medio [kg/m^3]
V = 5.14444      # velocidad [m/s]
valor_angulo_ataque = 15 # VALORES EN GRADOS [º]
valor_angulo_flap = 0   # VALRES EN GRADOS [º]
##########################################################
# GEOMETRIA DE LA VELA
#############################################################
cuerda_en_base = 2          # cuerda en la base de la vela [m]
cuerda_en_punta = 1.4       # cuerda en la punta de la vela [m]
span = 5                    # span total de la vela [m]
distancia_mastil_base = 0.5 # Posición del mastil fisico respecto al borde de ataque en la BASE de la vela (m)

# OPCION DE GEOMETRIA DE LA VELA 
# 1 = vela con forma de semielipse que llega hasta el span aunque la elipse sea mas grande
# 2 = vela con forma de trapecio rectangulo
tipo_de_vela = 1 #poner 1 para vela con distribucion de un cuartode elipse y poner 2 para trapezoidal
# en caso de usar la opcion 1 definir los semiejes de la elipse de diseño teorico
semi_eje_menor = cuerda_en_base # el semi eje menor es la cuerda en la base 
semi_eje_mayor = 7

# OPCIÓN DE LOS BORDES DE LA VELA
# 1 = Borde de ataque recto (BORDE ATAQUE RECTO), BORDE DE SALIDA EL DEFINIDO EN GEOMETRIA DE LA VELA
# 2 = Borde de salida recto (BORDE DE SALIDA RECTO), BORDE DE ATAQUE EL DEFINIDO EN GEOMETRIA DE LA VELA
opcion_geometria_vela = 1 

################################################################################
# TERMINOS PARA EL LLT (si no se sabe o entiende que es, dejar como esta)
#############################################################################
# termino = 1 en caso de diseño no simetrico, los terminos pares de la serie de fourier no son nulos y hay que tenerlos en cuenta
# termino = 2 en caso de diseño SIMETRICO, los terminos pares de la serie de fourier son nulos
termino = 1
# cantidad de estaciones para resolver el liftinting line theory
# mejor dejar en numero IMPAR, no poner menos de 5 ni mas de 40 y en caso de fallo bajar el numero 
n = 21 

# ==============================================================================
# VALIDACION DE GEOMETRIA EN CASO DE SELECCIONAR UNA ELIPTICA 
# ==============================================================================

if tipo_de_vela == 1:
    # Calcula la cuerda real en la punta (h = span) según la ecuación de la elipse
    cuerda_real_punta = semi_eje_menor * np.sqrt(1 - (span / semi_eje_mayor)**2)
    
    # avisa al usuario y sobreescribimos la variable para evitar resultados erroneos que no sabemos de donde vienen
    print(f"AVISO: Vela eliptica seleccionada. El valor manual de 'cuerda_en_punta' ({cuerda_en_punta}m) se ignora.")
    print(f"recalculado matematicamente a: {cuerda_real_punta:.3f}m")
    
    cuerda_en_punta = cuerda_real_punta

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
            df["CD"].to_numpy(),        # Columna CD
            df["Cm"].to_numpy()         # Columna Cm
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
    #para extraer los "Valores" (Output): Cm
    valores_cm = datos_array[:, 5]


    # creacion el interpolador
    # rescale=True creo que es necesario aqui porque Alpha (ej. -5 a 15) y Flap (ej. 0 a 40) tienen escalas parecidas,
    # pero si usara el Reynolds, las escalas serían muy distintas
    interpolador_cl = LinearNDInterpolator(puntos_entrada, valores_cl, rescale=True)
    interpolador_cd = LinearNDInterpolator(puntos_entrada, valores_cd, rescale=True)
    interpolador_cm = LinearNDInterpolator(puntos_entrada, valores_cm, rescale=True)
       
else:
    print("No se cargaron datos, revisa la configuración de archivos.")


#======================================================
# CODIGO DEL LIFTING LINE 
#=========================================================
# Conversion de grados a radianes
deg2rad = np.pi / 180
valor_angulo_ataque *= deg2rad
# Inicializacion de vectores para resolver
angle = np.zeros(n)
y = np.zeros(n)

# ==============================================================================
# GENERACIÓN DE LA LÍNEA SUSTENTADORA (GEOMETRÍA)
# ==============================================================================

for i in range(n):
    angle[i] = (i+1) * np.pi / (n+1)
    y[i] = span * 0.5 * np.cos(angle[i])

# diccionario para calcular las cuerdas
def calcular_distribucion_cuerdas(opcion, vector_y):
    """
    para añadir más opciones aquí en el futuro usando 'lambda y:'.
    """
    formulas = {
        # Opción 1: eliptica (añado np.maximum para proteger la raiz cuadrada de numeros negativos)
        1: lambda y_vec: semi_eje_menor * np.sqrt(np.maximum(0, 1 - ((y_vec + (span/2)) / semi_eje_mayor)**2)),
        # Opción 2: trapezoidal
        2: lambda y_vec: ((cuerda_en_punta - cuerda_en_base) / span) * (y_vec + span/2) + cuerda_en_base
    }
    return formulas.get(opcion, formulas[2])(vector_y)

distribucion_cuerda = calcular_distribucion_cuerdas(tipo_de_vela, y)


# ==============================================================================
# CREACIÓN DE LA MATRIZ (Fuera del bucle, solo se calcula una vez)
# ==============================================================================
Matriz_Seno = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        l = (termino * j) + 1
        Matriz_Seno[i, j] = np.sin(l * angle[i])

# ==============================================================================
# BUCLE NLLT  (usando scipy.optimize.fsolve)
# ==============================================================================

LIMITE_MIN_DEG = np.min(valores_alpha)
LIMITE_MAX_DEG = np.max(valores_alpha)

alpha_geom_rad = np.full(n, valor_angulo_ataque)

def calcular_error_aerodinamico(alpha_eff_guess):
    """
    Esta función: Toma un ángulo propuesto, calcula el downwash
    y devuelve la diferencia entre lo que debería ser y lo que propuso.
    ¡Fsolve buscará que esta diferencia sea CERO!
    """
    # Escudo de grados
    alpha_eff_deg = np.rad2deg(alpha_eff_guess)
    alpha_eff_deg_seguro = np.clip(alpha_eff_deg, LIMITE_MIN_DEG, LIMITE_MAX_DEG)
    flaps_deg = np.full(n, valor_angulo_flap)
    
    # Leemos la polar
    cl_2d = interpolador_cl(alpha_eff_deg_seguro, flaps_deg)
    cl_2d = np.nan_to_num(cl_2d, nan=0.0)
    
    # Resolvemos la matriz temporalmente
    RHS = (distribucion_cuerda / (4 * span)) * cl_2d
    A_1D_temp = np.linalg.solve(Matriz_Seno, RHS)
    
    # Calculamos el downwash
    alpha_w_nuevo = np.zeros(n)
    for i in range(n):
        suma_downwash = 0
        for j in range(n):
            l = (termino * j) + 1
            suma_downwash += l * A_1D_temp[j] * (np.sin(l * angle[i]) / np.sin(angle[i]))
        alpha_w_nuevo[i] = suma_downwash
        
    # Ángulo que dicta la física vs Ángulo que propusimos
    alpha_eff_calc = alpha_geom_rad - alpha_w_nuevo
    return alpha_eff_calc - alpha_eff_guess

print("\nIniciando convergencia NLLT con fsolve (Autopiloto)...")

# Le damos el punto de partida (El ángulo geométrico inicial)
alpha_inicial_rad = np.copy(alpha_geom_rad)

# datos de lo que me devuelve el fsolve
alpha_eff_rad_final, info_dict, ier, mesg = fsolve(calcular_error_aerodinamico, alpha_inicial_rad, full_output=True)

if ier == 1:
    print("Convergencia NLLT alcanzada con EXITOOOO por SciPy")
else:
    print("Aviso del solver:", mesg)

# Extraemos el vector de errores (residuos) y buscamos el más grande
vector_errores = info_dict["fvec"]
error_maximo = np.max(np.abs(vector_errores))

print(f"Error máximo de fsolve: {error_maximo:.8e} radianes")

# ==============================================================================
# RECONSTRUCCIÓN FINAL DE VARIABLES PARA EL RESTO DEL CÓDIGO
# ====================================================================
# Ahora que ya sabemos el ángulo efectivo perfecto, hacemos el cálculo una última vez
# para guardar los coeficientes A_1D y el cl_2d y pasárselos a las integrales
alpha_eff_deg = np.rad2deg(alpha_eff_rad_final)
cl_2d = np.nan_to_num(interpolador_cl(np.clip(alpha_eff_deg, LIMITE_MIN_DEG, LIMITE_MAX_DEG), np.full(n, valor_angulo_flap)), nan=0.0)
RHS_final = (distribucion_cuerda / (4 * span)) * cl_2d
A_1D = np.linalg.solve(Matriz_Seno, RHS_final)

# Cre0 las variables exactas que uso en el resto del código
alpha_w_nuevo = alpha_geom_rad - alpha_eff_rad_final
A = A_1D.reshape((n, 1))

print(f"\nRango de ángulo efectivo final: min {np.min(alpha_eff_deg):.2f}º, max {np.max(alpha_eff_deg):.2f}º")

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

def seleccionar_forma_vela(opcion):
    """
    geometrías de vela 
    """
    modelos = {
        1: partial(forma_eliptica_general, root_chord=cuerda_en_base, span_teorico=semi_eje_mayor),
        2: partial(forma_trapezoidal_general, root_chord=cuerda_en_base, tip_chord=cuerda_en_punta, span_teorico=span)
        # 3: partial(forma_rectangular, root_chord=cuerda_en_base, span_teorico=span)
    }
    # si el usuario mete la opcion 1 o 2, la saca. en casso de que meta un 8 por error, usa la 2 por defecto
    return modelos.get(opcion, modelos[2])

modelo_seleccionado_para_calculo = seleccionar_forma_vela(tipo_de_vela)

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

# Reciclamos el downwash que ya calcule en el bucle iterativo NLLT
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

# ==============================================================================
# OBTENCIÓN DEL CD VISCOSO FINAL (El Cl ya convergió en el bucle NLLT)
# ==============================================================================

# Usamos directamente el ángulo efectivo en grados que salió del bucle 'while'
flaps_grados = np.full_like(alpha_eff_deg, valor_angulo_flap) # creo un vector con todo valor angulo flap pero de la misma dimension que alpha_eff_deg

# Interpolar SOLO el Cd, porque el Cl ya obligamos a que fuera igual en el bucle
raw_cd = interpolador_cd(alpha_eff_deg, flaps_grados)
resultados_cdxfoil = np.nan_to_num(raw_cd, nan=0.0)

# Para la integral, el Cl de XFOIL es simplemente el que usamos en la última iteración del bucle
resultados_clxfoil = cl_2d 

#========================================================
# INTEGRACION DEL LIFT Y DRAG
#=======================================================

# datos 
theta =np.array(angle)  
C_integra= np.array(distribucion_cuerda)                    
Cl_integra = np.array(resultados_clxfoil) 
Cd_integra = np.array(resultados_cdxfoil)                    

# Funcion integrando
integrando_cl = C_integra * Cl_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)
integrando_cd = C_integra * Cd_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)

# Integracion con distintas reglas para verificar y ver errores
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
# print("error")
# print(error)

# ================
# CALCULO DE MOMENTOS ESTRUCTURALES: TORSION, ESCORA Y CABECEO
# basado en la integracion 3D del Producto Vectorial
# ================================================

#FUERZAS LOCALES (proyeccion estricta de lift y drag)
alpha_eff_rad = np.deg2rad(alpha_eff_deg)
# a) Fuerza normal (perpendicular a la cuerda)
integrando_Normal = integrando_cl * np.cos(alpha_eff_rad) + integrando_cd * np.sin(alpha_eff_rad)
# b) Fuerza axial (paralela a la cuerda, positiva hacia el borde de fuga)
#   Esta es la fuerza que empuja/frena el barco longitudinalmente
integrando_Axial = integrando_cd * np.cos(alpha_eff_rad) - integrando_cl * np.sin(alpha_eff_rad)


# Diccionario con las fórmulas de la línea del borde de ataque
geometrias_xle = {
    1: lambda: np.zeros_like(distribucion_cuerda),       # Opción 1: Borde de Ataque Recto (Nariz en X=0)
    2: lambda: cuerda_en_base - distribucion_cuerda      # Opción 2: Borde de Salida Recto (Nariz retrocede)
}

# Aviso de seguridad en caso de que el usuario meta un número raro
if opcion_geometria_vela not in geometrias_xle:
    print("Aviso: Opción de geometría inválida. Usando Borde de Ataque Recto por defecto.")

# Extraemos y ejecutamos () la fórmula elegida (por defecto usamos la 1 si hay fallo)
x_LE = geometrias_xle.get(opcion_geometria_vela, geometrias_xle[1])()

#VECTORES DE POSICION (brazos de palanca) mi eje global esta situado en el borde de ataque en la base de la vela
# El Centro Aerodinámico Global en el espacio 3D (a un 25% de la nariz local)
centro_aerodinamico_global = x_LE + (0.25 * distribucion_cuerda)
# El mástil es recto. Lo fijamos a la distancia elegida desde la nariz de la base (X=0)
posicion_mastil_global = np.full_like(y, distancia_mastil_base)
# a) Componente X de r: Brazo horizontal para la Torsión
brazo_horizontal = posicion_mastil_global - centro_aerodinamico_global 
# b) Componente Z de r: Brazo vertical para la Escora (Altura desde la base)
brazo_vertical = y + (span/2)                          # distancia Z (altura desde la base)

#INTEGRANDOS DE LOS 3 MOMENTOS (las 3 componentes de r x F)
Cm_vector = interpolador_cm(alpha_eff_deg, valor_angulo_flap)
integrando_Cm_puro = 0.5 * rho * (V**2) * (distribucion_cuerda**2) * Cm_vector * (span/2) * np.sin(theta)
#   Eje 1: Torsion (pitching local de la vela / ajuste de angulo)
integrando_Torsion = integrando_Cm_puro + (integrando_Normal * brazo_horizontal)
#   Eje 2: Escora (roll / vuelco lateral)
integrando_Escora = integrando_Normal * brazo_vertical
#   Eje 3: Cabeceo (pitch del barco / flexión longitudinal del mastil)
integrando_Cabeceo = integrando_Axial * brazo_vertical

#INTEGRACION NUMERICA (simpson)
momento_Torsion = simpson(integrando_Torsion, x=theta)
momento_Escora = simpson(integrando_Escora, x=theta)
momento_Cabeceo = simpson(integrando_Cabeceo, x=theta)

# RESULTADOS EN TERMINAL
print("ANALISIS ESTRUCTURAL 3D DE MOMENTOS")
print(f"Momento TORSION en el mastil: {momento_Torsion:.2f} Nm") #TORSION (Giro del mastil sobre si mismo / Trimado de AOA
print(f"Esfuerzo flector/escora en la base: {momento_Escora:.2f} Nm") #ESCORA (Vuelco lateral / Flexion transversal
print(f"Momento en la base/trimado, si sale negativo dobla mastil hacia proa: {momento_Cabeceo:.2f} Nm") #CABECEO (Empuje longitudinal / Flexion hacia atras

# #graficas 
# plt.plot(y, cdi)
# plt.xlabel("span")
# plt.ylabel("cl")
# plt.title("titulo")
# plt.grid(True)
# plt.show()

# ==========================================
# GRAFICA 3D DE CARGAS ESTRUCTURALES EN EL MASTIL
# ====================================================================

# quitar el Jacobiano para obtener la distribucion real (Nm/m) en el eje fisico (y)
jacobiano = (span/2) * np.sin(theta)

# uso np.divide para evitar dividir por cero en los extremos (donde el jacobiano es 0)
dM_dy_Torsion = np.divide(integrando_Torsion, jacobiano, out=np.zeros_like(integrando_Torsion), where=jacobiano!=0)
dM_dy_Escora  = np.divide(integrando_Escora, jacobiano, out=np.zeros_like(integrando_Escora), where=jacobiano!=0)
dM_dy_Cabeceo = np.divide(integrando_Cabeceo, jacobiano, out=np.zeros_like(integrando_Cabeceo), where=jacobiano!=0)

# configuro la grafica
plt.figure(figsize=(10, 6))

# dibuja las tres curvas (y va de -2.5 a 2.5)
plt.plot(y, dM_dy_Escora, color='blue', linestyle='-', linewidth=2.5, 
         label='Escora (Contribución al vuelco lateral)')
plt.plot(y, dM_dy_Cabeceo, color='green', linestyle='--', linewidth=2.5, 
         label='Cabeceo (Contribución a la flexión trasera)')
plt.plot(y, dM_dy_Torsion, color='red', linestyle='-.', linewidth=2.5, 
         label='Torsión (Contribución al giro en su eje)')

# detalles esteticos 
plt.title('Distribución de Cargas Aerodinámicas en el Mástil', fontsize=15, fontweight='bold')
plt.xlabel('Posición en la vela (y) [m]   (-2.5 = Base, 2.5 = Punta)', fontsize=12)
plt.ylabel('Carga Local aportada a la base (dM/dy) [Nm/m]', fontsize=12)

# lineas de referencia (ejes neutros)
plt.axhline(0, color='black', linewidth=1.5) # eje horizontal en 0
plt.axvline(0, color='gray', linestyle=':', alpha=0.7) # centro del msstil

# caja resumen con los resultados integrados
texto_resumen = (
    f"MOMENTOS TOTALES EN LA BASE:\n"
    f"  • Escora: {momento_Escora:.1f} Nm\n"
    f"  • Cabeceo: {momento_Cabeceo:.1f} Nm\n"
    f"  • Torsión: {momento_Torsion:.1f} Nm"
)
plt.annotate(texto_resumen, xy=(0.03, 0.75), xycoords='axes fraction',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=1.5, alpha=0.9),
             fontsize=11, fontweight='bold', color='black')

# rematar estetica y mostrar
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10, shadow=True)
plt.tight_layout() # ajusta msrgenes automaticamente
plt.show()


# ===========================================================
# [BLOQUE EXTRA] busqueda del punto donde la torsion del mastil es cero
# ================================

# funcion para calcular el torqie en cualquier posicion
def calcular_torque_residual(pos):
    brazo_prueba = np.full_like(centro_aerodinamico_global, pos) - centro_aerodinamico_global
    return simpson(integrando_Cm_puro + (integrando_Normal * brazo_prueba), x=theta)

# encutro el punto exacto con fsolve (dato de partida es 0.5m)
posicion_optima = fsolve(calcular_torque_residual, x0=0.5)[0]
print(f" POSICION MASTIL OPTIMO (Torque 0.00 Nm): a {posicion_optima:.3f} metros del borde de ataque en la base.")

# creo datos para la grafica 
posiciones_grafica = np.linspace(0.0,max(cuerda_en_base,cuerda_en_punta), 30)
torques_grafica = [calcular_torque_residual(p) for p in posiciones_grafica]

# dibuja la grafica 
plt.figure(figsize=(7, 4))
plt.plot(posiciones_grafica, torques_grafica, 'b-', linewidth=2, label='Esfuerzo del Motor')

# marca la linea del cero y el punto optimo
plt.axhline(0, color='black', linestyle='--', linewidth=1.5) # Eje neutro (0 Nm)
plt.plot(posicion_optima, 0, 'ro', markersize=8, label=f'Óptimo: {posicion_optima:.3f} m')

# estetica de la grafica ejes y tal
plt.title('Torque vs Posición del Mástil', fontsize=12, fontweight='bold')
plt.xlabel('Distancia del mástil desde la nariz [m]')
plt.ylabel('Torque de Torsión [Nm]')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()