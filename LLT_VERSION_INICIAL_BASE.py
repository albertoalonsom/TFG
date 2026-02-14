import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy import integrate

#funcion para encontrar valores en el dataframe###################################################################################

# ------------------------- función encontrar_cercanos -------------------------
def encontrar_cercanos(df, columna_buscar, columna_devolver, valor_buscado, n_cercanos=2, tol_exact=1e-6):
    """
    Devuelve un dict con:
      - 'exacto': fila (pd.Series) si existe un valor dentro de tol_exact, o None.
      - 'cercanos': pd.DataFrame con los n_cercanos más cercanos (si existiera exacto, se excluye de los 'cercanos').
      - 'indices': index de los elementos devueltos en 'cercanos'.
    """
    # Comprobaciones básicas
    if df is None or df.shape[0] == 0:
        raise ValueError("DataFrame vacío.")
    if columna_buscar not in df.columns:
        raise KeyError(f"Columna '{columna_buscar}' no encontrada en el DataFrame")
    if columna_devolver not in df.columns:
        raise KeyError(f"Columna '{columna_devolver}' no encontrada en el DataFrame")
    if n_cercanos < 1:
        raise ValueError("n_cercanos debe ser >= 1")

    # diferencias absolutas
    diffs = (df[columna_buscar] - valor_buscado).abs()
    idx_min = diffs.idxmin()
    min_diff = diffs.loc[idx_min]
    es_exacto = min_diff <= tol_exact

    resultado = {'exacto': None, 'cercanos': pd.DataFrame(), 'indices': pd.Index([])}

    if es_exacto:
        resultado['exacto'] = df.loc[idx_min]
        # Excluir el exacto para la selección de cercanos (como en tu código original)
        diffs_sin_exacto = diffs.drop(idx_min)
        # Si hay menos filas que n_cercanos, nsmallest devuelve las que existan
        indices_cercanos = diffs_sin_exacto.nsmallest(n_cercanos).index
    else:
        indices_cercanos = diffs.nsmallest(n_cercanos).index

    # Guardar índices y DataFrame de cercanos; reset_index para acceder por .iloc de forma fiable
    resultado['indices'] = indices_cercanos
    resultado['cercanos'] = df.loc[indices_cercanos].copy().reset_index(drop=True)

    return resultado

# ------------------------- lectura y creación del DataFrame -------------------------

file_path = r"C:\Users\aamal\OneDrive - Universidad Politécnica de Madrid\Escritorio\pruebas_tfg\xflr5_2D_prubas\NACA 0020_Re0.462.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# columnas y filas a eliminar (tal y como tenías)
columnas_a_eliminar = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
filas_a_eliminar = [0, 1, 2, 3, 4, 5, 6]
nombres_columnas = ["alpha", "CL", "CD", "Cm", "Xcp", "CDp"]

datos = []
for i, line in enumerate(lines):
    if i in filas_a_eliminar:
        continue
    valores = line.strip().split()
    if not valores:
        continue
    try:
        nueva_linea = [float(valor) for idx, valor in enumerate(valores) if idx not in columnas_a_eliminar]
        if len(nueva_linea) == len(nombres_columnas):
            datos.append(nueva_linea)
    except ValueError:
        continue

df = pd.DataFrame(datos, columns=nombres_columnas)

# Comprobar que el DataFrame tiene datos
if df.empty:
    raise RuntimeError("El DataFrame resultante está vacío. Revisa el archivo y las columnas/filas que eliminas.")

# ------------------------- interpolador lineal (si lo necesitas) -------------------------
interpoladorlinealCL = interp1d(df["alpha"], df["CL"], kind='linear', fill_value="extrapolate")

# ------------------------- ejemplo: cálculo de pendiente en un AOA elegido -------------------------
valor_AOA_deseado = 15
res_pend = encontrar_cercanos(df, "alpha", "CL", valor_buscado=valor_AOA_deseado, n_cercanos=4)

cercanos_pend = res_pend.get('cercanos', pd.DataFrame())
if len(cercanos_pend) < 2:
    raise RuntimeError("No hay suficientes puntos para calcular la pendiente en el AOA deseado")

# acceder de forma segura por posición (.iloc)
a0 = cercanos_pend.iloc[1]
a1 = cercanos_pend.iloc[0]
alpha_cercano1 = float(a0['alpha'])
alpha_cercano2 = float(a1['alpha'])
CL_cercano1    = float(a0['CL'])
CL_cercano2    = float(a1['CL'])

print(alpha_cercano1,alpha_cercano2,CL_cercano1,CL_cercano2)

pendiente = (CL_cercano2 - CL_cercano1) / np.deg2rad(alpha_cercano2 - alpha_cercano1)
Cl_geo = CL_cercano1 + pendiente * np.deg2rad(valor_AOA_deseado-alpha_cercano1)

# ------------------------- cálculo del zero-lift angle (CL = 0) (comportamiento original) -------------------------
valor_cl_cero = 0.0
res_zero = encontrar_cercanos(df, "CL", "alpha", valor_buscado=valor_cl_cero, n_cercanos=2)

if res_zero['exacto'] is not None:
    # existe CL == 0 exacto -> usamos ese alpha
    valor_cero_lift_angle_2D = float(res_zero['exacto']['alpha'])
else:
    cercanos_zero = res_zero.get('cercanos', pd.DataFrame())
    if len(cercanos_zero) < 2:
        raise RuntimeError("No hay suficientes puntos para interpolar CL=0")
    z0 = cercanos_zero.iloc[0]
    z1 = cercanos_zero.iloc[1]
    CL1 = float(z0['CL'])
    alpha1 = float(z0['alpha'])
    CL2 = float(z1['CL'])
    alpha2 = float(z1['alpha'])

    valor_cero_lift_angle_2D = alpha1 + (alpha2 - alpha1) * (valor_cl_cero - CL1) / (CL2 - CL1)

# ------------------------- salida (prints opcionales) -------------------------
print(f"Pendiente (a partir de {alpha_cercano1}°, {alpha_cercano2}°): {pendiente}")
print(f"Alpha de zero-lift (2D) para CL=0: {valor_cero_lift_angle_2D}°")



    
    #print(f"Interpolación entre:")
    #print(f"   Punto 1: CL={CL1}, alpha={alpha1}")
    #print(f"   Punto 2: CL={CL2}, alpha={alpha2}")
    #print(f"Valor alpha interpolado para CL={valorCLbuscar}: {valor_cero_lift_angle_2D}")

# print(f"alpha_cercanoZEROlift = {valor_cero_lift_angle_2D}")

#####################################################################################################################################################


#codigo del lifting line 

#####################################################################################################################################################
 # Entradas del usuario
cuerda_en_base = 2 #float(input("valor root chord: "))
cuerda_en_punta = 1.4 #float(input("valor tip chord: "))
span = 5 #float(input("valor Span2: "))
twistroot = 0 #float(input("valor Root twist angle in degrees: "))    # en nuestros casos simpre 0
twisttip = 0 #float(input("valor tip twist angle in degrees: "))      # en nuestros casos simpre 0
curve_slope2D = pendiente #float(input("valor root lift curve slope in units/radian: "))       #corregir con el 2D
#a0tip = np.pi*2 #float(input("valor lift curve slope at the tip in units/radian: "))  #corregir con el 2D
AOA = valor_AOA_deseado #float(input("valor angle of attack, in degrees: "))  
zero_lift_angle2D = valor_cero_lift_angle_2D #float(input("valor zero-lift angle at the root: "))    #mirar en 2D 
#alpha0tip = 0 # float(input("valor zero lift angle at the tip: "))     #mirar en 2D 

#valores para la distribucion de cuerda de una vela semieliptica
semiancho = 2
semilargo= 7

#==================================================================================================#
#terminos a cambiar si es ala simetrica o asimetrica 
#==================================================================================================#
#area simpson
#areasim = 0.5 #para caso simetrico 
areasim = 1 #para caso asimetrico

#terminos pares serie 
#termino = 2 # ala simetrica elimina terminos pares de la serie q son nulos 
termino = 1 # vela no es simetrica y hay q tener en cuenta los terminos pares

# Conversión de grados a radianes
deg2rad = np.pi / 180
twistroot *= deg2rad
twisttip *= deg2rad
AOA *= deg2rad
zero_lift_angle2D *= deg2rad
#alpha0tip *= deg2rad

n = 10 # número de estaciones dejar mejor un un numero impar 

# Inicialización de vectores para resolver 
angle = np.zeros(n)
y = np.zeros(n)
c = np.zeros(n)
cl = np.zeros(n)
alp = np.zeros(n)
a = np.zeros(n)
b=np.zeros(n)
ecuacionLLT = np.zeros((n, 1))
ecuacionLLT_2 = np.zeros((n, n))

# Definición de propiedades en estaciones
for i in range(n):
    angle[i] = (i+1) * np.pi / (n+1) #va desde 0 hasta pi sin incluir 0 y pi ya que conduce a una matriz singular q no se puede resolver 
    y[i] = span * 0.5 * np.cos(angle[i]) 
    #y[i] = np.abs(y[i]) # para poner positivos los y[i] terminos

    #c[i] = semiancho * (np.sqrt(1-((y[i])/semilargo)**2))                 # para hacerla simetrica y validar con xflr5
    c[i] = semiancho * (np.sqrt(1-((y[i]+2.5)/semilargo)**2))              # debido al distribucion eliptica del ala y desplazada por situar el eje el en medio de la vela

    #alp[i] = alpha + twistroot - (alpha0root + (alpha0tip - alpha0root + twistroot - twisttip) * y[i] * 2 / span)        #para vela simetrica sin flap es innecesario
    # alp[i] = AOA - zero_lift_angle2D # aplha=angulo ataque

    #a[i] = a0root + (a0tip - a0root) * y[i] * 2 / span               #para vela simetrica sin flap es innecesario
    a[i] = curve_slope2D #pendiente curva 2D

    b[i] = Cl_geo

print("Cl_geo:", Cl_geo)
# Sistema de ecuaciones lineales
for j in range(n):
    ecuacionLLT[j, 0] = ((b[j] * c[j]) / (4 * span)) * np.sin(angle[j])
    for i in range(n):
        l = (termino *i) + 1
        ecuacionLLT_2[j, i] = (np.sin(l * angle[j])) * (np.sin(angle[j]) + (l * a[j] * c[j]) / (4 * span))

# Resolver el sistema lineal
A = np.linalg.solve(ecuacionLLT_2, ecuacionLLT)

# ================================
# CÁLCULO DE SUPERFICIE ALAR Y AR
# ================================
def f(x):
    return semiancho * np.sqrt(1 - (x / semilargo)**2)

# Integración numérica (Simpson)
x0, x1 = 0, span*areasim 
k = 800
h = (x1 - x0) / k
x = np.linspace(x0, x1, k + 1)
fun = f(x)

Simp = fun[0] + fun[-1] + 4 * np.sum(fun[1:k:2]) + 2 * np.sum(fun[2:k-1:2])
area = (h / 3) * Simp
S = area / areasim
AR = span ** 2 / S


# Cálculo de CL
CL = A[0, 0] * np.pi * AR

# Cálculo de CD y angulo de downwash
CD=0
alpha_w = np.zeros(angle.size)
for i in range(n):
    CD += np.pi * AR * (termino * i + 1) * A[i, 0] ** 2
    alpha_w += ((termino * i + 1) * A[i, 0] * (np.sin((termino * i + 1) * angle))) / (np.sin(angle))


# Cálculo de distribución de carga (gamma)
gamma = np.zeros(n)
cl = np.zeros(n)
for i in range(n):
    gamma[i] = 0.0
    for j in range(n):
        gamma[i] += 2 * A[j, 0] * np.sin((termino * j + 1) * angle[i])
    gamma[i] *= span
    cl[i] = 2 * gamma[i] / (c[i])

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
punto_evaluado = AOA - alpha_w

resultados_clxfoil = []
for i in range(n):  
    valoralphaxfoilbuscar = 180 * punto_evaluado[i] / np.pi
    resultado = encontrar_cercanos(df, "alpha", "CL", valoralphaxfoilbuscar, n_cercanos=2)

    if resultado['exacto'] is not None:
        valorclxfoil = float(resultado['exacto']['CL'])
        resultados_clxfoil.append(valorclxfoil)  
        # print(f"[{i}] Valor exacto: {valorclxfoil}")
    else:
        # Acceso seguro con .iloc
        CL11 = float(resultado['cercanos'].iloc[0]['CL'])
        alpha11 = float(resultado['cercanos'].iloc[0]['alpha'])
        CL21 = float(resultado['cercanos'].iloc[1]['CL'])
        alpha21 = float(resultado['cercanos'].iloc[1]['alpha'])

        # Interpolación lineal
        valor = ((valoralphaxfoilbuscar - alpha11) / (alpha21 - alpha11)) * (CL21 - CL11) + CL11
        resultados_clxfoil.append(valor)  

# Convertir a array de numpy 
resultados_clxfoil = np.array(resultados_clxfoil)
print("\nVector de clxfoil:")
print(resultados_clxfoil)


# ------------------------- cálculo de cdxfoil -------------------------
resultados_cdxfoil = []
for i in range(n):  
    valoralphaxfoilbuscar = 180 * punto_evaluado[i] / np.pi
    resultado = encontrar_cercanos(df, "alpha", "CD", valoralphaxfoilbuscar, n_cercanos=2)

    if resultado['exacto'] is not None:
        valorcdxfoil = float(resultado['exacto']['CD'])
        resultados_cdxfoil.append(valorcdxfoil)  
        # print(f"[{i}] Valor exacto: {valorcdxfoil}")
    else:
        # Acceso seguro con .iloc
        CD11 = float(resultado['cercanos'].iloc[0]['CD'])
        alpha111 = float(resultado['cercanos'].iloc[0]['alpha'])
        CD21 = float(resultado['cercanos'].iloc[1]['CD'])
        alpha211 = float(resultado['cercanos'].iloc[1]['alpha'])
        
        # Interpolación lineal
        valor = ((valoralphaxfoilbuscar - alpha111) / (alpha211 - alpha111)) * (CD21 - CD11) + CD11
        resultados_cdxfoil.append(valor)  

# Convertir a array de numpy
resultados_cdxfoil = np.array(resultados_cdxfoil)
print("\nVector de cdxfoil:")
print(resultados_cdxfoil)
 

        #print(f"[{i}] Interpolación -> clxfoil={valor}")

# Convertir a array de numpy 
resultados_cdxfoil = np.array(resultados_cdxfoil)
# print("\n" + "="*40)
# print("valores cdxfoil verificacion")
# print("="*40)
print("\nVector de cdxfoil:")
print(resultados_cdxfoil)

############################################################################
#============================================================================================
#==================================integracion lift

# datos 
theta =np.array(angle)  
C_integra= np.array(c)                    
Cl_integra = np.array(resultados_clxfoil)                    

# Parámetros físicos
rho = 1.225  # densidad del aire [kg/m^3]
V = 7       # velocidad [m/s]

# Función integrando
integrand = C_integra * Cl_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)

# Integración con regla del trapecio
L_trapz = np.trapezoid(integrand, theta)


print("LIFT = ", L_trapz)

#=================================integracion DRAG

# datos                
Cd_integra = np.array(resultados_cdxfoil)                    

# Función integrando
integrand2 = C_integra * Cd_integra * 0.5 * (span/2) * rho * V**2 * np.sin(theta)

# Integración con regla del trapecio
L_trapz2 = np.trapezoid(integrand2, theta)


print("Drag = ", L_trapz2)

###############################################################################################################################

error = (cl - resultados_clxfoil)/resultados_clxfoil
print("error")
print(error)

# # # Graficar distribución de cl
# plt.plot(y, cl)
# plt.xlabel("y")
# plt.ylabel("cl")
# plt.title("Sectional lift coefficient distribution")
# plt.grid(True)

# #graficas de downwash frente a spam 
# plt.plot(y, alpha_w)
# plt.xlabel("y")
# plt.ylabel("angulow")
# plt.title("downwash distribution")
# plt.grid(True)

# # Graficar las dos curvas en la misma figura
# plt.plot(y, cl, label="cl vs y")
# plt.plot(y, alpha_w, label="alpha_w vs y")

# # Etiquetas y título
# plt.xlabel("y")
# plt.ylabel("Valores")
# plt.title("Distribución de cl y alpha_w")
# plt.grid(True)
# plt.legend()  # Para mostrar la leyenda

# plt.figure()
# plt.plot( df ["alpha"], df ["CL"], ".")
# plt.plot( vectoralp, vectorCL)
# plt.show()

