import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def get_char_value(char):
   """Mapea caracteres a valores según la tabla proporcionada"""
   char_map = {
       '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
       '.': 10, ',': 11, '$': 12, '¢': 13, '/': 14, '#': 15, '°': 16, '(': 17, ')': 18, '%': 19,
       '—': 20, '<': 21, '>': 22, '+': 23, '-': 24, '=': 25, '*': 26, '^': 27, '"': 28, ';': 29,
       '\\': 30, '|': 31, '[': 32, ']': 33, '{': 34, '}': 35, '@': 36, '&': 37, '_': 38, '¿': 39,
       '?': 40, '¡': 41, '!': 42, '~': 43, '`': 44, "'": 45, ':': 46, '©': 47, '®': 48, '™': 49,
       'Ó': 50, 'Á': 51, 'Ú': 52, 'Ü': 53, 'Ñ': 54, 'W': 55, 'X': 56, 'Z': 57, 'Y': 58, 'Q': 59,
       'U': 60, 'K': 61, 'H': 62, 'O': 63, 'G': 64, 'F': 65, 'L': 66, 'J': 67, 'E': 68, 'V': 69,
       'T': 70, 'I': 71, 'R': 72, 'M': 73, 'N': 74, 'S': 75, 'B': 76, 'D': 77, 'P': 78, 'C': 79,
       'A': 80, 'w': 81, 'k': 82, 'ü': 83, 'ú': 84, 'y': 85, 'x': 86, 'ñ': 87, 'ó': 88, 'q': 89,
       'j': 90, 'é': 91, 'v': 92, 'f': 93, 'z': 94, 'h': 95, 'í': 96, 'g': 97, 'á': 98, 'p': 99,
       'b': 100, 'u': 101, 'd': 102, 'm': 103, 'l': 104, 't': 105, 'c': 106, 'n': 107, 'o': 108,
       'i': 109, 's': 110, 'r': 111, 'a': 112, 'e': 113
   }
   return char_map.get(char, 0)

def convert_line_to_values(line):
   """Convierte línea de texto a valores numéricos (sin espacios ni indentaciones)"""
   # Eliminar todos los espacios en blanco de la línea
   line = ''.join(line.split())
   return [get_char_value(char) for char in line]

def calculate_mean(values):
   """Calcula la media"""
   return sum(values) / len(values)

def calculate_variance(values):
   """Calcula la varianza muestral"""
   mean = calculate_mean(values)
   squared_deviations = [(x - mean) ** 2 for x in values]
   return sum(squared_deviations) / (len(values) - 1)

def calculate_std_dev(variance):
   """Calcula la desviación estándar"""
   return math.sqrt(variance)

def calculate_skewness(values):
   """Calcula la asimetría (skewness)"""
   mean = calculate_mean(values)
   n = len(values)
   std_dev = calculate_std_dev(calculate_variance(values))
   
   if std_dev == 0:
       return 0
   
   moment3 = sum(((x - mean) / std_dev) ** 3 for x in values)
   return moment3 / n

def calculate_percentiles(values):
   """Calcula percentiles 25, 50, 75"""
   sorted_values = sorted(values)
   n = len(sorted_values)
   
   def percentile(p):
       index = (p / 100) * (n - 1)
       lower = int(index)
       upper = min(lower + 1, n - 1)
       weight = index - lower
       return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
   
   return {
       'p25': percentile(25),
       'p50': percentile(50),
       'p75': percentile(75)
   }

def calculate_iqr(p25, p75):
   """Calcula rango intercuartil"""
   return p75 - p25

def analyze_line(line):
    """Analiza una línea de texto y retorna estadísticas"""
    values = convert_line_to_values(line)
    
    if len(values) < 2:
        return None
    
    mean = calculate_mean(values)
    variance = calculate_variance(values)
    std_dev = calculate_std_dev(variance)
    skewness = calculate_skewness(values)
    percentiles = calculate_percentiles(values)
    iqr = calculate_iqr(percentiles['p25'], percentiles['p75'])
    
    return {
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'skewness': skewness,
        'p25': percentiles['p25'],
        'p50': percentiles['p50'],
        'p75': percentiles['p75'],
        'iqr': iqr,
        'count': len(values)
    }

def process_lines(lines):
   """Procesa múltiples líneas y muestra resultados"""
   # Encabezados con ancho fijo
   print(f"{'Media':<10}{'Varianza':<12}{'Desv.Est':<10}{'Asimetría':<12}{'P25':<8}{'P50':<8}{'P75':<8}{'IQR':<8}{'Conteo':<8}")
   print("-" * 80)

   for i, line in enumerate(lines):
       result = analyze_line(line)
       if result:
           # Valores con ancho fijo
           print(f"{result['mean']:<10.2f}{result['variance']:<12.2f}{result['std_dev']:<10.2f}{result['skewness']:<12.2f}{result['p25']:<8.2f}{result['p50']:<8.2f}{result['p75']:<8.2f}{result['iqr']:<8.2f}{result['count']:<8}")
       else:
           print(f"Línea {i+1}: Datos insuficientes")

def prepare_features_for_clustering(results):
   """Prepara features para DBSCAN"""
   features = []
   for result in results:
       if result:
           features.append([
               result['p25'],
               result['p50'], 
               result['p75'],
               result['std_dev']
           ])
   return np.array(features)

def cluster_lines(lines):
    """Aplica DBSCAN a las líneas con información de debug"""
    print("DEBUG: Activando función cluster_lines()")
    results = [analyze_line(line) for line in lines]
    print(f"DEBUG: Total de líneas analizadas: {len(results)}")
    
    valid_results = [r for r in results if r is not None]
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    print(f"DEBUG: Líneas válidas: {len(valid_results)}")
    print(f"DEBUG: Índices válidos: {valid_indices}")
    
    if len(valid_results) < 2:
        print("DEBUG: No hay suficientes líneas válidas para agrupar.")
        return [], []
    
    features = prepare_features_for_clustering(valid_results)
    print("DEBUG: Features sin normalizar:")
    print(features)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("DEBUG: Features normalizados:")
    print(features_scaled)
    
    n_samples = len(features_scaled)
    eps = 0.5  # parámetro de densidad ajustable
    min_samples = max(2, n_samples // 10)
    print(f"DEBUG: n_samples: {n_samples}, eps: {eps}, min_samples: {min_samples}")
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(features_scaled)
    print("DEBUG: Etiquetas de clustering resultantes:")
    print(labels)
    
    return labels, valid_indices

def identify_table_lines(lines):
   """Identifica líneas de tabla"""
   labels, valid_indices = cluster_lines(lines)
   
   if len(labels) == 0:
       print("No se encontraron clusters.")
       return []
   
   # Encontrar cluster más grande (excluyendo ruido -1)
   unique_labels = [l for l in set(labels) if l != -1]
   if not unique_labels:
       print("No se encontraron clusters válidos.")
       return []
   
   cluster_sizes = {label: list(labels).count(label) for label in unique_labels}
   print(f"Tamaños de clusters: {cluster_sizes}")
   main_cluster = max(cluster_sizes, key=cluster_sizes.get)
   print(f"Cluster principal: {main_cluster}")
   
   # Devolver índices originales de líneas de tabla
   table_indices = [valid_indices[i] for i, label in enumerate(labels) if label == main_cluster]
   print(f"Índices de líneas agrupadas: {table_indices}")
   
   return table_indices

def process_lines_with_clustering(lines):
    """Procesa líneas con clustering y señala las líneas agrupadas"""
    print(f"{'Índice':<8}{'Media':<10}{'Varianza':<12}{'Desv.Est':<10}{'Asimetría':<12}{'P25':<8}{'P50':<8}{'P75':<8}{'IQR':<8}{'Conteo':<8}")
    print("-" * 90)
    
    for i, line in enumerate(lines):
        result = analyze_line(line)
        if result:
            print(f"{i:<8}{result['mean']:<10.2f}{result['variance']:<12.2f}{result['std_dev']:<10.2f}{result['skewness']:<12.2f}{result['p25']:<8.2f}{result['p50']:<8.2f}{result['p75']:<8.2f}{result['iqr']:<8.2f}{result['count']:<8}")
        else:
            print(f"{i:<8}Línea {i+1}: Datos insuficientes")
    
    # Identificar líneas de tabla
    table_indices = identify_table_lines(lines)
    print(f"\nÍndices de líneas agrupadas por DBSCAN: {table_indices}")
    print("Líneas agrupadas (contenido):")
    for idx in table_indices:
        print(f"{idx}: {lines[idx]}")
    
    return table_indices

# Ejemplo de uso:
if __name__ == "__main__":
   # Aquí defines tus líneas de texto
   lines = [
      "FEDERICO GOMEZ6 ZUMPANGO EDO MEX CP55600",
      "JOSE MENDEZ DAVILA R.F.C.MEDJ630711HU5",
      "REGIMEN SIMPLIFICADO DE CONFIANZA",
      "SU SOLUCIÓN EN PAPELERIA DE MAYOREO",
      "04/09/2024 09:56 AM",
      "NOTAS DE VENTA",
      "A1251",
      "CAJERO: ML",
      "TICKET NO: 3399",
      "CANT. DESCRIPCIÓN PRECIO IMPORTE",
      "2	HOJA CTA CAN 09 DIEM100	$55.50	$111.00",
      "1	FOMY C DIA AM C/10	$30.00	$30.00",
      "3	RAFIA DECORATIVA AZ CL	$18.90	$56.70",
      "1	PLT 20 SURT C/12	$99.50	$99.50",
      "3	CAN C/50 —	$53.70 $161.107 .",
      "1	LUSTRE AM CAN C/25	$66.90 $66.90",
      "1	BOLA UNIC 9 100MM C/10	$78.91	$78.91",
      "1	BOLA UNIC 11 140MM C/3	$57.67	$57.67 +",
      "3	PINTDIGITAL250NGO	$24.90	$74.70",
      "NO. DE ARTICULOS: 16",
      "TOTAL: $736.48",
      "PAGO CON: $736.48",
      "SU CAMBIO: $0.00",
      "USTED AHORRO: $315.52"
   ]
   
   process_lines_with_clustering(lines)
