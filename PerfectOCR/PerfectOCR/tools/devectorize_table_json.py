# PerfectOCR/tools/devectorize_table_json.py
import json
import argparse
import os
import sys # sys sigue siendo útil para sys.exit
# No se necesita manipulación de sys.path si se ejecuta desde la raíz del proyecto.
# Las importaciones directas desde utils deberían funcionar.

from utils.geometry_transformers import devectorize_element_list
from utils.encoders import NumpyEncoder

# ... (el resto del script devectorize_table_json.py como se propuso anteriormente) ...

def devectorize_table_data_in_payload(payload_data_vectorized: dict) -> dict:
    # ... (lógica sin cambios) ...
    devectorized_payload = json.loads(json.dumps(payload_data_vectorized)) 
    table_matrix_struct = devectorized_payload.get("table_matrix")
    if isinstance(table_matrix_struct, dict) and "rows" in table_matrix_struct:
        devectorized_rows = []
        original_rows = table_matrix_struct.get("rows", [])
        if isinstance(original_rows, list):
            for row_vec in original_rows:
                new_row_cells = []
                if isinstance(row_vec, list):
                    for cell_vec in row_vec:
                        new_cell = cell_vec.copy() 
                        if 'words' in new_cell and isinstance(new_cell['words'], list):
                            new_cell['words'] = devectorize_element_list(new_cell['words'])
                        new_row_cells.append(new_cell)
                devectorized_rows.append(new_row_cells)
            devectorized_payload['table_matrix']['rows'] = devectorized_rows
    return devectorized_payload    

def main():
    # ... (lógica de argparse sin cambios) ...
    parser = argparse.ArgumentParser(
        description="Desvectoriza los elementos 'words' en un archivo JSON de extracción de tabla de PerfectOCR "
                    "(generado por TableAndFieldCoordinator cuando la devectorización interna está desactivada)."
    )
    parser.add_argument("input_json_path", help="Ruta al archivo JSON vectorizado (ej: _TABLE_EXTRACTION.json).")
    parser.add_argument("-o", "--output_json_path", 
                        help="Ruta para el archivo JSON desvectorizado (opcional). "
                             "Por defecto, añade '_devectorized' al nombre del archivo de entrada.")
    args = parser.parse_args()

    if not os.path.exists(args.input_json_path):
        print(f"Error: Archivo de entrada no encontrado: {args.input_json_path}")
        sys.exit(1)

    try:
        with open(args.input_json_path, 'r', encoding='utf-8') as f:
            vectorized_data_payload = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decodificando el JSON de entrada: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error leyendo el archivo JSON de entrada: {e}")
        sys.exit(1)

    print(f"Procesando y desvectorizando datos de: {args.input_json_path}...")
    devectorized_output_payload = devectorize_table_data_in_payload(vectorized_data_payload)

    output_path = args.output_json_path
    if not output_path:
        base, ext = os.path.splitext(args.input_json_path)
        output_path = f"{base}_devectorized{ext}"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(devectorized_output_payload, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"Archivo desvectorizado guardado en: {output_path}")
    except Exception as e:
        print(f"Error guardando el archivo JSON desvectorizado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()