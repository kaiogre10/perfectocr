import re
from datetime import datetime

class TextFormatter:
    @staticmethod
    def normalize_dates(text: str) -> str:
        """Estandariza formatos de fecha"""
        date_patterns = [
            (r'\b(\d{2})[-/](\d{2})[-/](\d{4})\b', r'\3-\2-\1'),  # DD-MM-YYYY → YYYY-MM-DD
            (r'\b(\d{4})[-/](\d{2})[-/](\d{2})\b', r'\1-\2-\3'),  # YYYY-MM-DD
            (r'\b(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\b', lambda m: f"{m.group(3)}-{TextFormatter._month_to_num(m.group(2))}-{m.group(1).zfill(2)}")
        ]
        
        for pattern, replacement in date_patterns:
            text = re.sub(pattern, replacement, text)
        return text

    @staticmethod
    def _month_to_num(month_name: str) -> str:
        months = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }
        return months.get(month_name.lower(), '00')

    @staticmethod
    def normalize_numbers(text: str) -> str:
        """Estandariza formatos numéricos"""
        # Unifica separadores decimales
        text = re.sub(r'(\d+)[,.](\d{2})\b', r'\1.\2', text)
        # Formato de miles (opcional)
        text = re.sub(r'\b(\d{1,3}(?:\.\d{3})+)\b', lambda m: m.group(0).replace('.', ''), text)
        return text

    @staticmethod
    def preserve_line_breaks(text: str, max_line_length: int = 80) -> str:
        """Mantiene estructura de párrafos original"""
        lines = text.split('\n')
        formatted = []
        
        for line in lines:
            if len(line) <= max_line_length:
                formatted.append(line)
                continue
                
            words = line.split()
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_line_length:
                    formatted.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                formatted.append(' '.join(current_line))
        
        return '\n'.join(formatted)