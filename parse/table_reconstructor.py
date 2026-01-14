"""
Intelligent Table Reconstructor
Reads noisy JSON tables, fixes common errors, and exports to HTML.
"""

import json
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import html


class TableReconstructor:
    """Intelligently reconstructs and cleans noisy table data."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.tables = []
        
    def load_json(self) -> List[Dict[str, Any]]:
        """Load JSON file with error handling."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both single table and list of tables
                if isinstance(data, dict):
                    return [data]
                elif isinstance(data, list):
                    return data
                else:
                    print(f"Warning: Unexpected data type: {type(data)}")
                    return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            # Try to fix common JSON errors
            return self._fix_and_load_json()
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []
    
    def _fix_and_load_json(self) -> List[Dict[str, Any]]:
        """Attempt to fix common JSON errors and reload."""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Fix common JSON issues
            content = re.sub(r',\s*}', '}', content)  # Remove trailing commas in objects
            content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
            content = re.sub(r'}\s*{', '},{', content)  # Add missing commas between objects
            
            data = json.loads(content)
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"Could not fix JSON: {e}")
            return []
    
    def clean_cell_value(self, value: Any) -> str:
        """Clean and normalize cell values."""
        if value is None:
            return ""
        
        # Convert to string
        value = str(value).strip()
        
        # Fix common symbol misplacements
        # Move currency symbols to the beginning
        value = re.sub(r'(\d+\.?\d*)\s*\$', r'$\1', value)
        value = re.sub(r'(\d+\.?\d*)\s*€', r'€\1', value)
        value = re.sub(r'(\d+\.?\d*)\s*£', r'£\1', value)
        
        # Fix percentage symbols
        value = re.sub(r'%\s*(\d+\.?\d*)', r'\1%', value)
        value = re.sub(r'(\d+\.?\d*)\s+%', r'\1%', value)
        
        # Remove multiple spaces
        value = re.sub(r'\s+', ' ', value)
        
        # Fix common character issues
        value = value.replace('â‚¬', '€')
        value = value.replace('â€"', '-')
        value = value.replace('â€™', "'")
        
        return value.strip()
    
    def detect_table_structure(self, table_data: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
        """
        Intelligently detect and extract table structure.
        Returns (headers, rows)
        """
        headers = []
        rows = []
        
        # Strategy 1: Check for explicit headers and rows/data keys
        if 'headers' in table_data and 'rows' in table_data:
            headers = [self.clean_cell_value(h) for h in table_data['headers']]
            rows = [[self.clean_cell_value(cell) for cell in row] 
                   for row in table_data['rows']]
        
        elif 'columns' in table_data and 'data' in table_data:
            headers = [self.clean_cell_value(h) for h in table_data['columns']]
            rows = [[self.clean_cell_value(cell) for cell in row] 
                   for row in table_data['data']]
        
        # Strategy 2: Check if it's a list of dictionaries (each dict is a row)
        elif 'data' in table_data and isinstance(table_data['data'], list):
            if table_data['data'] and isinstance(table_data['data'][0], dict):
                headers = list(table_data['data'][0].keys())
                rows = [[self.clean_cell_value(row.get(h, '')) for h in headers] 
                       for row in table_data['data']]
        
        # Strategy 3: Assume the whole dict is the table
        elif isinstance(table_data, dict) and not any(k in table_data for k in ['headers', 'rows', 'data', 'columns']):
            # Check if values are lists (columns)
            if all(isinstance(v, list) for v in table_data.values()):
                headers = list(table_data.keys())
                max_len = max(len(v) for v in table_data.values())
                rows = []
                for i in range(max_len):
                    row = [self.clean_cell_value(table_data[h][i] if i < len(table_data[h]) else '') 
                          for h in headers]
                    rows.append(row)
        
        # Strategy 4: Check if it's just a list (assume first row is headers)
        elif isinstance(table_data, list) and table_data:
            if isinstance(table_data[0], list):
                headers = [self.clean_cell_value(h) for h in table_data[0]]
                rows = [[self.clean_cell_value(cell) for cell in row] 
                       for row in table_data[1:]]
            elif isinstance(table_data[0], dict):
                headers = list(table_data[0].keys())
                rows = [[self.clean_cell_value(row.get(h, '')) for h in headers] 
                       for row in table_data]
        
        return headers, rows
    
    def fix_column_row_mismatch(self, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Fix mismatches between column count and row lengths."""
        if not headers or not rows:
            return headers, rows
        
        # Determine the most common row length
        row_lengths = [len(row) for row in rows]
        if not row_lengths:
            return headers, rows
        
        most_common_length = Counter(row_lengths).most_common(1)[0][0]
        target_length = max(len(headers), most_common_length)
        
        # Adjust headers
        if len(headers) < target_length:
            headers.extend([f'Column {i+1}' for i in range(len(headers), target_length)])
        elif len(headers) > target_length:
            headers = headers[:target_length]
        
        # Adjust rows
        fixed_rows = []
        for row in rows:
            if len(row) < target_length:
                # Pad with empty strings
                row.extend([''] * (target_length - len(row)))
            elif len(row) > target_length:
                # Try to merge cells if there are split values
                row = self._merge_split_cells(row, target_length)
            fixed_rows.append(row)
        
        return headers, fixed_rows
    
    def _merge_split_cells(self, row: List[str], target_length: int) -> List[str]:
        """Intelligently merge cells that might have been split incorrectly."""
        if len(row) <= target_length:
            return row
        
        # Strategy: Look for cells that might belong together
        merged_row = []
        i = 0
        while i < len(row) and len(merged_row) < target_length:
            if len(merged_row) == target_length - 1:
                # Merge all remaining cells into the last position
                merged_row.append(' '.join(row[i:]))
                break
            else:
                cell = row[i]
                # Check if next cell should be merged (e.g., split currency values)
                if i + 1 < len(row):
                    next_cell = row[i + 1]
                    # Merge if next cell looks like it continues current cell
                    if (next_cell and len(next_cell) < 3) or \
                       (cell and cell[-1] in ',$' and next_cell.replace('.', '').isdigit()):
                        cell = cell + ' ' + next_cell
                        i += 1
                merged_row.append(cell)
                i += 1
        
        # If still too long, truncate
        if len(merged_row) > target_length:
            merged_row = merged_row[:target_length]
        
        # If too short, pad
        while len(merged_row) < target_length:
            merged_row.append('')
        
        return merged_row
    
    def process_tables(self):
        """Process all tables from JSON."""
        raw_tables = self.load_json()
        
        for idx, table_data in enumerate(raw_tables):
            try:
                # Extract table structure
                headers, rows = self.detect_table_structure(table_data)
                
                # Fix column/row mismatches
                headers, rows = self.fix_column_row_mismatch(headers, rows)
                
                # Store processed table
                table_title = table_data.get('title', table_data.get('name', f'Table {idx + 1}'))
                self.tables.append({
                    'title': self.clean_cell_value(table_title),
                    'headers': headers,
                    'rows': rows
                })
                
                print(f"Processed table {idx + 1}: {len(headers)} columns, {len(rows)} rows")
            except Exception as e:
                print(f"Error processing table {idx + 1}: {e}")
                continue
    
    def export_to_html(self, output_file: str = 'tables_output.html'):
        """Export all processed tables to HTML."""
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reconstructed Tables</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .table-container {
            background-color: white;
            margin: 30px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .table-title {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .stats {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>Reconstructed Tables</h1>
"""
        
        for table in self.tables:
            html_content += f"""
    <div class="table-container">
        <h2 class="table-title">{html.escape(str(table['title']))}</h2>
        <table>
            <thead>
                <tr>
"""
            # Add headers
            for header in table['headers']:
                html_content += f"                    <th>{html.escape(str(header))}</th>\n"
            
            html_content += """                </tr>
            </thead>
            <tbody>
"""
            # Add rows
            for row in table['rows']:
                html_content += "                <tr>\n"
                for cell in row:
                    html_content += f"                    <td>{html.escape(str(cell))}</td>\n"
                html_content += "                </tr>\n"
            
            html_content += """            </tbody>
        </table>
        <div class="stats">
            {0} columns × {1} rows
        </div>
    </div>
""".format(len(table['headers']), len(table['rows']))
        
        html_content += """
</body>
</html>
"""
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nHTML output written to: {output_file}")
        return output_file


def main():
    """Main execution function."""
    import sys
    import os
    import glob
    
    # Get JSON file from command line or search in C:/share/
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Look for JSON files in C:/share/
        search_dir = 'C:/share/'
        json_files = glob.glob(os.path.join(search_dir, '*.json'))
        
        if json_files:
            json_file = json_files[0]
            if len(json_files) > 1:
                print(f"Found {len(json_files)} JSON files in {search_dir}")
                print(f"Using: {json_file}")
                print("Other files found:", ', '.join(os.path.basename(f) for f in json_files[1:]))
                print("-" * 50)
        else:
            print(f"No JSON files found in {search_dir}")
            print("Usage: python table_reconstructor.py <json_file> [output_file]")
            return
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'tables_output.html'
    
    print(f"Processing tables from: {json_file}")
    print("-" * 50)
    
    # Create reconstructor and process
    reconstructor = TableReconstructor(json_file)
    reconstructor.process_tables()
    
    # Export to HTML
    if reconstructor.tables:
        reconstructor.export_to_html(output_file)
        print(f"\nSuccessfully processed {len(reconstructor.tables)} table(s)")
    else:
        print("\nNo tables found or processed")


if __name__ == "__main__":
    main()
