"""
Intelligent Table Reconstructor
Reads noisy JSON tables, fixes common errors, and exports to HTML.
Handles LLM-generated JSON from PDF processing with mixed text and tables.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import html


class TableReconstructor:
    """Intelligently reconstructs and cleans noisy table data."""
    
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.tables = []
        self.min_table_rows = 2  # Minimum rows to consider something a table
        self.min_table_cols = 2  # Minimum columns to consider something a table
    
    def is_text_content(self, value: Any) -> bool:
        """Check if value is likely regular text rather than tabular data."""
        if not isinstance(value, str):
            return False
        
        # Long sentences are likely text, not table cells
        if len(value) > 200:
            return True
        
        # Contains multiple sentences
        if value.count('. ') > 2:
            return True
        
        # Contains paragraph indicators
        if '\n\n' in value or value.count('\n') > 3:
            return True
        
        return False
    
    def validate_table_quality(self, headers: List[str], rows: List[List[str]]) -> bool:
        """Validate if extracted table has sufficient quality to be exported."""
        if not headers or not rows:
            return False
        
        # Check minimum dimensions
        if len(headers) < self.min_table_cols or len(rows) < self.min_table_rows:
            return False
        
        # Check if all rows are empty
        non_empty_cells = sum(1 for row in rows for cell in row if cell.strip())
        total_cells = len(rows) * len(headers)
        
        if total_cells > 0 and (non_empty_cells / total_cells) < 0.3:
            return False
        
        # Check for text content in headers (might be misidentified)
        text_headers = sum(1 for h in headers if self.is_text_content(h))
        if text_headers > len(headers) * 0.5:
            return False
        
        return True
        
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
    
    def is_likely_table(self, data: Any) -> bool:
        """Determine if data structure is likely a table."""
        # Skip if it's just a string (text content)
        if isinstance(data, str):
            return False
        
        # List of dicts with consistent keys (common LLM output)
        if isinstance(data, list) and len(data) >= self.min_table_rows:
            if all(isinstance(item, dict) for item in data):
                # Check for consistent keys
                if data:
                    first_keys = set(data[0].keys())
                    if len(first_keys) >= self.min_table_cols:
                        # At least 70% of items should have similar keys
                        similar_count = sum(1 for item in data if len(set(item.keys()) & first_keys) >= len(first_keys) * 0.7)
                        if similar_count / len(data) >= 0.7:
                            return True
            
            # List of lists (rows)
            elif all(isinstance(item, list) for item in data):
                if len(data[0]) >= self.min_table_cols:
                    # Check if most rows have similar length
                    lengths = [len(row) for row in data]
                    most_common = Counter(lengths).most_common(1)[0][0]
                    similar = sum(1 for l in lengths if abs(l - most_common) <= 2)
                    if similar / len(data) >= 0.7:
                        return True
        
        # Dict with list values (columns)
        if isinstance(data, dict):
            list_values = [v for v in data.values() if isinstance(v, list)]
            if len(list_values) >= self.min_table_cols and len(list_values) == len(data):
                lengths = [len(v) for v in list_values]
                if lengths and max(lengths) >= self.min_table_rows:
                    # Check if lengths are similar
                    most_common = Counter(lengths).most_common(1)[0][0]
                    similar = sum(1 for l in lengths if abs(l - most_common) <= 1)
                    if similar / len(lengths) >= 0.7:
                        return True
        
        return False
    
    def extract_all_tables(self, data: Any, path: str = "root") -> List[Dict[str, Any]]:
        """Recursively find all table-like structures in the JSON."""
        found_tables = []
        
        # Check if current data is a table
        if self.is_likely_table(data):
            found_tables.append({
                'data': data,
                'path': path
            })
            return found_tables
        
        # Recursively search in dicts
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = f"{path}.{key}" if path != "root" else key
                
                # Check if this value is a table
                if self.is_likely_table(value):
                    found_tables.append({
                        'data': value,
                        'path': new_path,
                        'title': key
                    })
                # Recurse deeper
                elif isinstance(value, (dict, list)):
                    found_tables.extend(self.extract_all_tables(value, new_path))
        
        # Recursively search in lists
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                if isinstance(item, (dict, list)) and not self.is_likely_table(item):
                    found_tables.extend(self.extract_all_tables(item, new_path))
        
        return found_tables
    
    def detect_table_structure(self, table_data: Any) -> Tuple[List[str], List[List[str]]]:
        """
        Intelligently detect and extract table structure from various formats.
        Returns (headers, rows)
        """
        headers = []
        rows = []
        
        # Strategy 1: List of dictionaries (most common LLM format)
        if isinstance(table_data, list) and table_data and isinstance(table_data[0], dict):
            # Get all unique keys maintaining order
            all_keys = []
            for item in table_data:
                for key in item.keys():
                    if key not in all_keys:
                        all_keys.append(key)
            
            headers = [self.clean_cell_value(h) for h in all_keys]
            rows = []
            for item in table_data:
                row = [self.clean_cell_value(item.get(h, '')) for h in all_keys]
                rows.append(row)
        
        # Strategy 2: List of lists (rows)
        elif isinstance(table_data, list) and table_data and isinstance(table_data[0], list):
            # Try to detect if first row is headers (contains mostly strings, not numbers)
            first_row = table_data[0]
            is_header_row = True
            
            if len(table_data) > 1:
                # Check if first row looks different from second row
                try:
                    # If first row has strings and second has numbers, first is likely header
                    first_has_numbers = any(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','').replace('-','').replace(',','').isdigit()) 
                                          for x in first_row)
                    if len(table_data) > 1:
                        second_row = table_data[1]
                        second_has_numbers = any(isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.','').replace('-','').replace(',','').isdigit()) 
                                               for x in second_row)
                        # If first row has fewer numbers than second, it's likely a header
                        if second_has_numbers and not first_has_numbers:
                            is_header_row = True
                        elif first_has_numbers and second_has_numbers:
                            is_header_row = False
                except:
                    pass
            
            if is_header_row:
                headers = [self.clean_cell_value(h) for h in first_row]
                rows = [[self.clean_cell_value(cell) for cell in row] for row in table_data[1:]]
            else:
                # Generate headers
                headers = [f'Column {i+1}' for i in range(len(first_row))]
                rows = [[self.clean_cell_value(cell) for cell in row] for row in table_data]
        
        # Strategy 3: Dict with list values (columns)
        elif isinstance(table_data, dict):
            # Check if all values are lists
            list_values = {k: v for k, v in table_data.items() if isinstance(v, list)}
            
            if len(list_values) >= self.min_table_cols:
                headers = [self.clean_cell_value(k) for k in list_values.keys()]
                max_len = max(len(v) for v in list_values.values()) if list_values else 0
                
                rows = []
                for i in range(max_len):
                    row = []
                    for key in list_values.keys():
                        val = list_values[key][i] if i < len(list_values[key]) else ''
                        row.append(self.clean_cell_value(val))
                    rows.append(row)
            
            # Check for explicit table keys
            elif 'headers' in table_data or 'columns' in table_data:
                header_key = 'headers' if 'headers' in table_data else 'columns'
                data_key = 'rows' if 'rows' in table_data else 'data'
                
                if header_key in table_data and data_key in table_data:
                    headers = [self.clean_cell_value(h) for h in table_data[header_key]]
                    
                    if isinstance(table_data[data_key], list):
                        rows = []
                        for row in table_data[data_key]:
                            if isinstance(row, list):
                                rows.append([self.clean_cell_value(cell) for cell in row])
                            elif isinstance(row, dict):
                                rows.append([self.clean_cell_value(row.get(h, '')) for h in headers])
        
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
        """Process all tables from JSON by intelligently finding table structures."""
        raw_data = self.load_json()
        
        if not raw_data:
            print("No data loaded from JSON file")
            return
        
        # If it's a list, search each item
        if isinstance(raw_data, list):
            all_found_tables = []
            for item in raw_data:
                all_found_tables.extend(self.extract_all_tables(item))
        else:
            all_found_tables = self.extract_all_tables(raw_data)
        
        print(f"\nFound {len(all_found_tables)} potential table(s) in the JSON")
        print("-" * 50)
        
        for idx, table_info in enumerate(all_found_tables):
            try:
                table_data = table_info['data']
                
                # Extract table structure
                headers, rows = self.detect_table_structure(table_data)
                
                if not headers or not rows:
                    print(f"Skipping table {idx + 1}: Could not extract structure")
                    continue
                
                # Fix column/row mismatches
                headers, rows = self.fix_column_row_mismatch(headers, rows)
                
                # Validate table quality
                if not self.validate_table_quality(headers, rows):
                    print(f"Skipping table {idx + 1}: Failed quality validation (might be text or insufficient data)")
                    continue
                
                # Generate title from path or provided title
                if 'title' in table_info:
                    table_title = table_info['title']
                else:
                    path_parts = table_info['path'].split('.')
                    table_title = path_parts[-1] if path_parts else f'Table {idx + 1}'
                
                # Clean up title
                table_title = self.clean_cell_value(table_title)
                if not table_title or table_title.startswith('['):
                    table_title = f'Table {len(self.tables) + 1}'
                
                self.tables.append({
                    'title': table_title,
                    'headers': headers,
                    'rows': rows,
                    'path': table_info['path']
                })
                
                print(f"✓ Table {len(self.tables)}: '{table_title}' - {len(headers)} columns × {len(rows)} rows")
                
            except Exception as e:
                print(f"✗ Error processing table at {table_info.get('path', idx)}: {e}")
                continue
        
        print("-" * 50)
        print(f"Successfully processed {len(self.tables)} table(s)")
    
    def preview_tables(self, max_rows: int = 3):
        """Display a preview of detected tables."""
        if not self.tables:
            print("No tables to preview")
            return
        
        print("\n" + "=" * 70)
        print("TABLE PREVIEW")
        print("=" * 70)
        
        for idx, table in enumerate(self.tables, 1):
            print(f"\n[Table {idx}] {table['title']}")
            print(f"Dimensions: {len(table['headers'])} columns × {len(table['rows'])} rows")
            print(f"Path: {table.get('path', 'N/A')}")
            print("\nHeaders:", ' | '.join(table['headers'][:5]))  # Show first 5 headers
            if len(table['headers']) > 5:
                print(f"... and {len(table['headers']) - 5} more columns")
            
            print("\nSample rows:")
            for i, row in enumerate(table['rows'][:max_rows]):
                row_preview = ' | '.join(str(cell)[:20] for cell in row[:5])
                print(f"  Row {i+1}: {row_preview}")
                if len(row) > 5:
                    print(f"         ... and {len(row) - 5} more cells")
            
            if len(table['rows']) > max_rows:
                print(f"  ... and {len(table['rows']) - max_rows} more rows")
            print("-" * 70)
    
    def export_to_html(self, output_file: str = 'tables_output.html'):
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
        """Process all tables from JSON by intelligently finding table structures."""
        raw_data = self.load_json()
        
        if not raw_data:
            print("No data loaded from JSON file")
            return
        
        # If it's a list, search each item
        if isinstance(raw_data, list):
            all_found_tables = []
            for item in raw_data:
                all_found_tables.extend(self.extract_all_tables(item))
        else:
            all_found_tables = self.extract_all_tables(raw_data)
        
        print(f"\nFound {len(all_found_tables)} potential table(s) in the JSON")
        print("-" * 50)
        
        for idx, table_info in enumerate(all_found_tables):
            try:
                table_data = table_info['data']
                
                # Extract table structure
                headers, rows = self.detect_table_structure(table_data)
                
                if not headers or not rows:
                    print(f"Skipping table {idx + 1}: Could not extract structure")
                    continue
                
                # Fix column/row mismatches
                headers, rows = self.fix_column_row_mismatch(headers, rows)
                
                # Validate table quality
                if not self.validate_table_quality(headers, rows):
                    print(f"Skipping table {idx + 1}: Failed quality validation (might be text or insufficient data)")
                    continue
                
                # Generate title from path or provided title
                if 'title' in table_info:
                    table_title = table_info['title']
                else:
                    path_parts = table_info['path'].split('.')
                    table_title = path_parts[-1] if path_parts else f'Table {idx + 1}'
                
                # Clean up title
                table_title = self.clean_cell_value(table_title)
                if not table_title or table_title.startswith('['):
                    table_title = f'Table {len(self.tables) + 1}'
                
                self.tables.append({
                    'title': table_title,
                    'headers': headers,
                    'rows': rows,
                    'path': table_info['path']
                })
                
                print(f"✓ Table {len(self.tables)}: '{table_title}' - {len(headers)} columns × {len(rows)} rows")
                
            except Exception as e:
                print(f"✗ Error processing table at {table_info.get('path', idx)}: {e}")
                continue
        
        print("-" * 50)
        print(f"Successfully processed {len(self.tables)} table(s)")
    
    def preview_tables(self, max_rows: int = 3):
        """Display a preview of detected tables."""
        if not self.tables:
            print("No tables to preview")
            return
        
        print("\n" + "=" * 70)
        print("TABLE PREVIEW")
        print("=" * 70)
        
        for idx, table in enumerate(self.tables, 1):
            print(f"\n[Table {idx}] {table['title']}")
            print(f"Dimensions: {len(table['headers'])} columns × {len(table['rows'])} rows")
            print(f"Path: {table.get('path', 'N/A')}")
            print("\nHeaders:", ' | '.join(table['headers'][:5]))  # Show first 5 headers
            if len(table['headers']) > 5:
                print(f"... and {len(table['headers']) - 5} more columns")
            
            print("\nSample rows:")
            for i, row in enumerate(table['rows'][:max_rows]):
                row_preview = ' | '.join(str(cell)[:20] for cell in row[:5])
                print(f"  Row {i+1}: {row_preview}")
                if len(row) > 5:
                    print(f"         ... and {len(row) - 5} more cells")
            
            if len(table['rows']) > max_rows:
                print(f"  ... and {len(table['rows']) - max_rows} more rows")
            print("-" * 70)
    
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
        # Show preview
        reconstructor.preview_tables()
        
        # Export
        print("\n" + "=" * 70)
        print("EXPORTING TO HTML")
        print("=" * 70)
        reconstructor.export_to_html(output_file)
        print(f"\n✓ Successfully exported {len(reconstructor.tables)} table(s) to {output_file}")
    else:
        print("\n✗ No tables found or processed")
        print("\nTips:")
        print("  - Ensure JSON contains structured data (lists or dicts)")
        print("  - Tables need at least 2 rows and 2 columns")
        print("  - Check if the JSON is properly formatted")


if __name__ == "__main__":
    main()
