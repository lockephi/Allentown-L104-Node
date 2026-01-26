#!/usr/bin/env python3
"""
Data File Merger - Consolidates training data and databases without data loss
Merges multiple JSONL training files and consolidates databases
"""

import gzip
import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime

def merge_jsonl_gz_files(file_list, output_file):
    """Merge multiple gzipped JSONL files into one"""
    total_lines = 0
    unique_entries = set()
    merged_data = []
    
    print(f"\nMerging {len(file_list)} JSONL files...")
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"  ⚠️  Skipping missing file: {file_path}")
            continue
            
        print(f"  Reading: {file_path} ({os.path.getsize(file_path) / 1024:.1f} KB)")
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Create a hash to detect duplicates
                        line_hash = hash(line)
                        if line_hash not in unique_entries:
                            unique_entries.add(line_hash)
                            merged_data.append(line)
                            total_lines += 1
        except Exception as e:
            print(f"  ⚠️  Error reading {file_path}: {e}")
    
    # Write merged file
    print(f"\n  Writing merged file: {output_file}")
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        for line in merged_data:
            f.write(line + '\n')
    
    output_size = os.path.getsize(output_file) / 1024
    print(f"  ✓ Merged {total_lines} unique entries ({output_size:.1f} KB)")
    
    return total_lines, output_size

def merge_training_files():
    """Merge all kernel training files"""
    base_path = Path('/workspaces/Allentown-L104-Node')
    
    # Group related training files
    training_groups = {
        'kernel_training_merged.jsonl.gz': [
            'kernel_physics_training.jsonl.gz',
            'kernel_reasoning_data.jsonl.gz',
            'kernel_training_supabase.jsonl.gz',
            'professor_mode_kernel_training.jsonl.gz',
            'sage_mode_kernel_training.jsonl.gz',
        ],
        'pantheon_training_merged.jsonl.gz': [
            'pantheon_precise_training.jsonl.gz',
            'pantheon_training_data.jsonl.gz',
        ]
    }
    
    total_saved = 0
    
    for output_name, input_files in training_groups.items():
        output_path = base_path / output_name
        input_paths = [base_path / f for f in input_files if (base_path / f).exists()]
        
        if not input_paths:
            print(f"\n⊘ No files found for {output_name}")
            continue
        
        # Calculate original size
        original_size = sum(os.path.getsize(p) for p in input_paths) / 1024
        
        # Merge files
        _, merged_size = merge_jsonl_gz_files(input_paths, output_path)
        
        saved = original_size - merged_size
        total_saved += saved
        
        print(f"  Space saved: {saved:.1f} KB")
        print(f"  Original files can be removed after verification")
    
    return total_saved

def merge_sqlite_databases():
    """Merge similar SQLite databases"""
    base_path = Path('/workspaces/Allentown-L104-Node')
    
    # Merge data directory databases
    data_dbs = [
        'data/akashic_records.db',
        'data/genesis_vault.db',
        'data/sage_memory.db'
    ]
    
    output_db = base_path / 'data' / 'merged_memory.db'
    
    print(f"\n\nMerging {len(data_dbs)} memory databases...")
    
    # Create merged database
    conn_out = sqlite3.connect(output_db)
    cursor_out = conn_out.cursor()
    
    total_tables = 0
    total_rows = 0
    
    for db_name in data_dbs:
        db_path = base_path / db_name
        if not db_path.exists():
            continue
            
        print(f"  Processing: {db_name}")
        
        try:
            conn_in = sqlite3.connect(db_path)
            cursor_in = conn_in.cursor()
            
            # Get all tables
            cursor_in.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor_in.fetchall()
            
            for (table_name,) in tables:
                # Create prefixed table name to avoid conflicts
                prefix = Path(db_name).stem
                new_table_name = f"{prefix}_{table_name}"
                
                # Get table schema
                cursor_in.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                schema = cursor_in.fetchone()
                
                if schema and schema[0]:
                    # Create table with new name
                    create_sql = schema[0].replace(f"CREATE TABLE {table_name}", 
                                                  f"CREATE TABLE IF NOT EXISTS {new_table_name}")
                    cursor_out.execute(create_sql)
                    
                    # Copy data
                    cursor_in.execute(f"SELECT * FROM {table_name}")
                    rows = cursor_in.fetchall()
                    
                    if rows:
                        placeholders = ','.join(['?'] * len(rows[0]))
                        cursor_out.executemany(f"INSERT OR IGNORE INTO {new_table_name} VALUES ({placeholders})", rows)
                        total_rows += len(rows)
                    
                    total_tables += 1
                    print(f"    ✓ {new_table_name}: {len(rows)} rows")
            
            conn_in.close()
            
        except Exception as e:
            print(f"    ⚠️  Error processing {db_name}: {e}")
    
    conn_out.commit()
    conn_out.close()
    
    output_size = os.path.getsize(output_db) / 1024
    print(f"\n  ✓ Created merged database: {output_size:.1f} KB")
    print(f"  Total: {total_tables} tables, {total_rows} rows")
    
    return output_size

def create_merge_summary():
    """Create a summary of what was merged"""
    summary_file = Path('/workspaces/Allentown-L104-Node/MERGE_SUMMARY.txt')
    
    with open(summary_file, 'w') as f:
        f.write(f"Data Merge Summary\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Merged Files:\n")
        f.write(f"-" * 60 + "\n\n")
        
        f.write(f"Training Data:\n")
        f.write(f"  - kernel_training_merged.jsonl.gz\n")
        f.write(f"    (merged from: kernel_physics_training, kernel_reasoning_data,\n")
        f.write(f"     kernel_training_supabase, professor_mode, sage_mode)\n\n")
        
        f.write(f"  - pantheon_training_merged.jsonl.gz\n")
        f.write(f"    (merged from: pantheon_precise_training, pantheon_training_data)\n\n")
        
        f.write(f"Databases:\n")
        f.write(f"  - data/merged_memory.db\n")
        f.write(f"    (merged from: akashic_records, genesis_vault, sage_memory)\n\n")
        
        f.write(f"⚠️  IMPORTANT:\n")
        f.write(f"Original files are preserved for verification.\n")
        f.write(f"After confirming merged files work correctly, you can remove:\n")
        f.write(f"  - Individual training files that were merged\n")
        f.write(f"  - Individual database files that were merged\n")
    
    print(f"\n✓ Merge summary created: {summary_file}")

def main():
    print("=" * 60)
    print("Data File Merger")
    print("=" * 60)
    
    try:
        # Merge training files
        saved_kb = merge_training_files()
        
        # Merge databases
        merge_sqlite_databases()
        
        # Create summary
        create_merge_summary()
        
        print("\n" + "=" * 60)
        print("Merge Complete!")
        print(f"Potential space savings: {saved_kb:.1f} KB (after removing originals)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
