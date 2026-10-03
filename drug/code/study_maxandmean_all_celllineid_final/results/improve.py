import pandas as pd
import glob

# Paths
#file_pattern = 'global_greedy_expansion/*_css_ri_*.csv'
file_pattern = 'global_greedy_expansion/*.csv'
drug_mapping_file = '../../../data/drug.csv'
output_file = 'top_triplet_improvements.csv'

# 1. Create the Drug ID -> Drug Name mapping
print("Loading drug mapping...")
# Mapping 'cid' (6th col) to 'dname' (1st col)
drug_df = pd.read_csv(drug_mapping_file, usecols=['cid', 'dname'], dtype={'cid': str})
id_to_name = dict(zip(drug_df['cid'], drug_df['dname']))

# 2. Setup processing - Note the updated column name: 'added_drug'
dtype_settings = {
    'cell_line': str,
    'drug1': str,
    'drug2': str,
    'added_drug': str
}

cell_line_stats = {}
all_files = glob.glob(file_pattern)

# 3. Process shards
if not all_files:
    print("No shards found. Check your file_pattern.")
else:
    for filename in all_files:
        print(f"Processing: {filename}")
        df = pd.read_csv(filename, dtype=dtype_settings, low_memory=False)
        
        for name, group in df.groupby('cell_line'):
            max_pair = group['pair_score'].max()
            best_triplet_idx = group['triplet_score'].idxmax()
            best_triplet_row = group.loc[best_triplet_idx]
            
            if name not in cell_line_stats:
                cell_line_stats[name] = {'max_pair_score': max_pair, 'best_row': best_triplet_row}
            else:
                # Update max pair score for the cell line globally
                if max_pair > cell_line_stats[name]['max_pair_score']:
                    cell_line_stats[name]['max_pair_score'] = max_pair
                # Update best triplet row for the cell line globally
                if best_triplet_row['triplet_score'] > cell_line_stats[name]['best_row']['triplet_score']:
                    cell_line_stats[name]['best_row'] = best_triplet_row

    # 4. Filter, Calculate Diff, and Map Names
    final_rows = []
    for name, stats in cell_line_stats.items():
        if stats['best_row']['triplet_score'] > stats['max_pair_score']:
            # Create a copy to avoid modifying original data accidentally
            row = stats['best_row'].copy()
            
            # Calculate the improvement (diff)
            row['diff'] = row['triplet_score'] - stats['max_pair_score']
            
            # Replace IDs with names (using .get to handle missing IDs safely)
            row['drug1'] = id_to_name.get(row['drug1'], row['drug1'])
            row['drug2'] = id_to_name.get(row['drug2'], row['drug2'])
            row['added_drug'] = id_to_name.get(row['added_drug'], row['added_drug'])
            
            final_rows.append(row)

    # 5. Save Results
    if final_rows:
        output_df = pd.DataFrame(final_rows)
        # Reorder columns for readability if you like
        cols = ['cell_line', 'drug1', 'drug2', 'added_drug', 'pair_score', 'triplet_score', 'diff']
        # Use existing columns if your shard has more
        final_cols = [c for c in cols if c in output_df.columns]
        
        output_df[final_cols].to_csv(output_file, index=False)
        print(f"\nSuccess! Filtered {len(final_rows)} improvements. Saved to {output_file}")
    else:
        print("\nNo triplets found that outperformed pairs.")
