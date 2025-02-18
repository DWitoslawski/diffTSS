import pandas as pd
import argparse

def replace_ensid(data_file, mapping_file, output_file):
    # Load the data file
    df = pd.read_csv(data_file, sep='\t')  # Adjust separator if necessary
    
    # Load the mapping file
    mapping_df = pd.read_csv(mapping_file, sep='\t')  # Adjust separator if necessary
    
    # Create a dictionary for ENSID to newENSID mapping
    ensid_dict = dict(zip(mapping_df['ENSID'], mapping_df['newENSID']))
  
    # Create a backup of the original ENSID column
    df['ENSID_old'] = df['ENSID']
    
    # Replace ENSID with newENSID
    df['ENSID'] = df['ENSID'].map(ensid_dict).fillna('NA')  # Preserve original if no mapping exists
    
    # Save the updated DataFrame
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Updated file saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace ENSID in a data file with newENSID using a mapping file.")
    parser.add_argument("data_file", help="Path to the input data file with ENSID column")
    parser.add_argument("mapping_file", help="Path to the mapping file with ENSID and newENSID")
    parser.add_argument("output_file", help="Path to save the output file with updated ENSID values")
    
    args = parser.parse_args()
    
    replace_ensid(args.data_file, args.mapping_file, args.output_file)

