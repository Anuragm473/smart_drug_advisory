import csv
import pandas as pd

# Path to the CSV file
input_file = 'datasets/workout_df.csv'
output_file = 'datasets/workout_df_fixed.csv'

try:
    # Read the CSV file with pandas
    df = pd.read_csv(input_file, engine='python', error_bad_lines=False)
    
    # Write the DataFrame back to CSV with proper quoting
    df.to_csv(output_file, quoting=csv.QUOTE_NONNUMERIC, index=False)
    
    print(f"Fixed CSV file saved to {output_file}")
    print(f"Original row count: {len(pd.read_csv(input_file, error_bad_lines=False))}")
    print(f"Fixed row count: {len(df)}")
    
except Exception as e:
    print(f"Error: {e}")
    
    # Alternative approach using csv module
    try:
        print("Trying alternative approach...")
        
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            # Read the file as text
            content = infile.read()
            
        # Split by newlines
        lines = content.strip().split('\n')
        
        # Parse header
        header = lines[0].split(',')
        
        # Create a new list for fixed lines
        fixed_lines = [lines[0]]  # Start with header
        
        # Process each data line
        for i in range(1, len(lines)):
            line = lines[i]
            parts = line.split(',')
            
            # If we have more than 4 parts, we need to fix this line
            if len(parts) > 4:
                # Assuming the format is: index,unnamed,disease,workout
                # Where workout might contain commas
                index = parts[0]
                unnamed = parts[1]
                disease = parts[2]
                workout = ','.join(parts[3:])
                
                # Create fixed line
                fixed_line = f'{index},{unnamed},{disease},"{workout}"'
                fixed_lines.append(fixed_line)
            else:
                # Line is already correct
                fixed_lines.append(line)
        
        # Write fixed content to output file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            outfile.write('\n'.join(fixed_lines))
        
        print(f"Fixed CSV file saved to {output_file} using alternative approach")
        print(f"Original line count: {len(lines)}")
        print(f"Fixed line count: {len(fixed_lines)}")
        
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}") 