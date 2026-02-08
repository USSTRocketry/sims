import csv

def extract_data_to_csv(input_file, output_file):

    data_rows = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or '------' in line:
            i += 1
            continue

        numbers = line.split()

        if len(numbers) >= 4:
            try:
                # Extract 1st, 3rd, and 4th numbers (index 0, 2, 3)
                first_num = float(numbers[0])
                third_num = float(numbers[2])
                fourth_num = float(numbers[3])
                
                data_rows.append([first_num, third_num, fourth_num])
                
                # Skip the next 5 lines of the block (or until blank line)
                i += 6
            except (ValueError, IndexError):
                
                i += 1
        else:
            i += 1
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mach', 'CD PWR OFF', 'CD PWR ON'])
        writer.writerows(data_rows)
    
    print(f"Extracted {len(data_rows)} rows of data")
    print(f"Data saved to {output_file}")
    print(f"\nFirst few rows:")
    for i, row in enumerate(data_rows[:5]):
        print(f"  {row}")

if __name__ == "__main__":
    input_file = #filepath
    output_file = #filepath
    
    extract_data_to_csv(input_file, output_file)
