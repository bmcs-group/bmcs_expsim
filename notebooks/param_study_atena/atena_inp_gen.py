#!/usr/bin/env python3 
""" Inject time-dependent (time,value) data into an ATENA GiD-generated input file.

Reads an existing ATENA GiD-generated input file line by line.
Searches for FUNCTION ID blocks (e.g., lines like "FUNCTION ID = 3" or "FUNCTION ID= 5", depending on the actual format).
Replaces the content in that FUNCTION block with user-defined (time, value) data from a Python dictionary called time_functions.
Writes an updated file to an output path.
Usage:

Adjust the time_functions dictionary in this script with the correct function IDs and lists of (time, value) pairs.
Adjust any string or regex patterns to match the exact structure of your ATENA input file.
Run this script, providing the input file path and the output file path.
Example: python inject_functions.py input_file.dat updated_file.dat 
"""

import sys 
import re

def parse_function_id(line): 
    """ Attempt to parse out the function ID from a line that contains 
    something like: FUNCTION ID 5 or FUNCTION ID 3 Returns the integer 
    ID if found, otherwise None. """ 
    # Simple regex that captures an integer after "FUNCTION ID"
    match = re.search(r'FUNCTION\s+ID\s+(\d+)', line, re.IGNORECASE) 
    if match: 
        return int(match.group(1)) 
    return None

def format_function_block(func_id, time_value_pairs): 
    """ Given a function ID and a list of (time, value) points, construct 
    the lines that typically define such a function in the ATENA input file.
    
    This format may need to be adapted depending on how your input file is structured.
    Below is just a simple example where we list each pair as:

    XVALUES ...
    YVALUES ...

    or possibly in a single table. Adjust as needed.
    """
    lines = []
    lines.append(f"FUNCTION ID = {func_id}")

    # Example approach: we store as separate XVALUES, YVALUES
    # Suppose the file format demands a line with the number of points
    lines.append(f" NUM_POINTS = {len(time_value_pairs)}")

    # X-values in one line
    x_vals = " ".join(str(point[0]) for point in time_value_pairs)
    lines.append(f" XVALUES = {x_vals}")

    # Y-values in another line
    y_vals = " ".join(str(point[1]) for point in time_value_pairs)
    lines.append(f" YVALUES = {y_vals}")

    lines.append("END FUNCTION")
    return lines

def inject_time_functions(input_file, output_file, time_functions): 
    """ Read the input file line by line, identify FUNCTION ID blocks, 
    and replace them with new ones from 'time_functions' if available. 
    Otherwise, pass them through unmodified.
    :param input_file: str, path to the original GiD-generated input file
    :param output_file: str, path to the updated file to write
    :param time_functions: dict, { function_id: [(time, value), (time, value), ...], ... }
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        inside_function_block = False
        current_function_id = None
        
        for line in f_in:
            # Check if this is the start of a FUNCTION block
            matched_id = parse_function_id(line)
            
            if matched_id is not None:
                # Found a FUNCTION block’s ID
                current_function_id = matched_id

                # Write the FUNCTION ID line
                f_out.write(line)              
                
                # Read the next three lines (name, type, and xvalues/yvalues)
                name_line = f_in.readline()
                type_line = f_in.readline()
                xvalues_line = f_in.readline()
                yvalues_line = f_in.readline()
                # Write the name and type lines as-is
                f_out.write(name_line)
                f_out.write(type_line)

                if current_function_id in time_functions:
                    print(f"Injecting new data for FUNCTION ID {current_function_id}")
                    # Write the new xvalues and yvalues lines
                    new_xvalues = "        xvalues " + " ".join(str(point[0]) for point in time_functions[current_function_id])
                    new_yvalues = "        xvalues " + " ".join(str(point[1]) for point in time_functions[current_function_id])
                    f_out.write(new_xvalues + "\n")
                    f_out.write(new_yvalues + "\n")
                    
                else:
                    # If we do NOT have a new data set for this function,
                    # we simply write the original xvalues and yvalues lines
                    f_out.write(xvalues_line)
                    f_out.write(yvalues_line)
                
            else:
                # Not a FUNCTION ID line, just write it normally
                f_out.write(line)

