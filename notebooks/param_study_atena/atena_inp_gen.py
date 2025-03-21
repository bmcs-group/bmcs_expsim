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
                    x_values, y_values = time_functions[current_function_id]
                    # ATENA creep does not accept zero values for time - setting time is the start time - at least 0.01
                    new_xvalues = "        xvalues " + " ".join(str(val) for val in x_values[1:])
                    new_yvalues = "        yvalues " + " ".join(str(val) for val in y_values[1:])
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


def generate_input_section(stop_time, num_steps, load_case_id):
    step_template = """
STEP ID {step_id} Type CREEP NAME "Load ... BC#1" AT    {time_value:.5f}
Load CASE
          FIXED
               1 *     0.00100

          INCREMENT
      {load_case_id} * 1.0000000e-03
 ;
"""
    interval_template = """
MAX_COORDS_TOL 2.0000000e-01
                                                                                 
SET STOP_TIME    {stop_time:.5f} // Redefinition of Stop Time by start of Next step definition

/* Total of steps in Interval# 1: {num_steps} */
"""
    
    interval_section = interval_template.format(stop_time=stop_time, num_steps=num_steps)
    
    steps_section = ""
    time_increment = stop_time / num_steps
    
    for step_id in range(1, num_steps + 1):
        time_value = step_id * time_increment
        steps_section += step_template.format(step_id=step_id, time_value=time_value, load_case_id=load_case_id)
    
    return interval_section + steps_section

