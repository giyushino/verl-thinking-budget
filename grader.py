import re

def extract_solution(solution_str):
    """
    Extract the answer from \\boxed{} format in the model's response.

    Args:
        output: The model's response text

    Returns:
        The extracted answer or None if not found
    """
    # Pattern to match \boxed{...} with balanced braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution_str)

    if matches:
        # Return the last boxed answer (in case there are multiple)
        return matches[-1].strip()
    return None


def normalize_latex_string(s):
    """
    Final robust normalization function:
    1. Removes LaTeX delimiters and metadata (\left, \right, \boxed, etc.).
    2. Removes all internal whitespace.
    3. Handles numerical equivalence (4.00 -> 4).
    """
    
    # 1. Remove LaTeX delimiters, metadata, and trim
    s = re.sub(r'\\left|\\right|\[.*?\]|\\boxed', '', s)
    s = s.strip()
    
    # 2. Remove all internal whitespace aggressively
    s = re.sub(r'\s+', '', s) 
    
    # 3. Handle Numerical Equivalence (e.g., "4.00" -> "4", "3.10" -> "3.1")
    # This step is tricky because we don't want to break the LaTeX structure.
    # We will look for numbers that might need normalization.
    
    def normalize_number_match(match):
        """Helper function to clean a found number."""
        try:
            num = float(match.group(0))
            # Convert float back to string, which automatically handles '4.0' or '4.00' to '4.0', 
            # and then use .rstrip('0').rstrip('.') to get the desired minimal integer form '4'
            return str(num).rstrip('0').rstrip('.')
        except ValueError:
            return match.group(0) # Return the original string if conversion fails

    # Find sequences of digits, decimal points, and optional sign.
    # We apply this to the string *before* any other complex normalization
    # if the number stands alone. Since we've already removed all spaces (Step 2),
    # we need a targeted approach. Let's focus on the entire string first
    # if it's a simple number.
    
    # If the string is *just* a number (e.g., '4.00'):
    try:
        if s and s.replace('.', '', 1).isdigit(): # Check if it's purely a number
            return str(float(s)).rstrip('0').rstrip('.')
    except ValueError:
        pass # Not a simple number, continue with original string 's'
    
    # If the string contains numbers *within* a larger structure (e.g., (4.00, \pi/2))
    # This regex looks for floating point numbers: (\d+\.?\d*)
    s = re.sub(r'\d+\.\d+', normalize_number_match, s)

    return s.strip()



# the print functions are somewhat useful for debugging
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Extract \\boxed content from completion and compare to ground truth, return list of rewards
    """
    print(f"{solution_str=}")
    solution = extract_solution(solution_str)
    # If no solution was extracted, return 0 (incorrect)
    if solution is None:
        return 0
    if normalize_latex_string(str(ground_truth)) == normalize_latex_string(solution):
        return 1
    return 0



