import math

# Function to calculate the square root of a number
def square_root(num):
    """This function calculates the square root of a number."""
    return math.sqrt(num)

# Function to calculate the absolute value of a number
def absolute_value(num):
    """This function calculates the absolute value of a number."""
    return abs(num)

# Function to calculate the factorial of a number
def factorial(num):
    """This function calculates the factorial of a number."""
    return math.factorial(num)

# Function to calculate the logarithm of a number with base 10
def logarithm(num):
    """This function calculates the logarithm of a number with base 10."""
    return math.log10(num)

# Function to calculate the power of a number
def power(base, exponent):
    """This function calculates the power of a number."""
    return math.pow(base, exponent)

# Function to calculate the sine of an angle in radians
def sine(angle):
    """This function calculates the sine of an angle in radians."""
    return math.sin(angle)

# Function for Fibonacci sequence
def fibonacci_sequence(n):
    """This function calculates the Fibonacci sequence up to the given number of terms."""
    sequence = [0, 1]  # Initialize the sequence with the first two numbers
    # Calculate the Fibonacci sequence up to the given number of terms
    for i in range(2, n):
        next_number = sequence[i-1] + sequence[i-2]
        sequence.append(next_number)
    return sequence

# Function to calculate the sum of the first 100 natural numbers
def sum_first_100_natural_numbers(): 
    """This function calculates the sum of the first 100 natural numbers."""   
    return sum(range(1, 101))