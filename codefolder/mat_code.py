import math

# Function to calculate the square root of a number
def square_root(num):
    return math.sqrt(num)

# Function to calculate the absolute value of a number
def absolute_value(num):
    return abs(num)

# Function to calculate the factorial of a number
def factorial(num):
    return math.factorial(num)

# Function to calculate the logarithm of a number with base 10
def logarithm(num):
    return math.log10(num)

# Function to calculate the power of a number
def power(base, exponent):
    return math.pow(base, exponent)

# Function to calculate the sine of an angle in radians
def sine(angle):
    return math.sin(angle)

# Function for Fibonacci sequence
def fibonacci_sequence(n):
    sequence = [0, 1]  # Initialize the sequence with the first two numbers
    # Calculate the Fibonacci sequence up to the given number of terms
    for i in range(2, n):
        next_number = sequence[i-1] + sequence[i-2]
        sequence.append(next_number)
    return sequence

# Function to calculate the sum of the first 100 natural numbers
def sum_first_100_natural_numbers():    
    return sum(range(1, 101))