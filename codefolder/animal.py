class Animal():
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def fibonacci_sequence(n):
        sequence = [0, 1]