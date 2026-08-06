class Fruit:
    def __init__(self, name, price):
        self.name = name
        self.price = price


class Apple(Fruit):
    def __init__(self, name, price):
        super().__init__(name, price)
        self.name = "apple"
        self.price = price


class Banana(Fruit):
    def __init__(self, name, price):
        super().__init__(name, price)
        self.name = "banana"
        self.price = price


class Oranges(Fruit):
    def __init__(self, name, price):
        super().__init__(name, price)
        self.name = "oranges"
        self.price = price


