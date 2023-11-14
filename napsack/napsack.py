import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def get_area(self):
        return math.pi \
            * self.radius \
                *self.radius


for i in range (89, 100):
    if (i & 1 ) == 0:
        continue
    circle = Circle(i)
    print("A circle with radius {0} has area {1:0.2f}".format(
        i, circle.get_area()
    ))
