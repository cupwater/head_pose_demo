class VirtualDesk:
    def __init__(self, init_height=120, height_range=(100, 160)):
        self.height_range = height_range
        self.height = init_height

    def get_height(self):
        return self.height

    def up(self, distance):
        if self.height+distance >= self.height_range[1]:
            print("exceed max height, will keep max height")
            self.height = self.height_range[1]
        else:
            self.height += distance

    def down(self, distance):
        if self.height-distance <= self.height_range[0]:
            print("exceed min height, will keep min height")
            self.height = self.height_range[0]
        else:
            self.height -= distance

    def adjust(self, distance):
        if distance < 0:
            self.down(abs(distance))
        else:
            self.up(abs(distance))