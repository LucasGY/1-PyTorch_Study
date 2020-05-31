class Animal():

    def __init__(self, animal_list):
        self.animal_list = animal_list

    def __getitem__(self, index):
        return self.animal_list[index]


animals = Animal(["dog", "cat", "fish"])
for animal in animals:
    print(animal)