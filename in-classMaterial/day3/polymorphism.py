class Animal(object):
  living="Yes!"
  def __init__(self, name):    # Constructor of the class
      self.name = name
      
  def talk(self):              # Abstract method, defined by convention only
  	raise NotImplementedError("Subclass must implement abstract method")
  	 
class Cat(Animal):
  def talk(self):
    return self.meow()
    
  def meow(self):
    return 'Meow!'
 
class Dog(Animal):
  def talk(self):
    return self.bark()
  
  def bark(self):
    return 'Woof! Woof!'
      
class Fish(Animal):
  
  def swim(self):
    pass
  
  def __str__(self):
    return "I am a fish!"
      
animals = [Cat('Foo'),
           Dog('Bar'),
           Fish('nemo')]
 
# for animal in animals:
#   print(animal.name + ': ' + animal.talk())
  
# f = Fish("foo")
# print("Hi, " + str(f))


