
def FizzBuzz(i):
  try:
    if i % 15 == 0:
      raise Exception("Divisible by 3 and 5!")
    if i % 3 == 0:
      return "Fizz"
    if i % 5 == 0:
      return "Buzz"
    print("finally")
  except:
    if i % 15 == 0:
      return("FizzBuzz")
  else:
    return str(i)
  finally:
    print("finally")
  

for i in range(18):
  print(str(i) + ": " + FizzBuzz(i))
