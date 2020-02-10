def print_integer(integer):
	return "Here is my integer: " + str(integer)
	
def print_integer(integer):
	try:
		int(integer)
	except ValueError:
		print("Put in a number.")
	else:
		print("Here is my integer: " + str(integer))

def print_integer(integer):
	if type(integer)==int:
		print("Here is my integer: " + str(integer))
	else:
		raise Exception("This is not an integer")

def print_integer(integer):
	if type(integer)==int: 
		return "Here is my integer: " + str(integer)
	else:
		raise TypeError("Enter an integer!")
				
def print_integer(integer):
	try:
		if integer %1==0:
			return "Here is my integer: " + str(integer)
		else:
			return "This has decimals!"
	except:
		raise TypeError("Enter a number!")
		
def print_integer(integer):
	try:
		if integer %1==0:
			print("Congratulations! You entered an integer!")
		else:
			raise Exception
	except:
		raise TypeError("This is not an integer!")
	else:
		return "Here is my integer: " + str(integer)


def print_integer(integer):
	try:
		if integer %1==0:
			print("Here is my integer: " + str(integer))
		else:
			raise Exception
	except TypeError:
		print("Enter a number!")
	except:
		print("Integers can't have decimals!")

				
def print_integer(integer):
	try:
		if integer %1==0:
			print("Congratulations! You entered an integer!")
		else:
			raise Exception
	except TypeError:
		raise TypeError("Enter a number!")
	except:
		raise TypeError("Integers can't have decimals!"	)
	else:
		return "Here is my integer: " + str(integer)

		
#Create your own exception		
class CustomException(Exception): 
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return str(self.value)
    
    
def print_integer(integer):
	try:
		if integer %1==0:
			print("Congratulations! You entered an integer!")
		else:
			raise CustomException(integer%1)
	except CustomException as e:
		raise TypeError("Your number has a decimal: " + str(e.value))
	except TypeError:
# 		pass
		raise TypeError("Enter a number!")
	else:
		return "Here is my integer: " + str(integer)
	finally:
		print("I'm done!")
