import traceback

class CustomException(Exception): # inherits from Exception
  def __init__(self, value):
    self.value = value
      
  def __str__(self):
    return self.value

    def i_call_a_function_with_errors():
      try:
        print("Calling a function....")
        #function_with_generic_error()
        #function_with_custom_error()
        #function_with_unknown_error(1)
        function_that_does_not_exist()
        print("Tada!")
      except CustomException as inst: # `as' gives us access to the exception
        print("Custom Error Caught! Error({0})".format(inst.value))
      except NameError or AttributeError:
        print("Whoa, chill out")
      except: # any exception is caught, even ones you don't know about
        print("Default Error Caught!")
      else: # if nothing broke, then run this block
        print("No error raised.")
        traceback.print_exc() # this prints the traceback
      finally: # this block is always run
        print("Goodbye!")
          
    def function_with_generic_error():
      raise Exception("Foo!") # this method doesn't know what to do with the exception
        
    def function_with_custom_error():
      raise CustomException("Foo Bar!") # this will be handled in the function above}
        
    def function_with_unknown_error(foo):
      foo.bar()

    i_call_a_function_with_errors()
