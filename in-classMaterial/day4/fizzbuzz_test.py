#FizzBuzzTest

import unittest
import fizzbuzz

class FizzBuzzTest(unittest.TestCase):
	
	def test_fizz(self):
		self.assertEqual('Fizz',fizzbuzz.FizzBuzz(9))
		self.assertNotEqual('Fizz',fizzbuzz.FizzBuzz(15))
	
	def test_buzz(self):
		self.assertEqual('Buzz',fizzbuzz.FizzBuzz(10))
		
	def test_fizzbuzz(self):
		self.assertEqual('FizzBuzz',fizzbuzz.FizzBuzz(15))

	def test_error(self):
		with self.assertRaises(TypeError):
			fizzbuzz.FizzBuzz('b')
			
	def test5(self):
	    self.assertEqual('Buzz',fizzbuzz.FizzBuzz(15))
  
if __name__ == '__main__': #Add this if you want to run the test with this script.
  unittest.main()
