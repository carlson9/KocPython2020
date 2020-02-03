def binarify(num): 
  """convert positive integer to base 2"""
  if num<=0: return '0'
  digits=[]
  while num>0:
    digits.append(num%2)
    num=num//2
  digits=digits[::-1]
  return ''.join(str(e) for e in digits)

def int_to_base(num, base):
  """convert positive integer to a string in any base"""
  if num==0:  return '0' 
  if base<=0: return '0'
  if base==1: return '1'*num
  digits = []
  negative=False
  if num<0: num*=(-1); negative=True
  while num>0:
    digits.append(num%base)
    num=num//base
  digits=digits[::-1]
  if negative: return '-'+''.join(str(e) for e in digits)
  return ''.join(str(e) for e in digits)

def base_to_int(string, base):
  """take a string-formatted number and its base and return the base-10 integer"""
  if string=="0" or base <= 0 : return 0 
  negative=False
  if string[0]=='-': string=string[1:]; negative=True
  result = 0 
  num=len(string)
  for i in string:
    num-=1
    result+=((base**num)*int(i))
  if negative: return result*(-1)
  return result 

def flexibase_add(str1, str2, base1, base2):
  """add two numbers of different bases and return the sum"""
  return base_to_int(str1, base1)+base_to_int(str2,base2)


def flexibase_multiply(str1, str2, base1, base2):
  """multiply two numbers of different bases and return the product"""
  return base_to_int(str1,base1)*base_to_int(str2,base2) 

def romanify(num):
  """given an integer, return the Roman numeral version"""
  result = ""
  result+=(num//1000*'M')
  num%=1000
  hold=num//100
  num%=100
  if hold<=3: result+=hold*'C'
  elif hold==4: result+='CD'
  elif hold>4 and hold<9: result+=('D'+'C'*(hold-5))
  else: result+='CM'
  hold=num//10
  if hold<=3: result+=hold*'X'
  elif hold==4: result+='XL'
  elif hold>4 and hold<9: result+=('L'+'X'*(hold-5))
  else: result+='XC'
  hold=num%10
  if hold<=3: result+=hold*'I'
  elif hold==4: result+='IV'
  elif hold>4 and hold<9: result+=('V'+'I'*(hold-5))
  else: result+='IX'
  return result

