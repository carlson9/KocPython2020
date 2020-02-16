import sys
import os

os.chdir('KocPython2020/in-classMaterial/day5')

#The cleanest way to handle files (gracefully handles exceptions)
with open('readfile.txt') as f:
  #We can read files in chunks
  the_whole_thing = f.read()
  print("The Whole Thing\n*****************************************************************************\n{0}".format(the_whole_thing))
  
  #We can read files line by line
  print("\nLooping over lines\n*****************************************************************************\n")
  f.seek(0)
  lines = f.readlines()
  for l in lines:
    print("{0}".format(l))
    
  #More efficiently we can loop over the file object (i.e. we don't need the variable lines)
  print("\nLooping over the file object\n********************\n")
  f.seek(0)
  for l in f:
    print("{0}".format(l))
    
  #You can also go byte by byte (don't do this)
  print("\nByte by Byte\n********************\n")
  f.seek(0)
  next_byte = f.read(1)
  while next_byte != "":
    sys.stdout.write(next_byte)
    next_byte = f.read(1)
    
# We can also manually open and close files, now we need to handle exceptions and closing files
f =  open('readfile.txt', 'r')
print("\nManually Opened File\n********************\n")
print(f.read())
f.close()

#Writing files is easy, open command takes r, w, a plus some others
with open('writefile.txt', 'w') as f:
  #wipes the file clean and opens it
  f.write("Hi guys.")
  f.write("Does this go on the second line?")
  f.writelines(['a', 'b', 'c'])
  # f.flush() # If using the file object interactively you may need to flush the buffer

with open('writefile.txt', 'a') as f:
  #just tacks some things on the end
  f.write("\nI got appended!")
  f.flush()
