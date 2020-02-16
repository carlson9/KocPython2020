from bs4 import BeautifulSoup
import urllib 
import random
import time
import os
import re

# Open a web page
web_address='https://case.ku.edu.tr/en/academics/international-relations/faculty/'
web_page = urllib.request.urlopen(web_address)

# Parse it
soup = BeautifulSoup(web_page.read())
soup.prettify()

# Find all cases of a certain tag
soup.find_all('a')


# Get the attributes
my_a_tag=soup.find_all('a')[2]
re.sub(r'<[^>]+>', '', str(my_a_tag)) #remove tags
my_a_tag.attrs #Gives a dictionary with the attributes
my_a_tag.attrs.keys()
my_a_tag['href']

# Refine search by using attributes
soup.find_all('span', {'class':'name'})

# There may be tags within tags
mysection=soup.find_all('div')[0]
mysection.a #Gives the 'a' tag within the 'div' tag
mysection.find_all('a') #Gives the list of all 'a' tags within the 'div' tag
mysection.get_text()


# Creating a tree of objects

mysection.contents #Gives a list of all children
mysection.children #Creates an iterator for children

for child in mysection.children:
	print(child)

mysection.descendants #Creates an iterator for children, grandchildren, etc.

# Other methods to check family:
# parent
# parents
# next_siblings
# previous_siblings

# Beautiful Soup documentation
# http://www.crummy.com/software/BeautifulSoup/bs4/doc/

# Function to save a web page

def download_page(address,path,filename,wait=5):
	time.sleep(random.uniform(0,wait))
	page = urllib.request.urlopen(address)
	page_content = page.read()
	if os.path.exists(path+filename)==False:
		with open(path+filename, 'wb') as p_html:
			p_html.write(page_content)
	else:
		print("Can't overwrite file" + filename)

download_page('http://www.crummy.com/software/BeautifulSoup/bs4/doc/', '', 'Docket05-1.html',0)

#You can also parse a page that is saved on your computer
with open('Docket05-1.html') as f:
  #We can read files in chunks
  myfile = f.read()
  
soup = BeautifulSoup(myfile)
soup.prettify()

#Scrape the names and email addresses of INTL faculty and save the result as a csv

