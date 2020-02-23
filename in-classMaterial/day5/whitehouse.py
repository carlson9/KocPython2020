# Scraper to collect petition info from petitions.whitehouse.gov

from bs4 import BeautifulSoup
import csv 
from nltk.util import clean_html
import urllib 
import re

# What page? 
page_to_scrape = 'https://petitions.whitehouse.gov/'

# What info do we want? 
headers = ["Summary", "Signatures"]

# Where do we save info?
filename = "whitehouse-petitions.csv"
readFile = open(filename, "w")
csvwriter = csv.writer(readFile)
csvwriter.writerow(headers)

# Open webpage
webpage = urllib.request.urlopen(page_to_scrape)

# Parse it
soup = BeautifulSoup(webpage.read())
soup.prettify()

# Extract petitions on page
petitions = soup.findAll("a", href=re.compile('^/petition'))

print(len(petitions))
for petition in petitions:
  p = BeautifulSoup.get_text(petition)
  print(p)
  
pets = []  
for petition in petitions:
  p = BeautifulSoup.get_text(petition)
  if 'Sign It' not in p and 'Create a Petition' not in p and 'Load More' not in p: pets.append(p)

#signatures
#html tag:
#<span class="signatures-number">364,223</span>
signatures = soup.findAll("span", attrs={'class':'signatures-number'})
print(len(signatures))
sigs = []
for signature in signatures:
  s = BeautifulSoup.get_text(signature)
  sigs.append(s)

for i in range(20):
  csvwriter.writerow([pets[i], sigs[i]])

readFile.close()

#change this file to loop through all pages and scrape every petition (hint: look at the url of the page when you click load more)
#then add a third column for goal, and a fourth for percentage of goal reached





# Scraper to collect petition info from petitions.whitehouse.gov

from bs4 import BeautifulSoup
import csv 
from nltk.util import clean_html
import urllib 
import re

# What page? 
page_to_scrape = 'https://petitions.whitehouse.gov/'

# What info do we want? 
headers = ["Summary", "Signatures", "Goal", "Prop"]

# Where do we save info?
filename = "whitehouse-petitions.csv"
readFile = open(filename, "w")
csvwriter = csv.writer(readFile)
csvwriter.writerow(headers)

# Open webpage
webpage = urllib.request.urlopen(page_to_scrape)

# Parse it
soup = BeautifulSoup(webpage.read())
soup.prettify()

# Extract petitions on page
petitions = soup.findAll("a", href=re.compile('^/petition'))

print(len(petitions))
for petition in petitions:
  p = BeautifulSoup.get_text(petition)
  print(p)
  
pets = []  
for petition in petitions:
  p = BeautifulSoup.get_text(petition)
  if 'Sign It' not in p and 'Create a Petition' not in p and 'Load More' not in p: pets.append(p)

#signatures
#html tag:
#<span class="signatures-number">364,223</span>
#<div class="goal-text-container"><span class="goal">100,000</span>
signatures = soup.findAll("span", attrs={'class':'signatures-number'})
goals = soup.findAll("div", attrs={'class':'goal-text-container'})
print(len(signatures))
print(len(goals))
sigs = []
gls = []
props = []
for i in range(len(signatures)):
  s = BeautifulSoup.get_text(signatures[i])
  sigs.append(s)
  g = BeautifulSoup.get_text(goals[i])
  g = re.sub(',', '', g)
  g = re.sub('\ngoal\n', '', g)
  gls.append(g)
  props.append(float(re.sub(',' , '', s))/float(g))

i = 1
while True:
  try:
    new_webpage = page_to_scrape + '?page=' + str(i)
    #do above
    i += 1
  except: break 

for i in range(len(sigs):
  csvwriter.writerow([pets[i], sigs[i], gls[i], props[i])

readFile.close()

