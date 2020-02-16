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
