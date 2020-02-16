from bs4 import BeautifulSoup
import csv 
from nltk.util import clean_html
import urllib
import re

def webcrawler(csvwriter, page_to_scrape = 'http://www.mathofpolitics.com'):
    webpage = urllib.request.urlopen(page_to_scrape)
    soup = BeautifulSoup(webpage.read(), 'lxml')
    soup.prettify()
    links=[]
    for link in soup.findAll('a'):
      try:
        if link['rel']==['bookmark']: #if link is a bookmark, points to a blog post
            links.append(link['href'])
      except KeyError:
        pass
    links2 = []
    for link in links: #get rid of duplicates
        if link not in links2:
            links2.append(link)
    csvwriter.writerow([page_to_scrape, 0, 'NA', 'NA', 'NA']) #if there are bookmark links on the page, it is not a post page, so all entries are NA
    for link in links2:
        getInfo(csvwriter, str(link)) #get the info for each blog post
    prev_div = soup.findAll('div', attrs = {'class':'nav-previous'})[0] #checks for older posts link on nonblog post pages
    if prev_div.findAll('a'): #if it contains a link
        webcrawler(csvwriter, str(prev_div.findAll('a')[0]['href'])) #recursively run this function with older post link

def getInfo(csvwriter, page_to_scrape):
    webpage = urllib.request.urlopen(page_to_scrape)
    soup = BeautifulSoup(webpage.read(), 'lxml')
    soup.prettify()
    date = soup.findAll('time', attrs={'class':'entry-date'})[0] #find time
    date = re.sub(r'<[^>]+>', '', str(date)) #remove tags
    title = soup.findAll('title')[0] #find title of blog post
    title = re.sub(r'<[^>]+>', '', str(title)) #remove tags
    comment_count = len(soup.findAll('div', attrs={'class':'comment-content'})) #counts number of comments - all div of class comment-content
    csvwriter.writerow([page_to_scrape, 1, date, title, comment_count]) #add row

headers = ["url", "is_post", "publish_date", "post_title", "comment_count"] #header
filename = "mathofpolitics.csv"
readFile = open(filename, "w")
csvwriter = csv.writer(readFile)
csvwriter.writerow(headers)
webcrawler(csvwriter)
readFile.flush()
readFile.close()

