from urllib import *

url1 = urllib.parse.urljoin("http://www.wustl.edu", "bob/test.html")
url2 = urllib.parse.urljoin("http://www.wustl.edu", "/")
url3 = urllib.parse.urljoin("http://www.wustl.edu", "http://www.cnn.com")
url4 = urllib.parse.urljoin("http://www.wustl.edu", "http://www.cnn.com/test.html")

for url in [url1, url2, url3, url4]:
  p = urllib.parse.urlsplit(url)
  print("{0}://{1}{2}: {3}".format(p.scheme, p.hostname, p.path, "is wustl" if (p.hostname == "www.wustl.edu") else "is not wustl"))
  
#go to a webpage and extract all links. then filter which ones are of the same host
