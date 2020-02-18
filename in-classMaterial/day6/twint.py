import twint

c = twint.Config()
c.Username = "Jacob_Montg" #user name to search under
c.Links = "include" #return tweets sent by user containing links
twint.run.Search(c)

c = twint.Config()
c.Search = "medicare for all"
c.Min_likes = 5 #only return tweets that have at least 5 likes
twint.run.Search(c)

#search for up to 100 tweets by ZiyaOnis that were written in Turkish, translate them to English, and save to a csv
c = twint.Config()
c.Username = "ZiyaOnis"
c.Limit = 100
c.Store_csv = True
c.Output = "KocPython2020/in-classMaterial/day6/ziya.csv"
c.Lang = "tr"
c.Translate = True
c.TranslateDest = "en"
twint.run.Search(c)
