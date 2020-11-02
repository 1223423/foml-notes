library(twitteR)

consumerKey = "*******************"  
consumerSecret = "****************"
accessToken = "******************************"
accessSecret = "*********************"
options(httr_oauth_cache=TRUE)
setup_twitter_oauth(consumer_key = consumerKey, consumer_secret = consumerSecret,
                    access_token = accessToken, access_secret = accessSecret)
rt2 <- searchTwitter("lang:en", n = 10000, since = "2020-10-26", until = "2020-10-27")
rt3 <- searchTwitter("lang:en", n = 10000, since = "2020-10-27", until = "2020-10-28")
rt4 <- searchTwitter("lang:en", n = 10000, since = "2020-10-28", until = "2020-10-29")
rt5 <- searchTwitter("lang:en", n = 10000, since = "2020-10-29", until = "2020-10-30")
rt6 <- searchTwitter("lang:en", n = 10000, since = "2020-10-30", until = "2020-10-31")
rt7 <- searchTwitter("lang:en", n = 10000, since = "2020-10-31", until = "2020-11-01")
rt8 <- searchTwitter("lang:en", n = 10000, since = "2020-11-01", until = "2020-11-02")

newrt2 = twListToDF(rt2)
newrt3 = twListToDF(rt3)
newrt4 = twListToDF(rt4)
newrt5 = twListToDF(rt5)
newrt6 = twListToDF(rt6)
newrt7 = twListToDF(rt7)
newrt8 = twListToDF(rt8)

newrt2["created"] = "MON"
newrt3["created"] = "TUE"
newrt4["created"] = "WED"
newrt5["created"] = "THU"
newrt6["created"] = "FRI"
newrt7["created"] = "SAT"
newrt8["created"] = "SUN"

library(dplyr)

twitMON = select(newrt2, "text", "created" )
twitTUE = select(newrt3, "text", "created" )
twitWED = select(newrt4, "text", "created" )
twitTHU = select(newrt5, "text", "created" )
twitFRI = select(newrt6, "text", "created" )
twitSAT = select(newrt7, "text", "created" )
twitSUN = select(newrt8, "text", "created" )

write.csv(twitMON, "C:/Users/Augustin/Documents/MON")
write.csv(twitTUE, "C:/Users/Augustin/Documents/TUE")
write.csv(twitWED, "C:/Users/Augustin/Documents/WED")
write.csv(twitTHU, "C:/Users/Augustin/Documents/THU")
write.csv(twitFRI, "C:/Users/Augustin/Documents/FRI")
write.csv(twitSAT, "C:/Users/Augustin/Documents/SAT")
write.csv(twitSUN, "C:/Users/Augustin/Documents/SUN")