import random

class Portfolio(object):
    def __init__(self):
        self.cash = 0.00
        self.assets = {"stock" : {}, "mutual funds" : {}, "bonds" : {}} #dictionary will use asset classes as reference to number owned
        self.hist = "Portfolio initialized\n"

    def addCash(self, cash):
        self.cash += int(100*cash)/100.0    #ensures adding currency compatible numbers
        self.hist+="Added $%.2f\n" %(int(100*cash)/100.0)

    def withdrawCash(self, cash):
        if cash > self.cash: print("Portfolio does not contain enough cash.")
        else:
            self.cash -= int(100*cash)/100.0
            self.hist+="Withdrew $%.2f\n" %(int(100*cash)/100.0)

    def buyAsset(self, number, asset):
        if self.cash < number*asset.price:
            print("Portfolio does not contain enough cash.")
            return None
        self.withdrawCash(number*asset.price)
        if asset in self.assets[asset.getClass()]:
            self.assets[asset.getClass()][asset]+=number #see below for getClass()
        else: self.assets[asset.getClass()][asset] = number
        self.hist+="Bought %d of %s named %s\n" % (number, asset.getClass(), asset.name)

    def buyStock(self, number, asset): self.buyAsset(int(number), asset) #same as buyAsset, but enforcing integer purchases

    buyMutualFund = buyBonds = buyAsset #exactly the same as buyAsset

    def sellAsset(self, number, asset):
        if asset in self.assets[asset.getClass()]: #check that it's in the portfolio
            if self.assets[asset.getClass()][asset] < number: #check that there is enough to sell
                print("The portfolio does not contain enough of %s %s" %(asset.name, asset.getClass()))
            else:
                self.assets[asset.getClass()][asset]-=number
                if self.assets[asset.getClass()][asset] == 0: #check if sold all of it - delete key if so
                    del self.assets[asset.getClass()][asset]
                self.addCash(number*asset.SellPrice()) #call function asset.SellPrice to calculate price of asset
                self.hist+="Sold %d of %s named %s\n" % (number, asset.getClass(), asset.name)
        else: print("The portfolio does not contain %s with name %s" %(asset.getClass(), asset.name))

    def sellStock(self, number, asset): self.sellAsset(int(number), asset) #enforce integer sales

    sellMutualFund = sellBonds = sellAsset

    def __str__(self):
        output = "cash: $%-15.2f\n" %self.cash
        for asset in self.assets:
            output+= "%s: \n"%asset
            if not self.assets[asset]: output+='\tnone\n'
            for ast in self.assets[asset]:
                output += str(self.assets[asset][ast]).rjust(5) + str(ast.name).rjust(5) + "\n"
        return output

    def history(self): print(self.hist)

class Asset(object): #superclass for stocks, bonds, and mutual funds
    def __init__(self, price, name):
        self.price = price
        self.name = name

    def SellPrice(self):
        return int(100*random.uniform(.9*self.price, 1.2*self.price))/100.0 #we'll make bonds and mutual funds sell by the same distribution
    
    
class Stock(Asset):
    def __init__(self, price, name):
        Asset.__init__(self, price, name)

    def getClass(self): return "stock" #a simple way to get the class as a string to use for calling the asset dictionary

    def SellPrice(self):
        return int(100*random.uniform(.5*self.price, 1.5*self.price))/100.0 #change the distribution for stock sales

class MutualFund(Asset):
    def __init__(self, name):
        Asset.__init__(self, 1.0, name)

    def getClass(self): return "mutual funds"


class Bonds(Asset):
    def __init__(self, price, name):
        Asset.__init__(self, price, name)

    def getClass(self): return "bonds"



