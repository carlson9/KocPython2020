from hw1 import *
import unittest

class PortfolioTest(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio()
        self.mut1 = MutualFund("MUTA")
        self.mut2 = MutualFund("MUTB")
        self.stock1 = Stock(25, "STKA")
        self.stock2 = Stock(36.52, "STKB")
        self.bond1 = Bonds(19, "BNDA")
        self.bond2 = Bonds(44, "BNDB")

    def test_an_empty_portfolio(self):
        self.assertEqual(0.0, self.portfolio.cash)
        self.assertEqual({}, self.portfolio.assets['mutual funds'])
        self.assertEqual({}, self.portfolio.assets['bonds'])
        self.assertEqual({}, self.portfolio.assets['stock'])
        self.assertEqual("Portfolio initialized\n", self.portfolio.hist)

    def test_assets(self):
        self.assertEqual(1, self.mut1.price)
        self.assertEqual(1, self.mut2.price)
        self.assertEqual(25.00, self.stock1.price)
        self.assertEqual(36.52, self.stock2.price)
        self.assertEqual(19.00, self.bond1.price)
        self.assertEqual(44, self.bond2.price)

        self.assertEqual("MUTA", self.mut1.name)
        self.assertEqual("MUTB", self.mut2.name)
        self.assertEqual("STKA", self.stock1.name)
        self.assertEqual("STKB", self.stock2.name)
        self.assertEqual("BNDA", self.bond1.name)
        self.assertEqual("BNDB", self.bond2.name)

    def test_add_cash(self):
        self.portfolio.addCash(30956.45)
        self.assertEqual(30956.45, self.portfolio.cash)
        self.assertTrue("Added $30956.45" in self.portfolio.hist)

    def test_withdraw_cash(self):
        self.portfolio.withdrawCash(345)
        self.assertEqual(0, self.portfolio.cash)

        self.portfolio.addCash(30956.45)
        self.portfolio.withdrawCash(56.45)
        self.assertEqual(30900, self.portfolio.cash)
        self.assertTrue("Withdrew $56.45" in self.portfolio.hist)

    def test_buy_stock(self):
        self.portfolio.buyStock(100, self.stock1)
        self.assertEqual({}, self.portfolio.assets['stock'])

        self.portfolio.addCash(10000)
        self.portfolio.buyStock(100, self.stock1)
        self.assertEqual({self.stock1: 100}, self.portfolio.assets['stock'])
        self.assertEqual(10000-100*25.0, self.portfolio.cash)
        self.assertTrue("Bought 100 of stock named STKA" in self.portfolio.hist)

        self.assertTrue(self.stock2 not in self.portfolio.assets['stock'])

    def test_sell_stock(self):
        self.portfolio.sellStock(100, self.stock1)
        self.assertEqual({}, self.portfolio.assets['stock'])
        self.assertEqual(0.0, self.portfolio.cash)

        self.portfolio.addCash(10000)
        self.portfolio.buyStock(100, self.stock1)
        self.portfolio.sellStock(50, self.stock1)
        self.assertEqual(50, self.portfolio.assets['stock'][self.stock1])
        newcash = 10000 - 100*25.0
        self.assertTrue(self.portfolio.cash <= newcash +50*25.0*1.5 and 50*25.0*.5 + newcash <= self.portfolio.cash)
        self.assertTrue("Sold 50 of stock named STKA" in self.portfolio.hist)

    
    def test_buy_mutual_fund(self):
        self.portfolio.buyMutualFund(100, self.mut1)
        self.assertEqual({}, self.portfolio.assets['mutual funds'])

        self.portfolio.addCash(10000)
        self.portfolio.buyMutualFund(100, self.mut1)
        self.assertEqual({self.mut1: 100}, self.portfolio.assets['mutual funds'])
        self.assertEqual(10000-100, self.portfolio.cash)
        self.assertTrue("Bought 100 of mutual funds named MUTA" in self.portfolio.hist)

        self.assertTrue(self.mut2 not in self.portfolio.assets['mutual funds'])

    def test_sell_mutual_fund(self):
        self.portfolio.sellMutualFund(100, self.mut1)
        self.assertEqual({}, self.portfolio.assets['mutual funds'])
        self.assertEqual(0.0, self.portfolio.cash)

        self.portfolio.addCash(10000)
        self.portfolio.buyMutualFund(100, self.mut1)
        self.portfolio.sellMutualFund(50, self.mut1)
        self.assertEqual(50, self.portfolio.assets['mutual funds'][self.mut1])
        newcash = 10000 - 100
        self.assertTrue(self.portfolio.cash <= newcash +50*1.2 and 50*.9 + newcash <= self.portfolio.cash)
        self.assertTrue("Sold 50 of mutual funds named MUTA" in self.portfolio.hist)
        
    def test_buy_bonds(self):
        self.portfolio.buyBonds(100, self.bond1)
        self.assertEqual({}, self.portfolio.assets['bonds'])

        self.portfolio.addCash(10000)
        self.portfolio.buyBonds(100, self.bond1)
        self.assertEqual({self.bond1: 100}, self.portfolio.assets['bonds'])
        self.assertEqual(10000-100*19.0, self.portfolio.cash)
        self.assertTrue("Bought 100 of bonds named BNDA" in self.portfolio.hist)

        self.assertTrue(self.bond2 not in self.portfolio.assets['bonds'])

    def test_sell_bonds(self):
        self.portfolio.sellBonds(100, self.bond1)
        self.assertEqual({}, self.portfolio.assets['bonds'])
        self.assertEqual(0.0, self.portfolio.cash)

        self.portfolio.addCash(10000)
        self.portfolio.buyBonds(100, self.bond1)
        self.portfolio.sellBonds(50, self.bond1)
        self.assertEqual(50, self.portfolio.assets['bonds'][self.bond1])
        newcash = 10000 - 100*19.0
        self.assertTrue(self.portfolio.cash <= newcash +50*19*1.2 and 50*.9*19 + newcash <= self.portfolio.cash)
        self.assertTrue("Sold 50 of bonds named BNDA" in self.portfolio.hist)

    def test_print(self):
        self.assertTrue('cash: $' in self.portfolio.__str__())
        self.assertTrue('mutual funds:' in self.portfolio.__str__())
        self.assertTrue('stock:' in self.portfolio.__str__())
        self.assertTrue('bonds:' in self.portfolio.__str__())

        self.portfolio.addCash(10000)
        self.portfolio.buyBonds(100, self.bond1)
        self.portfolio.buyStock(5, self.stock1)
        self.portfolio.buyMutualFund(10, self.mut1)
        self.assertTrue('BNDA' in self.portfolio.__str__())
        self.assertTrue('STKA' in self.portfolio.__str__())
        self.assertTrue('MUTA' in self.portfolio.__str__())
        self.assertTrue('100' in self.portfolio.__str__())
        self.assertTrue('5' in self.portfolio.__str__())
        self.assertTrue('10' in self.portfolio.__str__())

                
if __name__ == '__main__':
    unittest.main()

