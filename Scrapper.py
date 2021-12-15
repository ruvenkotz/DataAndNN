# This code is adapted from this article: http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26646701.pdf
# Link to the github used: https://github.com/Fenmaz/connect4

import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

# Translate what was parsed from the website to game results
def gameStatus(s):
    if s == "Yellow won":
        return [0,1,0]
    elif s == "Red won":
        return [1,0,0]
    elif s == "Draw game":
        return [0,0,1]
    elif s == "Red can draw":
        return [0,0,1]
    elif s == "Yellow can draw":
        return [0,0,1]
    else: # either Red or Yellow wins with x number of moves left
        if s.find("Red") != -1 and s.find("win") != -1: # red found and wins
            return [1,0,0]
        elif s.find("Yellow") != -1 and s.find("win") != -1: # yellow found and wins
            return [0,1,0]
        elif s.find("Red") != -1 and s.find("loses") != -1: # red found and loses
            return [0,1,0]
        elif s.find("Yellow") != -1 and s.find("loses") != -1: # yellow found and loses
            return [1,0,0]

# Parse from ConnectFour Solver and write results in a CSVs
def getResults():
    with open('FixedQuery.csv') as csv_file, open("FixedResults2.csv", "w", newline="") as results_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_writer = csv.writer(results_file)
        line_count = 0
        results = []
        s = Service(ChromeDriverManager().install())
        i = 0
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(service=s, options=options)
        for row in csv_reader:
            if i > 31552:
                url = 'https://connect4.gamesolver.org/en/?pos='+row[0]
                driver.get(url)
                time.sleep(.5)
                content = driver.page_source
                soup = BeautifulSoup(content, features="html.parser")
                c = soup.find("div", {"id": "solution_header"})
                if c != None:
                    res = gameStatus(c.text)
                    if res != None:
                        csv_writer.writerow(res)
                    else:
                        csv_writer.writerow([0,0,0])
                else:
                    csv_writer.writerow([0,0,0])
            if i % 1000 == 0:
                print(i)
            i = i + 1
    return results

