import random 
import pyautogui as pg
import time
animal=('Hi','Hello','Bhagja')
time.sleep(8)
for i in range(1001):
    a=random.choice(animal)
    pg.write(""+a)
    pg.press('enter')