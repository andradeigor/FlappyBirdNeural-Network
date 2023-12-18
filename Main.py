from cv2 import COLOR_BGR2GRAY
from mss import mss
from Genetic import Genetic
import cv2
import numpy as np
from pynput.keyboard import Key, Controller
import pygame
import time

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255,255,255)
thickness = 1
lineType  = 2

def jump(keyboard):
    keyboard.press(Key.space)
    keyboard.release(Key.space)

def reset(keyboard):
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)


def findCano(screen, cano):
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    local = cv2.matchTemplate(screen_gray, cano, cv2.TM_CCOEFF_NORMED)
    corte = 0.9
    local_cord = np.where(local >= corte)
    for i in zip(*local_cord[::-1]):
        cv2.rectangle(screen, i, (i[0] + 50, i[1] + 40), (0,0,255),2)
    return screen, local_cord[::-1]




def findBird(screen):# [103,203,248] = BGR
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower = np.array([20,100,20])
    upper = np.array([30,255,255])
    local = cv2.inRange(screen, lower, upper)
    local = np.where(local)
    local = local[::-1]
    if len(local[0]) >0:
        cv2.rectangle(screen, (local[0][0]-10,local[1][0]-10), (local[0][0] + 24, local[1][0] + 24), (0,255,255),2)
    screen = cv2.cvtColor(screen,cv2.COLOR_HSV2BGR)
    return screen, local

     
def isRunning(screen,gameOver):
    game = True
    x, y = gameOver.shape[::-1]
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
    res_bot = cv2.matchTemplate(screen_gray,gameOver,cv2.TM_CCOEFF_NORMED)
    corte = 0.56
    local_bot = np.where(res_bot >= corte)
    for i in zip(*local_bot[::-1]):
        cv2.rectangle(screen, i, (i[0] + x, i[1] + y), (0,0,255),1)
        game = False
        return screen,game
    return screen,game

def writeInfos(screen, generation, fitness,index):
    cv2.putText(screen,f"Generation:{generation}", 
    (20,20), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    cv2.putText(screen,f"Rede:{index+1}", 
    (20,40), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    cv2.putText(screen,f"Fitness:{fitness:.2f}", 
    (20,60), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    
    return screen


def main():
    gameOver = cv2.imread('templates/gameover.png')
    gameOver = cv2.cvtColor(gameOver, cv2.COLOR_BGRA2GRAY) 
    cano = cv2.imread("./templates/cano_top.png")
    cano = cv2.cvtColor(cano, cv2.COLOR_BGRA2GRAY)
    keyboard = Controller()
    
    clock = pygame.time.Clock()
    g = Genetic(30,0.01,[3,4,1],2)
    generation = 1
    for i in range(3):
        print(f"{3-(i)}...")
        time.sleep(1)
    while True:
        run = True
        for index,NN in enumerate(g.populationList):
            time.sleep(0.15)
            while run:    
                clock.tick(30)
                with mss() as sct:
                    monitor = {"top": 120, "left": 10, "width": 280, "height": 480}
                    screen = np.array(sct.grab(monitor))
                    #screen = cv2.imread("./templates/teste3.png")
                    screen,canoLocation = findCano(screen, cano)
                    screen,birdLocation = findBird(screen)
                    screen, running = isRunning(screen,gameOver) 
                    screen = writeInfos(screen,generation, NN.fitness, index)
                    
                    canoLocalizado = False
                    try:
                        canoValueTop = canoLocation[1][0]+63
                        canoLocalizado = True
                        cv2.line(screen, (10,canoValueTop), (280,canoValueTop), (255,255,255), 1) 
                        cv2.line(screen, (10,canoValueTop + 60), (280,canoValueTop + 60), (255,255,255), 1) 
                    except:
                        canoValueTop= 200
                    try:
                        birdValue = birdLocation[1][0]
                    except:
                        birdValue = 0
                    cv2.imshow("Eye", screen)
                    cv2.waitKey(1)
                    if(not running):
                        reset(keyboard)
                        break
                    if(canoLocalizado and canoValueTop < birdValue and canoValueTop + 60 > birdValue):
                        NN.fitness+=5
                    NNInput = [birdValue, canoValueTop,canoValueTop+60]
                    prediction = NN.feedforward(NNInput)
                    if(prediction > 0.5):
                        jump(keyboard)
                    NN.fitness+=0.1

                    
        g.evolve()
        generation+=1


if __name__ == "__main__":
    main()

