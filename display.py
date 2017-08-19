import os
import time


def pauseOne():
    time.sleep(1)


def printRoad(n, e, w, s, time):
    os.system('clear')
    tempD = n / 5
    tempM = n % 5
    for j in range(0, 20):
        i = 20 - j
        if (i <= tempD):
            if (i == 1):
                print("__" * 20 + "|  @|" + "__" * 20)
            else:
                print(" " * 40 + "|  @|")
        elif (i == tempD + 1):
            if (i == 1):
                print("__" * 20 + "|  +|" + "__" * 20)
            else:
                print(" " * 40 + "|  +|")
        else:
            if (i == 1):
                print("__" * 20 + "|   |" + "__" * 20)
            else:
                print(" " * 40 + "|   |")
    tempD = w // 5
    if (w % 5 > 0):
        tempM = 1
    else:
        tempM = 0
    print("  " * ((20 - tempD) - 1) + " " + "+" * tempM + " @" * tempD)

    tempD = e // 5
    if (e % 5 > 0):
        tempM = 1
    else:
        tempM = 0
    print("  " * 22 + " " + "@ " * tempD + "+" * tempM)

    tempD = s / 5
    tempM = s % 5
    for i in range(0, 20):
        if (i <= tempD):
            if (i == 0):
                print("__" * 20 + "    " + "__" * 20)
            else:
                print(" " * 40 + "|@  |")
        elif (i == tempD + 1):
            if (i == 0):
                print("__" * 20 + "    " + "__" * 20)
            else:
                print(" " * 40 + "|+  |")
        else:
            if (i == 0):
                print("__" * 20 + "    " + "__" * 20)
            else:
                print(" " * 40 + "|   |")
    print("Time: " + str(time))
    #pauseOne()
    return 0

if __name__ == '__main__':

    while True:
        j = printRoad(21, 17, 64, 32, "Hello")

