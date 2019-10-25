
import os



for x in range(600):
    command="python mainTrain.py "+str(x+1)
    print(command)
    os.system(command)