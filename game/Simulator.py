import pygame
import random
import sys
import time
import zmq


class OpenVibeSimulator:
    # ---------------------- parameters ----------------------
    port = "5555"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)

    processingTime = 500

    # ---------------------- public interface methods ----------------------
    def __init__(self):
        self.stop = False

    def run(self):
        while not self.stop:
            code_event = OpenVibeSimulator.socket.recv()
            if code_event == 'tagging':
                prob_values = self.get_probabilities()
                print("OpenVibeSimulator [tagging], probabilities = " + str(prob_values[0]) + " " + str(prob_values[1]))
                OpenVibeSimulator.socket.send(str(prob_values[0]) + " " + str(prob_values[1]))

            elif code_event == '0':
                # pygame.time.wait(self.processingTime)
                # OpenVibeSimulator.socket.send('0')
                pass
            elif code_event == '1':
                # pygame.time.wait(self.processingTime)
                # OpenVibeSimulator.socket.send('0')
                pass

    def stop(self):
        self.stop = True

    # ---------------------- private methods ----------------------
    def get_probabilities(self):
        pygame.time.wait(self.processingTime)
        return random.random(), random.random()

# ---------------------- main ----------------------
sim = OpenVibeSimulator()
sim.run()
