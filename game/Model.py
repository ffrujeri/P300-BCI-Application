import math
import numpy
import pygame
import random
import thread
import time
import xml.etree.ElementTree as ET
import zmq
from random import shuffle
from Tkinter import Scale
from Tkinter import IntVar
from xml.dom import minidom

from custom_termcolor import *
# --------------------------------------------
# --------------------------------------------
class AI:
    # ---------------------- parameters ----------------------
    a = 0
    memory_length = 1
    memory = []  # memory[i] is a pair [index, type] of a memorized card
    pair = None

    # ---------------------- public interface methods ----------------------

    def __init__(self, difficulty):
        self.set_difficulty(difficulty)

    def play_first_card(self, index_array):
        self.pair = self._find_pair()
        if not self.pair:
            return self._choose_random_unknown_card(index_array)
        return self.pair[0]

    def play_second_card(self, index_array):
        if not self.pair:
            self.pair = self._find_pair()
            if not self.pair:
                return self._choose_random_unknown_card(index_array)
            return self.pair[0]
        return self.pair[1]

    def update_memory(self, index_array, card=None):
        # card is a pair [index, image] to be inserted in the memory
        # index_array is an indexes array of cards that are still in the game
        if card:
            ind, im = card
            for card_from_memory in self.memory:
                if card_from_memory.ind == ind:
                    self.memory.remove(card_from_memory)
            if len(self.memory) >= self.memory_length:
                self.memory.pop(0)
            new_card = self.CardDescription(ind, im)
            self.memory.append(new_card)
        for card_from_memory in self.memory:
            if not (card_from_memory.ind in index_array):
                self.memory.remove(card_from_memory)

    def set_difficulty(self, difficulty):
        self.memory_length = difficulty

    # ---------------------- private methods and classes ----------------------

    def _choose_random_unknown_card(self, index_array):
        # randomly choose a card which is still in the game and which is not in the memory
        indexes = index_array[:]
        for card in self.memory:
            if card.ind in indexes:
                indexes.remove(card.ind)
        random.seed()
        return random.choice(indexes)

    def _find_pair(self):
        # returns indexes of two cards with the same image if such cards exist in the memory
        for i in range(len(self.memory)):
            for j in range(i + 1, len(self.memory)):
                if self.memory[i].image == self.memory[j].image:
                    return [self.memory[i].ind, self.memory[j].ind]

    class CardDescription:
        ind = 0  # index of the card
        image = 0

        def __init__(self, ind, im):
            self.ind = ind
            self.image = im


# --------------------------------------------
# --------------------------------------------
class Calibrator:
    def __init__(self, ui):
        self.n_lines = Parameters.table_dimensions_calibration[0]
        self.n_columns = Parameters.table_dimensions_calibration[1]
        self.n_cards = self.n_lines * self.n_columns
        self.ui = ui
        self.ui.draw_calibration_screen(self.n_lines, self.n_columns)
        self.communication = OpenVibeCommunication()

    def run(self, n_targets):
        cards_in_game = [i for i in range(0, self.n_cards)]
        for i in range(0, n_targets):
            index = int(self.n_cards * random.random())
            flash_matrix = Flasher.get_flash_matrix(cards_in_game, self.n_lines, self.n_columns, self.n_cards)
            self.ui.target_card(index)
            # pygame.time.wait(int(1000 * Parameters.get_value("time_show_target")))
            self.ui.draw_clock("Focus on the\ntargeted card!",
                               int(1000 * Parameters.get_value("time_show_target")))
            self.ui.hide_card(index)
            pygame.time.wait(int(1000 * Parameters.get_value("time_pause_before_target")))
            for _ in range(Parameters.get_value("n_repetitions")):
                for seq in range(0, len(flash_matrix)):
                    self.communication.communicate_calibration_flash(index in flash_matrix[seq])
                    self.ui.flash_cards(flash_matrix[seq])
                    pygame.time.wait(Parameters.get_value("time_ms_flash"))
                    self.ui.unflash_cards(flash_matrix[seq])
                    pygame.time.wait(Parameters.get_value("time_ms_between_flashes"))
            pygame.time.wait(int(1000 * Parameters.get_value("time_pause_after_target")))
        self.ui.print_sidebar_message("The calibration\nis finished!")


# --------------------------------------------
# --------------------------------------------
class DecisionMaker:
    # ---------------------- public interface methods ----------------------
    def __init__(self, threshold):
        self.number_of_cards = 0
        self.pdf = None
        self.threshold = threshold
        self.update_probabilities = False
        self.epsilon = 0.00000000000000001
        self.valid_cards_in_game = None

    def get_decision(self):
        decision = numpy.array(self.pdf).argmax()
        custom_print_in_green('self.pdf', [round(pdf, 5) for pdf in self.pdf], numpy.sum(self.pdf), 'the decision is', decision)
        return decision

    def run(self, flash_matrix, communication):
        for i in range(0, Parameters.get_value("n_repetitions")):
            for seq in range(0, len(flash_matrix)):  # flash each row of matrix
                # print("DecisionMaker.run, inside loop, i = " + str(i) + ", seq = " + str(seq))
                p = communication.get_probabilities()
                # print("DecisionMaker.run, p = " + str(p))
                self.update_pdf(flash_matrix[seq], p) #FIXME
                # self.update_pdf(seq, p)
                # if not self.update_probabilities:
                    # print("DecisionMaker.run, leaving on early stop")
                    # return
        print("DecisionMaker.run, leaving on flash finish")

    def start_deciding(self, number_of_cards, flash_matrix, communication, valid_cards_in_game):
        self.number_of_cards = number_of_cards
        self.valid_cards_in_game = valid_cards_in_game
        self.pdf = [1.0 / number_of_cards] * number_of_cards
        self.update_probabilities = True
        thread.start_new_thread(self.run, (flash_matrix, communication))

    # for test only
    def start_deciding_simulation(self, number_of_cards):
        self.number_of_cards = number_of_cards
        self.pdf = [1.0 / number_of_cards] * number_of_cards

    def stop_deciding(self):
        self.update_probabilities = False

    def update_pdf(self, flashed_cards_indexes, lf):
        flashed_cards = [0] * self.number_of_cards

        custom_print_in_blue('prob {}\t diff = {}\tindexes {} '.format(lf, lf[1] - lf[0], flashed_cards_indexes))

        for i in range(0, len(flashed_cards_indexes)):
            flashed_cards[self.valid_cards_in_game.index(flashed_cards_indexes[i])] = 1
#        custom_print_in_green(i)
#FIXME
        #flashed_cards[flashed_cards_indexes] = 1
        # custom_print_in_blue('flashed_cards', flashed_cards, 'len:', len(flashed_cards))

        # print self.pdf

        # obs: p[0] - p[1]; pdf[i] > 0
        # log_p = [0]*self.number_of_cards
        # for i in range(0, self.number_of_cards):
        #     if flashed_cards[i]:
        #         log_p[i] = (lf[1] - lf[0]) + (math.log(self.pdf[i], math.e) if self.pdf[i] > 0 else self.epsilon)
        #     else:
        #         log_p[i] = (math.log(self.pdf[i], math.e) if self.pdf[i] > 0 else self.epsilon)
        # max_log_p = max(log_p)
        #
        # exp_log_p = [math.exp(log_p[i] - max_log_p + 1.0) for i in range(self.number_of_cards)]
        # sum_elp = sum(exp_log_p)
        #
        # self.pdf = [exp_log_p[i] / sum_elp for i in range(self.number_of_cards)]
        # custom_print_in_blue('self.pdf', [round(pdf, 5) for pdf in self.pdf], numpy.sum(self.pdf))

        delta = lf[1] - lf[0]
        prior = 1. / 5
        frac_prior = prior / float(1. - prior)
        post = frac_prior * numpy.exp(delta) / (1 + frac_prior * numpy.exp(delta))

        post_no_target = post + (1 - 2 * post)
        post_target = 1 - post_no_target

        for i in flashed_cards_indexes:
            if post_target > 0.5:
                self.pdf[self.valid_cards_in_game.index(i)] += 0.6
            else:
                self.pdf[self.valid_cards_in_game.index(i)] -= 0.9

    # ---------------------- private methods ----------------------
    def compute_entropy(self):
        return sum([(-self.pdf[i] * math.log(self.pdf[i], math.e) if self.pdf[i] > 0 else 0)
                    for i in range(len(self.pdf))])


# --------------------------------------------
# --------------------------------------------
class Flasher:
    # ---------------------- public interface methods ----------------------
    # returns flash matrix - rows contain indexes of cards to flash
    @staticmethod
    def get_flash_matrix(cards_in_game, n_lines_cards, n_columns_cards, n_cards):
        # create cards in game table
        cards_in_game_table = [[0 for x in range(n_columns_cards)] for x in range(n_lines_cards)]
        for i in range(len(cards_in_game)):
            index = cards_in_game[i]
            cards_in_game_table[index/n_columns_cards][index%n_columns_cards] = 1

        # n_items = len(cards_in_game)
        n_items = n_cards
        n_items_per_group = max(Flasher.floor(n_items * Parameters.percentage_of_cards), 1)  # at least 1
        rip_matrix = Flasher.get_rip_matrix(n_items, n_items_per_group)

        min_neighbors = n_items*1000
        best_flash_matrix = None
        for i in range(100):
            # get rip matrix: each row is either 1 (flash card) or 0 (don't flash)
            rip_matrix_shuffled = zip(*rip_matrix)
            random.shuffle(rip_matrix_shuffled)
            rip_matrix_shuffled = [list(x) for x in zip(*rip_matrix_shuffled)]

            # get flash matrix: each row contains indexes of cards to flash
            flash_matrix = [0] * len(rip_matrix_shuffled)
            for i in range(0, len(rip_matrix_shuffled)):
                flash_matrix[i] = [0] * sum(rip_matrix_shuffled[i])
                counter = 0
                for j in range(0, len(rip_matrix[i])):
                    if rip_matrix_shuffled[i][j] == 1:
                        flash_matrix[i][counter] = cards_in_game[j]
                        counter += 1

            # count number of neighbors, save result if minimal
            count_neighbors = Flasher.count_neighbors(flash_matrix, cards_in_game_table, n_lines_cards, n_columns_cards)
            if count_neighbors < min_neighbors:
                best_flash_matrix = flash_matrix
                min_neighbors = count_neighbors
        return best_flash_matrix

    # ---------------------- private methods ----------------------
    @staticmethod
    def ceiling(x):
        y = x * 2 // 2
        return y + (1 if y < x else 0)

    @staticmethod
    def count_neighbors(flash_matrix, cards_in_game_table, n_lines_cards, n_columns_cards):
        counter = 0
        for i in range(len(flash_matrix)):
            for j in range(len(flash_matrix[i])):
                index = flash_matrix[i][j]
                index_line = index/n_columns_cards
                index_column = index%n_columns_cards
                if index_line > 0: # up
                    if ((index_line-1)*n_columns_cards + index_column) in flash_matrix[i]:
                        counter += 1
                if index_line < n_lines_cards-1: # down
                    if ((index_line+1)*n_columns_cards + index_column) in flash_matrix[i]:
                        counter += 1
                if index_column > 0: # left
                    if (index_line*n_columns_cards + (index_column-1)) in flash_matrix[i]:
                        counter += 1
                if index_column < n_columns_cards-1: # right
                    if (index_line*n_columns_cards + (index_column+1)) in flash_matrix[i]:
                        counter += 1
        return counter

    @staticmethod
    def create_matrix(lines, columns):
        a = [[0] * columns] * lines
        for i in range(lines):
            a[i] = [0] * columns
        return a

    @staticmethod
    def cut(a, row, column):
        b = Flasher.create_matrix(row, column)
        for i in range(0, row, 1):
            for j in range(0, column, 1):
                b[i][j] = a[i][j]
        return b

    @staticmethod
    def fill(a, row, i, j, value):
        for index in range(int(i), int(j) + 1, 1):
            a[row][index] = value
        return a

    @staticmethod
    def floor(x):
        y = x * 2 // 2
        return y - (1 if y > x else 0)

    @staticmethod
    def get_rip_matrix(n_items, n_items_per_group):
        fr = 1. / Flasher.floor(n_items / n_items_per_group)
        n = Flasher.ceiling(n_items * fr) / fr
        i_row = 1
        n_group = Flasher.ceiling(n * fr)

        a = Flasher.create_matrix(int(n), 100)

        iteration = 0
        while n_group >= 1:
            beg_win = 1
            i_col = 1
            while i_col <= (1 / fr):
                beg_w_curr = beg_win
                a = Flasher.fill(a, i_row - 1, 0, n - 1, 0)
                while beg_w_curr <= n:
                    a = Flasher.fill(a, i_row - 1, beg_w_curr - 1, beg_w_curr + n_group - 2, 1)
                    beg_w_curr += n_group / fr
                i_row += 1
                beg_win += n_group
                i_col += 1
                iteration += 1
            n_group = Flasher.ceiling(n_group * fr) if n_group > 1 else 0

        a = Flasher.cut(a, iteration, n_items)
        return a

# --------------------------------------------
# --------------------------------------------
class OpenVibeCommunication:
    # ---------------------- parameters ----------------------
    # ---------------------- public interface methods ----------------------
#    def __init__(self, ip='localhost', port='5555'):
#        self.context = zmq.Context()
#        self.socket = self.context.socket(zmq.PAIR)
#        try:
#            self.socket.connect('tcp://%s:%s' % (ip, port))
#        except:
#            print('Connection error')

#    def communicate_game_flash(self):
#        # send 1 when cards are flashed
#        self.socket.send('1')

#    def communicate_calibration_flash(self, target_flashed):
#        # sends 1 if the target wasn't flashed and 2 otherwise
#        self.socket.send('2' if target_flashed else '1')
#        # return self.socket.recv()

#    def get_probabilities(self):
#        probabilities = self.translate_reply(self.socket.recv())
#        return probabilities

#    # used for test only: pre-defined probabilities for targets and
#    # non-targets are read from file
#    def get_probabilities(self, target_flashed):
#        # TODO
#        probabilities = self.translate_reply(self.socket.recv())
#        return probabilities
    def __init__(self, ip='127.0.0.1', port_pub='5556', port_sub='6666'):
        self.context = zmq.Context()
        # self.socket = self.context.socket(zmq.PAIR)
        self.pubsocket = self.context.socket(zmq.PUB)
        self.pubsocket.bind('tcp://{}:{}'.format(ip, port_pub))


        self.subsocket = self.context.socket(zmq.SUB)
        self.subsocket.setsockopt(zmq.SUBSCRIBE,'') 
        self.subsocket.connect('tcp://{}:{}'.format(ip, port_sub))

    def communicate_game_flash(self):
        self.pubsocket.send('1')
        # self.socket.send('tagging')

    def communicate_calibration_flash(self, target_flashed):
        # sends 0 if the target wasn't flashed and 1 otherwise
        self.pubsocket.send('2' if target_flashed else '1')
        # return self.socket.recv()

    def get_probabilities(self):
        probabilities = self.translate_reply(self.subsocket.recv())
        # print
        # print '-----------ZMQ received', probabilities
        # print
        return probabilities

    def __del__(self):
        self.pubsocket.close()
        self.subsocket.close()
    # ---------------------- private methods ----------------------
    @staticmethod
    def translate_reply(reply):  # depends on the server response format
        l = reply.split(' ')
        return float(l[0]), float(l[1])


# --------------------------------------------
# --------------------------------------------
class Game:
    def __init__(self, ui, player1_type_index, player2_type_index, difficulty, table_dimensions_index, theme_index,
                 threshold):
        n_lines = Parameters.table_dimensions[table_dimensions_index][0]
        n_columns = Parameters.table_dimensions[table_dimensions_index][1]
        self.n_cards = n_lines * n_columns
        self.first_card_index = None
        self.cards_in_game = []
        self.cards = []

        self.ui = ui
        ui.define_game(self)

        self.player1 = Player(player1_type_index, difficulty, threshold, ui, n_lines, n_columns)
        self.player2 = Player(player2_type_index, difficulty, threshold, ui, n_lines, n_columns)
        self.player1_turn = True
        self.first_card_index = None
        self.first_card = 0

        self.make_set(theme_index)
        self.ui.draw_game_screen(n_lines, n_columns, player1_type_index, player2_type_index)

    def game_result(self):
        compare = self.player1.points - self.player2.points
        if compare == 0:
            self.ui.print_sidebar_message("\n  It's a draw!")
        else:
            self.ui.print_sidebar_message("\n Player " + ("1" if compare > 0 else "2") + " wins!")

    def make_set(self, theme_index):
        theme = Parameters.themes_prefix[theme_index]
        figures = self.ui.get_figures(theme)
        n_distinct_cards = self.n_cards // 2
        if n_distinct_cards is len(figures):
            self.cards = (figures[:len(figures)]) * 2
        else:
            self.cards = (figures[:n_distinct_cards % len(figures)]) * 2
        shuffle(self.cards)
        self.cards_in_game = list(range(n_distinct_cards * 2))

    def play_again(self):
        pass

    def play_turn(self, index):
        self.ui.turn_card(self.cards[index], index)
        image = self.cards[index]
        if (self.first_card_index is None) or (index == self.first_card_index):
            self.first_card_index = index
            self.player1.update_memory(self.cards_in_game, [index, image])
            self.player2.update_memory(self.cards_in_game, [index, image])
            pygame.time.wait(int(1000 * Parameters.get_value("time_result")))
            return

        equal_cards = (self.cards[index] == self.cards[self.first_card_index])
        if equal_cards:
            self.cards[index] = self.cards[self.first_card_index] = None
            self.cards_in_game.remove(index)
            self.cards_in_game.remove(self.first_card_index)
            self.player1.points += self.player1_turn
            self.player2.points += (not self.player1_turn)
            self.player1.update_memory(self.cards_in_game, [index, image])
            self.player2.update_memory(self.cards_in_game, [index, image])
            self.ui.print_sidebar_message("\n   Pair found!")
            pygame.time.wait(int(1000 * Parameters.get_value("time_result")))
        else:  # if cards turned are different
            # self.ui.print_sidebar_message("Tour du\n   joueur " + ("1" if not self.player1_turn else "2"))
            pygame.time.wait(int(1000 * Parameters.get_value("time_result")))
            self.ui.hide_card(self.first_card_index)
            self.ui.hide_card(index)
            self.player1.update_memory(self.cards_in_game, [index, image])
            self.player2.update_memory(self.cards_in_game, [index, image])

        self.ui.print_sidebar_message()
        self.player1_turn = ((not self.player1_turn) ^ equal_cards)
        # self.ui.print_sidebar_message("Tour du\n     joueur " + ("1" if self.player1_turn else "2"))
        self.first_card_index = None

    def kill(self):
        self.cards = []

    def run(self):
        print("Game.run() -- start!")

        self.ui.print_sidebar_message("\n Player 1 turn")
        pygame.time.wait(1000)
        self.ui.print_sidebar_message()

        while any(self.cards):
            if self.player1_turn:
                index = self.player1.get_play(self.first_card_index, self.cards_in_game)
            else:
                index = self.player2.get_play(self.first_card_index, self.cards_in_game)
            if index is not None:
                self.play_turn(index)
                self.ui.update_game_score(self.player1.points, self.player2.points, self.player1_turn)
            else:
                continue
        else:
            self.game_result()
            self.play_again()

        print("Game.run() -- end!")


# --------------------------------------------
# --------------------------------------------
class Parameters:
    # ---------------------- UI ----------------------
    # ui radio_buttons options
    table_dimensions_options = [("8", 1), ("12", 2), ("16", 3), ("20", 4), ("24", 5), ("30", 6)]
    theme_options = [("Asterix et Obelix", 1), ("Colors", 2), ("Disney", 3)]
    player_type_options = [("AI", 1), ("EEG headset", 2), ("Mouse", 3)]
    player_type = ["ai", "brain", "mouse"]

    # cards
    card_size = 115
    space_between_cards = dict([("8", 60), ("12", 60), ("16", 30), ("20", 30), ("24", 10), ("30", 10)])
    table_dimensions = [[2, 4], [3, 4], [4, 4], [4, 5], [4, 6], [5, 6]]
    table_dimensions_calibration = [4, 5]
    sidebar_proportion = 30
    screen_data_center = 80
    flashing_card = "img/flashing_card.gif"
    hidden_card = "img/hidden_card.gif"
    target_card = "img/target_card.gif"
    themes_prefix = ["img/themes/Asterix et Obelix/", "img/themes/Colors/", "img/themes/Disney/"]
    figures_names = ["img01.jpg", "img02.jpg", "img03.jpg", "img04.jpg", "img05.jpg", "img06.jpg", "img07.jpg",
                     "img08.jpg", "img09.jpg", "img10.jpg", "img11.jpg", "img12.jpg", "img13.jpg", "img14.jpg",
                     "img15.jpg"]

    # ---------------------- settings values (ms) ----------------------
    values = {}

    # ---------------------- flashing ----------------------
    threshold = 2.99
    percentage_of_cards = 0.2

    # ---------------------- AI ----------------------
    ai_thinking_time = 500

    @staticmethod
    def get_value(key):
        if (Parameters.values[key].__class__ is Scale) or (Parameters.values[key].__class__ is IntVar):
            return Parameters.values[key].get()
        return Parameters.values[key]

    @staticmethod
    def get_values_from_xml(file_name):
        tree = ET.parse(file_name)
        settings = tree.getroot()

        dictionary = {}
        for child in settings:
            if "time" in child.tag and not ("time_ms" in child.tag):
                dictionary[child.tag] = float((int(child.text)) / 1000.)
            else:
                dictionary[child.tag] = int(child.text)
        return dictionary

    @staticmethod
    def save_values_to_xml(file_name):
        tree = ET.ElementTree()
        settings = ET.Element('settings')
        tree._setroot(settings)

        for key in Parameters.values:
            sub_element = ET.SubElement(settings, str(key))
            if (Parameters.values[key].__class__ is Scale) or (Parameters.values[key].__class__ is IntVar):
                if "time" in key and not ("time_ms" in key):
                    text = str(int(Parameters.values[key].get() * 1000))
                else:
                    text = str(int(Parameters.values[key].get()))
            else:
                if "time" in key and not ("time_ms" in key):
                    text = str(int(Parameters.values[key] * 1000))
                else:
                    text = str(int(Parameters.values[key]))
            sub_element.text = text

        # print pretty
        f = open(file_name, 'w')
        rough_string = ET.tostring(settings, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        f.write(reparsed.toprettyxml(indent="\t"))
        f.close()


# --------------------------------------------
# --------------------------------------------
class Player:
    def __init__(self, player_type_index, difficulty, threshold, ui, n_lines_cards, n_columns_cards):
        self.type = Parameters.player_type[player_type_index - 1]
        self.points = 0
        self.last_card_click = None
        self.n_lines_cards = n_lines_cards
        self.n_columns_cards = n_columns_cards
        if self.type == "ai":
            self.ai = AI(difficulty)
        elif self.type == "brain":
            self.decision_maker = DecisionMaker(threshold)
            self.ui = ui
            self.communication = OpenVibeCommunication()

    def get_play(self, first_card_index, cards_in_game):
        is_first_card = first_card_index is None
        if self.type == "ai":
            pygame.time.wait(Parameters.ai_thinking_time)
            if is_first_card:
                return self.ai.play_first_card(cards_in_game)
            else:
                return self.ai.play_second_card(cards_in_game)
        elif self.type == "brain":
            if is_first_card:
                valid_cards_in_game = cards_in_game
                n_cards = len(cards_in_game)
            else:
                valid_cards_in_game = cards_in_game[:]
                valid_cards_in_game.remove(first_card_index)
                n_cards = len(cards_in_game)-1
            print "first_card_index = " + str(first_card_index)
            print "cards_in_game = " + str(cards_in_game)
            print "valid cards = " + str(valid_cards_in_game)
            print "ncards = " + str(n_cards)
            # pygame.time.wait(int(1000 * Parameters.get_value("time_thinking")))
            self.ui.draw_clock("\n Get ready!", int(1000 * Parameters.get_value("time_thinking")))
            flash_matrix = Flasher.get_flash_matrix(valid_cards_in_game, self.n_lines_cards, self.n_columns_cards, n_cards)
            print "flash_matrix = " + str(flash_matrix)
            self.decision_maker.start_deciding(n_cards, flash_matrix, self.communication, valid_cards_in_game)
            for i in range(0, Parameters.get_value("n_repetitions")):  # flash each row of matrix
                for seq in range(0, len(flash_matrix)):
                    self.communication.communicate_game_flash()
                    self.ui.flash_cards(flash_matrix[seq])
                    pygame.time.wait(Parameters.get_value("time_ms_flash"))
                    self.ui.unflash_cards(flash_matrix[seq])
                    pygame.time.wait(Parameters.get_value("time_ms_between_flashes"))
###                    decision = self.decision_maker.get_decision()
###                    print("Player.get_play, decision = " + str(decision))
#                    if decision is not None: # early stop, use of threshold!
#                        self.decision_maker.stop_deciding()
#                    pygame.time.wait(int(1000 * Parameters.get_value("time_before_result")))
#                    return decision
###            print("Player.get_play, random decision!")
            self.decision_maker.stop_deciding()
            pygame.time.wait(Parameters.get_value("time_ms_between_flashes"))
            decision = self.decision_maker.get_decision()
            print("Player.get_play, decision = " + str(decision))
            pygame.time.wait(int(1000 * Parameters.get_value("time_before_result")))
            return valid_cards_in_game[decision]
###            return valid_cards_in_game[int(n_cards * random.random())]
        else:
            if is_first_card:
                self.last_card_click = None
            while self.last_card_click is None:
                pygame.time.wait(50)
                if self.last_card_click == first_card_index:
                    self.last_card_click = None
            click = self.last_card_click
            self.last_card_click = None
            return click

    def set_last_card_click(self, index):
        self.last_card_click = index

    def update_memory(self, index_array, card=None):
        if self.type == "ai":
            self.ai.update_memory(index_array, card)


# --------------------------------------------
# --------------------------------------------
class SimulationDecisionMaker:
    # ---------------------- EEG data simulation (used to test DecisionMaker) ----------------------
    def __init__(self, ui):
        self.n_cards = 20
        self.n_flashes = 126
        self.simulation_matrix = open('simulation data/flash matrix.txt')
        self.simulation_target = open('simulation data/target.txt')
        self.simulation_no_target = open('simulation data/no target.txt')

        # read files
        self.flash_matrix = [0]*self.n_flashes
        self.p_target = [0]*self.n_flashes
        self.p_no_target = [0]*self.n_flashes
        for i in range(0, self.n_flashes):
            self.flash_matrix[i] = self.simulation_matrix.readline().split('\t')
            self.p_no_target[i] = [float(value) for value in self.simulation_no_target.readline().split('\t')]
            self.p_target[i] = [float(value) for value in self.simulation_target.readline().split('\t')]
        print("All files read:")
        print(self.flash_matrix)
        print(self.p_no_target)
        print(self.p_target)

        n_lines = 4
        n_columns = 5
        player1_type_index = 0
        player2_type_index = 0
        self.ui = ui
        self.ui.draw_game_screen(n_lines, n_columns, player1_type_index, player2_type_index)

        self.decision_maker = DecisionMaker(Parameters.threshold)

    def get_flash_row(self, rip_row):  # returns indexes of flashed cards
        flash_row = [1 if rip_row[i] == "TRUE" else 0 for i in range(0, len(rip_row))]
        flash_row_indexes = [0]*sum(flash_row)
        counter = 0
        for j in range(0, self.n_cards):
            if rip_row[j] == "TRUE":
                flash_row_indexes[counter] = j
                counter += 1
        return flash_row_indexes

    def simulate(self):
        print("\n---------------------------------------------")
        print("Expected decision\t\t\tObtained decision")
        print("---------------------------------------------")
        # print("----------------------------------------------------------------------")
        # print("Expected decision\t\t\tObtained decision\t\t\t# flashes")
        # print("----------------------------------------------------------------------")
        for j in range(0, self.n_cards):
            self.decision_maker.start_deciding_simulation(self.n_cards)
            # n_flashes = 0
            for i in range(0, self.n_flashes):  # flash each row of matrix
                # flashed cards
                rip_row = self.flash_matrix[i]
                flash_row = self.get_flash_row(rip_row)
                # self.ui.flash_cards(flash_row)
                # pygame.time.wait(Parameters.values["time_ms_flash"]/10)
                # self.ui.unflash_cards(flash_row)
                # pygame.time.wait(Parameters.values["time_ms_between_flashes"]/20)

                # probabilities
                p = [0, 0]
                p[0] = self.p_no_target[i][j]/10000
                p[1] = self.p_target[i][j]/10000

                # update decision_maker
                self.decision_maker.update_pdf(flash_row, p)
                # if self.decision_maker.get_decision() is not None:
                #    n_flashes = i
                #    break
                # print(self.decision_maker.compute_entropy())

            decision = self.decision_maker.get_decision()
            if decision is not None:  # decision made!
                print("\t" + str(j) + "\t\t\t\t\t\t\t\t" + str(decision))
                # print("\t" + str(j) + "\t\t\t\t\t\t\t\t" + str(decision) + "\t\t\t\t\t\t" + str(n_flashes))
                # print("\t" + str(j) + "\t\t\t\t\t\t\t\t" + str(decision) + "\t" + str(self.decision_maker.compute_entropy()))
            else:
                print("\t" + str(j) + "\t\t\t\t\t\t\t\tNONE")
                # print("\t" + str(j) + "\t\t\t\t\t\t\t\tNONE\t\t\t\t\t\t" + str(n_flashes))
                # print("\t" + str(j) + "\t\t\t\t\t\t\t\tNONE\t" + str(self.decision_maker.compute_entropy()))
        pygame.time.wait(5000)
