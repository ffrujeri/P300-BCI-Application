from Model import *
import thread


# --------------------------------------------
# --------------------------------------------
class Controller:
    def __init__(self):
        self.game = None
        self.ui = None
        self.settings_get_values_from_xml()

    def calibrate(self):
        calibrator = Calibrator(self.ui)
        thread.start_new_thread(calibrator.run, (Parameters.get_value("n_targets"), ))

    def settings_get_values_from_xml(self):
        xml_values = Parameters.get_values_from_xml('settings.xml')
        Parameters.values.update(xml_values)

    def settings_update(self):
        Parameters.save_values_to_xml('settings.xml')
        self.ui.draw_initial_screen()

    def set_ui(self, ui):
        self.ui = ui

    def set_last_card_click(self, index):
        self.game.player1.set_last_card_click(index)
        self.game.player2.set_last_card_click(index)

    def start_game(self):
        player1_type_index = Parameters.get_value("player1")
        player2_type_index = Parameters.get_value("player2")
        difficulty = Parameters.get_value("difficulty")
        number_of_cards_index = Parameters.get_value("n_cards") - 1
        theme_index = Parameters.get_value("theme") - 1
        threshold = Parameters.threshold
        self.game = Game(self.ui, player1_type_index, player2_type_index, difficulty, number_of_cards_index,
                         theme_index, threshold)
        thread.start_new_thread(self.game.run, ())
        # self.game.run()

        """
        # DATA SIMULATION
        simulation = SimulationDecisionMaker(self.ui)
        thread.start_new_thread(simulation.simulate, ())
        # simulation.simulate()
        """