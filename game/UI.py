from Tkinter import *
from PIL import Image, ImageTk
import pygame
import tkFont
from Model import Parameters


class UserInterface:
    def __init__(self, controller):
        self.controller = controller

        self.root = Tk()
        self.root.title("Memory Game")
        self.root.attributes("-fullscreen", False)
        self.frame = Frame(self.root)
        self.frame.pack()
        self.game = None

        self.root.geometry("1100x700")
#        self.width = self.root.winfo_screenwidth()
#        self.height = self.root.winfo_screenheight()
        self.width = 1100 #640
        self.height = 700 #480
        
        self.root.bind("<Escape>", self.quit)

        self.color_button = "#7030A0"
        self.color_button2 = "#AE78D6"
        self.color_button_selected = "#7030A0"
        self.color_background = "#CEC5DE"
        self.color_player_turn = "#33FF00"
        self.label_font = tkFont.Font(family='Helvetica', size='20')
        self.label_font_message = tkFont.Font(family='Helvetica', size='20', weight=tkFont.BOLD)
        self.label_font_small = tkFont.Font(family='Helvetica', size='14', weight=tkFont.BOLD)

        self.background_image = None
        self.background_label = None

        self.sidebar_message = None

        # game variables
        self.d = 0
        self.x_cards = 0
        self.y_cards = 0
        self.n_lines = 0
        self.n_columns = 0
        self.n_cards = 0
        self.player1_type = 0
        self.player2_type = 0

    def run(self):
        self.draw_initial_screen()
        self.root.mainloop()

    # ---------------------- // ----------------------
    # ---------------------- util ----------------------
    def add_button(self, x0, y0, text0, command0):
        b = Button(self.root, font=tkFont.Font(family='Forte', size='20'), text=text0, fg="white", bg=self.color_button,
                   activebackground=self.color_button_selected, width=15, command=command0)
        b.place(x=x0, y=y0)

    def add_button_smaller(self, x0, y0, text0, command0):
        b = Button(self.root, font=tkFont.Font(family='Forte', size='20'), text=text0, fg="white", bg=self.color_button,
                   activebackground=self.color_button_selected, width=10, command=command0)
        b.place(x=x0, y=y0)

    def add_label(self, x0, y0, text0):
        l = Label(self.root, text=text0, bg=self.color_background, font=self.label_font)
        l.place(x=x0, y=y0)

    def add_label_small(self, x0, y0, text0):
        l = Label(self.root, text=text0, bg=self.color_button, font=self.label_font_small, fg="white")
        l.place(x=x0, y=y0)

    def add_label_small_selected(self, x0, y0, text0):
        l = Label(self.root, text=text0, bg=self.color_button, font=self.label_font_small, fg=self.color_player_turn)
        l.place(x=x0, y=y0)

    def add_radio_buttons(self, options, x0, x1, y0, text0, command0, var_name):
        self.add_label(x0, y0, text0)
        radiogroup = Frame(self.root)
        Parameters.values[var_name] = IntVar()
        Parameters.values[var_name].set(1)  # initialization
        for txt, val in options:
            Radiobutton(radiogroup, text=txt, variable=Parameters.values[var_name], command=command0,
                        value=val).pack(side=LEFT)
        radiogroup.place(x=x1, y=y0)

    def add_scale(self, x0, y0, from0, to0, interval0, set0, var_name):
        Parameters.values[var_name] = Scale(self.root, from_=from0, to=to0, resolution=interval0, orient=HORIZONTAL)
        Parameters.values[var_name].set(set0)
        Parameters.values[var_name].place(x=x0, y=y0)

    def clear_screen(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.root.update()
        self.root.update_idletasks()

    def print_sidebar_message(self,
            text0="                          \n                          \n                          "):
        x0 = self.width*Parameters.sidebar_proportion/200-125 + 35
        y0 = self.height * 12 / 20 + 10

        self.sidebar_message = Label(self.root,
            text="                        \n                        \n                        ",
            bg="white", font=self.label_font_message, fg="black")
        self.sidebar_message.place(x=x0, y=y0)

        self.sidebar_message = Label(self.root, text=text0, bg="white", font=self.label_font_message, fg="black")
        self.sidebar_message.place(x=x0, y=y0)

    def quit(self, event=None):
        self.root.quit()

    def set_background_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((self.width, self.height))
        self.background_image = ImageTk.PhotoImage(image)
        self.background_label = Label(self.root, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # ---------------------- // ----------------------
    # ---------------------- screens ----------------------
    def draw_calibration_screen(self, n_lines, n_columns):
        self.clear_screen()
        self.set_background_image("img/background_calibration.gif")
        self.add_button_smaller(self.width * 8 / 10, self.height * 1 / 20, "Menu", self.draw_initial_screen)

        self.n_lines = n_lines
        self.n_columns = n_columns
        self.n_cards = n_lines*n_columns
        self.d = Parameters.card_size + Parameters.space_between_cards[str(self.n_cards)]

        self.x_cards = self.width * Parameters.sidebar_proportion // 200 + (self.width - self.n_columns * self.d) // 2
        self.y_cards = (self.height * 11 / 10 - self.n_lines * self.d) // 2

        self.draw_hidden_cards()
        self.draw_game_sidebar()

    def draw_game_options_screen(self):
        self.clear_screen()
        self.set_background_image("img/background_game_options.gif")
        self.add_button_smaller(self.width * 8 / 10, self.height * 1 / 20, "Menu", self.draw_initial_screen)

        options = Parameters.player_type_options
        self.add_radio_buttons(options, self.width / 7, self.width * 3 / 7, self.height * 6 / 20, "Player 1",
                               None, "player1")
        self.add_radio_buttons(options, self.width / 7, self.width * 3 / 7, self.height * 8 / 20, "Player2",
                               None, "player2")
        self.add_label(self.width / 7, self.height * 10 / 20, "Difficulty (AI)")
        set0 = Parameters.values["difficulty"]
        self.add_scale(self.width * 3 / 7, self.height * 10 / 20, 1, 10, 1, set0, "difficulty")
        options = Parameters.table_dimensions_options
        self.add_radio_buttons(options, self.width / 7, self.width * 3 / 7, self.height * 12 / 20, "Number of cards",
                               None, "n_cards")
        options = Parameters.theme_options
        self.add_radio_buttons(options, self.width / 7, self.width * 3 / 7, self.height * 14 / 20, "Theme", None,
                               "theme")

        self.add_button(self.width * 7 / 10, self.height * 17 / 20, "Play!", self.controller.start_game)

    def define_game(self, game):
        self.game = game

    def kill_game(self):
        self.game.kill()
        pygame.time.wait(1500)
        self.draw_initial_screen()

    def draw_game_screen(self, n_lines, n_columns, player1_type_index, player2_type_index):
        self.clear_screen()
        self.set_background_image("img/background_game.gif")
        self.add_button_smaller(self.width * 81 / 100, self.height * 1 / 50, "Menu", self.kill_game)

        self.n_lines = n_lines
        self.n_columns = n_columns
        self.n_cards = self.n_lines * self.n_columns
        self.player1_type = Parameters.player_type_options[player1_type_index - 1][0]
        self.player2_type = Parameters.player_type_options[player2_type_index - 1][0]

        self.d = Parameters.card_size + Parameters.space_between_cards[str(self.n_cards)]
        self.x_cards = self.width * Parameters.sidebar_proportion // 200 + (self.width - self.n_columns * self.d) // 2
        self.y_cards = (self.height * 11 / 10 - self.n_lines * self.d) // 2

        self.draw_game_sidebar()
        self.update_game_score(0, 0, True)
        self.draw_hidden_cards()

    def draw_initial_screen(self):
        self.clear_screen()
        self.set_background_image("img/background_initial.gif")

        self.add_button(self.width / 7, self.height * 8 / 20, "Game", self.draw_game_options_screen)
        self.add_button(self.width / 7, self.height * 11 / 20, "Calibration", self.controller.calibrate)
        self.add_button(self.width / 7, self.height * 14 / 20, "Settings", self.draw_settings_screen)

    def draw_settings_screen(self):
        self.clear_screen()
        self.set_background_image("img/background_settings.gif")
        self.add_button_smaller(self.width * 8 / 10, self.height * 1 / 20, "Menu", self.controller.settings_update)

        # Jeu
        self.add_label(self.width / 14, self.height * 6 / 20, "Flash duration (ms)")
        set0 = Parameters.get_value("time_ms_flash")
        self.add_scale(self.width * 3 / 7, self.height * 6 / 20, 40, 150, 5, set0, "time_ms_flash")

        self.add_label(self.width / 14, self.height * 8 / 20, "Time between flashes (ms)")
        set0 = Parameters.get_value("time_ms_between_flashes")
        self.add_scale(self.width * 3 / 7, self.height * 8 / 20, 150, 1000, 5, set0, "time_ms_between_flashes")

        self.add_label(self.width / 14, self.height * 10 / 20, "Number of flash series")
        set0 = Parameters.get_value("n_repetitions")
        self.add_scale(self.width * 3 / 7, self.height * 10 / 20, 2, 5, 1, set0, "n_repetitions")

        self.add_label(self.width / 14, self.height * 12 / 20, "Thinking time before flashes (s)")
        set0 = Parameters.get_value("time_thinking")
        self.add_scale(self.width * 3 / 7, self.height * 12 / 20, 2, 10, 0.2, set0, "time_thinking")

        self.add_label(self.width / 14, self.height * 14 / 20, "Pause before result display (s)")
        set0 = Parameters.get_value("time_before_result")
        self.add_scale(self.width * 3 / 7, self.height * 14 / 20, 1, 2, 0.1, set0, "time_before_result")

        self.add_label(self.width / 14, self.height * 16 / 20, "Result display time (s)")
        set0 = Parameters.get_value("time_result")
        self.add_scale(self.width * 3 / 7, self.height * 16 / 20, 1, 2.5, 0.1, set0, "time_result")

        # Calibration
        self.add_label_small(self.width * 7 / 12, self.height * 25 / 40, "Number of targets")
        set0 = Parameters.get_value("n_targets")
        self.add_scale(self.width * 6 / 7, self.height * 25 / 40, 5, 25, 1, set0, "n_targets")

        self.add_label_small(self.width * 7 / 12, self.height * 28 / 40, "Target display duration (s)")
        set0 = Parameters.get_value("time_show_target")
        self.add_scale(self.width * 6 / 7, self.height * 28 / 40, 0.5, 2, 0.1, set0, "time_show_target")

        self.add_label_small(self.width * 7 / 12, self.height * 31 / 40, "Pause before target display (s)")
        set0 = Parameters.get_value("time_pause_before_target")
        self.add_scale(self.width * 6 / 7, self.height * 31 / 40, 0.5, 2.5, 0.1, set0, "time_pause_before_target")

        self.add_label_small(self.width * 7 / 12, self.height * 34 / 40, "Pause after target display (s)")
        set0 = Parameters.get_value("time_pause_after_target")
        self.add_scale(self.width * 6 / 7, self.height * 34 / 40, 0.5, 2.5, 0.1, set0, "time_pause_after_target")

    # ---------------------- // ----------------------
    # ---------------------- calibration ----------------------
    def target_card(self, index):
        self.draw_card(Parameters.target_card, index)

    # ---------------------- // ----------------------
    # ---------------------- game ----------------------
    def draw_card(self, figure_name, index):
        x0, y0 = self.get_position_from_card_index(index)
        # Label(self.root, image=img).place(x=x0, y=y0)
        img = ImageTk.PhotoImage(file=figure_name)
        b = Button(self.root, image=img, bg=self.color_background, relief=FLAT,
                   command=lambda: self.controller.set_last_card_click(index))
        b.image = img
        b.place(x=x0, y=y0)
        # Button(self.root, image=ImageTk.PhotoImage(file=figure_name), command=self.root.quit).place(x=x0, y=y0)

    def draw_clock(self, message, time_in_ms):
        R = 150
        dx = 5
        x0 = self.width*Parameters.sidebar_proportion/200-R/2-dx
        y0 = self.height * 6 / 20
        canvas = Canvas(self.root, width = R+dx*2, height= R+dx*2+30, bg = self.color_button, highlightthickness=0)
        canvas.place(x=x0, y=y0)

        self.print_sidebar_message(message)
        canvas.create_arc(dx, dx, dx+R, dx+R, start=90, extent=359, fill="#FFCC00")
        dt = time_in_ms/72
        i = 0
        while i < 360:
            canvas.create_arc(dx, dx, dx+R, dx+R, start=90, extent=-i, fill="#999999")
            pygame.time.wait(dt)
            time_in_ms -= dt
            self.add_label_small(x0+dx+30, y0+R+2*dx, str(time_in_ms/1000) + "." + str((time_in_ms/10)%100) + " s")
            i += 5
        canvas.create_arc(dx, dx, dx+R, dx+R, start=90, extent=-359, fill="#999999")
        self.add_label_small(x0+dx+30, y0+R+2*dx, "0.00 s")
        self.print_sidebar_message()

    def draw_game_sidebar(self):
        x0 = self.width*Parameters.sidebar_proportion/200-125
        y0 = self.height * 12 / 20
        image = Image.open("img/obelix sign 8.gif")
        photo = ImageTk.PhotoImage(image)
        label = Label(self.root, image=photo, bg=self.color_button)
        label.image = photo # keep a reference!
        label.place(x=x0, y=y0)
        pass

    def draw_hidden_cards(self):
        for index in range(0, self.n_cards):
            self.draw_card(Parameters.hidden_card, index)
        pass

    def flash_cards(self, flashing_cards):
        for i in range(0, len(flashing_cards)):
            self.draw_card(Parameters.flashing_card, flashing_cards[i])

    def unflash_cards(self, flashing_cards):
        for i in range(0, len(flashing_cards)):
            self.draw_card(Parameters.hidden_card, flashing_cards[i])

    def get_figures(self, theme_prefix):
        figures = [""] * len(Parameters.figures_names)
        for i in range(0, len(Parameters.figures_names)):
            figures[i] = theme_prefix + Parameters.figures_names[i]
        return figures

    def get_position_from_card_index(self, index):
        x = (index % self.n_columns) * self.d
        y = (index // self.n_columns) * self.d
        return x + self.x_cards, y + self.y_cards

    def hide_card(self, index):
        self.draw_card(Parameters.hidden_card, index)

    def turn_card(self, card_name, index):
        self.draw_card(card_name, index)

    def update_game_score(self, player1_points, player2_points, is_player1_turn):
        if is_player1_turn:
            self.add_label_small_selected(self.width / 40, self.height * 3 / 20,
                                 ">> Player 1 << (" + self.player1_type + "): " + str(player1_points) + "  ")
            self.add_label_small(self.width / 40, self.height * 4 / 20,
                                 "     Player 2      (" + self.player2_type + "): " + str(player2_points) + "  ")
        else:
            self.add_label_small(self.width / 40, self.height * 3 / 20,
                                 "     Player 1      (" + self.player1_type + "): " + str(player1_points) + "  ")
            self.add_label_small_selected(self.width / 40, self.height * 4 / 20,
                                 ">> Player 2 << (" + self.player2_type + "): " + str(player2_points) + "  ")
