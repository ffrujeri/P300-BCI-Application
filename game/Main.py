from Controller import Controller
from UI import UserInterface


controller = Controller()
ui = UserInterface(controller)
controller.set_ui(ui)
ui.run()
