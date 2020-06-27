from gui import Ui_Form
import sys
from PyQt5 import QtWidgets 

app = QtWidgets.QApplication(sys.argv)
current_exit_code = 0

Form = QtWidgets.QWidget()
ui = Ui_Form()
ui.setupUi(Form)
Form.show()
sys.exit(app.exec_())
