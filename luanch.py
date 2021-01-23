import sys

from PyQt5.QtWidgets import QApplication

from activity.MainActivity import MainWindow

from util.Resource import res

if __name__ == '__main__':
    app = QApplication(sys.argv)
    css = res.readCssAsStream("main.qss")
    window = MainWindow()
    window.setStyleSheet(css)
    window.display()
    sys.exit(app.exec_())
