
from PyQt5.QtGui import QPalette


class Color(object):

    def getColor(self, color):
        cl = QPalette()
        cl.setColor(color)
        return cl


# red = QPalette()
# red.setColor(QColor(255, 0, 0))
