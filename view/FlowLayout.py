from PyQt5.QtWidgets import QLayout, QWidget


class FlowLayout(QLayout):

    def __init__(self):
        super(FlowLayout, self).__init__()
        self.childrenWidth = 0
        self.childrenHeight = 0
        self.children = []

    def addChild(self, widget: QWidget):
        self.childrenWidth += widget.width()
        self.childrenHeight += widget.height()
        self.children.append(widget)

