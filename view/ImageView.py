from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap


class ImageView(QLabel):
    """
    自定义ImageView类
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.callback = None
        self.hoverImage = None
        self.doubleClick = None

    def setImage(self, path):
        self.setPixmap(QPixmap(path))

    def setHoverImage(self, path):
        self.hoverImage = QPixmap(path)

    def setDoubleClick(self, callback):
        self.doubleClick = callback

    def mousePressEvent(self, event):
        print(self.objectName(), "pressed")
        QLabel.mousePressEvent(self, event)
        if self.callback is not None:
            self.callback(self)

    def setCallback(self, callback):
        self.callback = callback

    def enterEvent(self, event):
        print(self.objectName(), "hover")
        QLabel.enterEvent(self, event)
        if self.hoverImage is not None:
            self.setPixmap(self.hoverImage)

    def mouseDoubleClickEvent(self, event):
        QLabel.mouseDoubleClickEvent(self, event)
        if self.doubleClick is not None:
            self.doubleClick(self)
