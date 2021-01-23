from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt


class EditText(QTextEdit):
    """
    自定义的文本输入框，实现Enter键的时候触发回调
    获取输入内容的功能
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.callback = None

    def keyPressEvent(self, event):
        QTextEdit.keyPressEvent(self, event)
        # print(event.key())
        if self.callback is not None:
            # Qt.Key_Return 相当于 Enter键
            if event.key() == Qt.Key_Return:
                # print("key:", event.key())
                self.callback(self.toPlainText())

    def setCallback(self, callback):
        """
        设置回调函数
        :param callback: 回调函数
        :return: no return
        """
        self.callback = callback
