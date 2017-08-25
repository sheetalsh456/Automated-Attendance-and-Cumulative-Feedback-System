import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import QApplication,QPlainTextEdit,QWidget
import subprocess

class MyDialog(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MyDialog, self).__init__(parent)
        text=open('file.txt').read()

        self.buttonBox = QtGui.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)

        self.textBrowser = QtGui.QTextBrowser(self)
        self.textBrowser.append(text)

        self.verticalLayout = QtGui.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)

class QDataViewer(QtGui.QWidget):

    def __init__(self):
        QtGui.QWidget.__init__(self)
        # Layout Init.
        self.setGeometry(650, 300, 600, 600)
        self.setWindowTitle('Data Viewer')
        self.quitButton = QtGui.QPushButton('QUIT', self)
        self.uploadButton = QtGui.QPushButton('UPLOAD', self)
        self.detectorButton = QtGui.QPushButton('DETECTOR', self)
        self.recognizerButton = QtGui.QPushButton('RECOGNIZER', self)
        #self.trialButton = QtGui.QPushButton('TRIAL', self)
        #self.myTextBox = QtGui.QTextEdit(self)
        #self.text_edit = QPlainTextEdit(self)
        hBoxLayout = QtGui.QHBoxLayout()
        hBoxLayout.addWidget(self.quitButton)
        hBoxLayout.addWidget(self.uploadButton)
        hBoxLayout.addWidget(self.detectorButton)
        hBoxLayout.addWidget(self.recognizerButton)
        #hBoxLayout.addWidget(self.trialButton)
        self.setLayout(hBoxLayout)
        
        
        # Signal Init.
        self.connect(self.quitButton,   QtCore.SIGNAL('clicked()'), QtGui.qApp, QtCore.SLOT('quit()'))
        self.connect(self.uploadButton, QtCore.SIGNAL('clicked()'), self.open1)
        #self.connect(self.trialButton, QtCore.SIGNAL('clicked()'), self.trial)
        #self.trialButton.clicked.connect(self.trial)
        self.dialogTextBrowser = MyDialog(self)
        self.detectorButton.clicked.connect(lambda:self.detect('face_eye_detect.py'))
        self.recognizerButton.clicked.connect(lambda:self.recognize('face_recognizer.py'))



    def trial(self):
        self.dialogTextBrowser.exec_()


    def open1 (self):
    	#global filename
    	
        self.filename = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '.')
        #self.myTextBox.setText(self.filename)
        #print(self.filename)
        print 'Path file :', self.filename
        return self.filename

    def detect(self,path):
        subprocess.call(['python',path,self.filename])

    def recognize(self,path):
        subprocess.call(['python',path])
        self.dialogTextBrowser.exec_()
        
   

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    mw = QDataViewer()
    mw.show()
    sys.exit(app.exec_())

