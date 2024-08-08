import re
import pandas as pd

from PyQt5.QtCore import QTimer,Qt,pyqtSlot,QCoreApplication,QUrl,QThread,pyqtSignal,pyqtSlot, QThread, pyqtSignal,Qt
from PyQt5.QtGui import QIcon,QFont,QCloseEvent, QPixmap,QMovie,QColor,QBrush,QPixmap,QStandardItemModel,QStandardItem
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox,QMainWindow,QWidget,QMessageBox,QApplication,QMainWindow,QDesktopWidget,QHeaderView
import os
import sys
from utils.ui import Ui_mainWindow
from utils.main_new import gen
from utils.main_new import pre
import pathlib



class App(QMainWindow,Ui_mainWindow):

    def __init__(self):
        super(App, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Network-aware Persistent Homology")
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        self.setWindowFlag(Qt.WindowStaysOnTopHint,True)

        screen = QDesktopWidget().screenGeometry()
        # 获取窗口的尺寸信息
        size = self.geometry()
        # self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))
        self.move(int((screen.width() - size.width()) / 2), 10)
        self.temp_dir=None
        self.tensor_img=None
        self.processed_data = None


    @pyqtSlot()
    def on_upload_clicked(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv);;All Files (*)", options=options)
        
        if not file_path:
            QMessageBox.warning(self,'Path does not exist','Path does not exist')
            return
        
        try:
            data = pd.read_csv(file_path) 
            required_columns = ['Dst IP', 'Src IP', 'Timestamp']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"The CSV file must contain the following columns: {required_columns}")
            extracted_data = data[required_columns]  
            self.processed_data = pre(extracted_data)
            QMessageBox.information(self, "Success", "File uploaded successfully.")
            
        except Exception as e:
            print(e)
            QMessageBox.warning(self, 'Error', f'An error occurred: {str(e)}')
            return

    @pyqtSlot()
    def on_process_clicked(self):
        self.clean()
        if self.processed_data is None:
            QMessageBox.warning(self, 'Error', 'No data uploaded')
            return

        try:
            # 使用存储的 processed_data
            self.temp_dir,self.tensor_img = gen(self.processed_data)
            print('ssssss',self.tensor_img)
            self.show_result()


        except Exception as e:
            print(e)
            QMessageBox.warning(self, 'Processing error', 'An error occurred during processing')
            return

    @pyqtSlot()
    def on_cleanbutton_clicked(self):
        self.processed_data=None
        self.clean()

    def show_result(self):
        if self.tensor_img:
            html_content = "<b><font size='5'>Generated Representation:</font></b>"
            tensor_string = str(self.tensor_img)
            html_content += "<p>" + tensor_string + "</p>"
            self.textBrowser.setHtml(html_content)

        a_img_path=os.path.join(self.temp_dir,'a.png')
        if os.path.exists(a_img_path):
            pix_img = QPixmap(a_img_path)
            pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.A.setScaledContents(True)
            self.A.setPixmap(pix_img)

        b1_img_path=os.path.join(self.temp_dir,'b1.png')
        if os.path.exists(b1_img_path):
            pix_img = QPixmap(b1_img_path)
            pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.B1.setScaledContents(True)
            self.B1.setPixmap(pix_img)

        b2_img_path=os.path.join(self.temp_dir,'b2.png')
        if os.path.exists(b2_img_path):
            pix_img = QPixmap(b2_img_path)
            pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.B2.setScaledContents(True)
            self.B2.setPixmap(pix_img)

        c_img_path=os.path.join(self.temp_dir,'c.png')
        if os.path.exists(c_img_path):
            pix_img = QPixmap(c_img_path)
            pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.C.setScaledContents(True)
            self.C.setPixmap(pix_img)

        d_img_path=os.path.join(self.temp_dir,'d.png')
        if os.path.exists(d_img_path):
            pix_img = QPixmap(d_img_path)
            pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.D.setScaledContents(True)
            self.D.setPixmap(pix_img)

    def __del__(self):
        if self.temp_dir is not None:
            if os.path.exists(self.temp_dir):
                for file in pathlib.Path(self.temp_dir).glob("*"):
                    print(file)
                    os.remove(file)
                os.rmdir(self.temp_dir)
    def clean(self):
        pix_img = QPixmap()
        pix_img = pix_img.scaled(800,800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.A.setScaledContents(True)
        self.A.setPixmap(pix_img)
        self.B1.setScaledContents(True)
        self.B1.setPixmap(pix_img)
        self.B2.setScaledContents(True)
        self.B2.setPixmap(pix_img)
        self.C.setScaledContents(True)
        self.C.setPixmap(pix_img)
        self.D.setScaledContents(True)
        self.D.setPixmap(pix_img)

        self.tensor_img=None
        self.textBrowser.setText('')

        if self.temp_dir is not None:
            if os.path.exists(self.temp_dir):
                for file in pathlib.Path(self.temp_dir).glob("*"):
                    os.remove(file)
                os.rmdir(self.temp_dir)

        self.temp_dir=None



app=QApplication(sys.argv)
window=App()
window.show()
sys.exit(app.exec_())