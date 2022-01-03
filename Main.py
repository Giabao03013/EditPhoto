

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2 , imutils
import sys
import numpy as np

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(588, 452)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 10, 581, 311))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 10, 581, 311))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.Slider1 = QtWidgets.QSlider(self.centralwidget)
        self.Slider1.setGeometry(QtCore.QRect(90, 350, 391, 22))
        self.Slider1.setOrientation(QtCore.Qt.Horizontal)
        self.Slider1.setObjectName("Slider1")
        self.Slider1.setMinimum(0)
        self.Slider1.setMaximum(100)
        self.Slider1.setValue(0)
        self.Slider1.setTickPosition(QSlider.TicksBelow)
        self.Slider1.setSingleStep(1)
        self.Slider1.hide()
        self.Slider2 = QtWidgets.QSlider(self.centralwidget)
        self.Slider2.setGeometry(QtCore.QRect(90, 390, 391, 22))
        self.Slider2.setOrientation(QtCore.Qt.Horizontal)
        self.Slider2.setObjectName("Slider2")
        self.Slider2.setMinimum(0)
        self.Slider2.setMaximum(50)
        self.Slider2.setValue(0)
        self.Slider2.setTickPosition(QSlider.TicksBelow)
        self.Slider2.setSingleStep(1)
        self.Slider2.hide()
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(490, 350, 47, 13))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(490, 390, 47, 13))
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 575, 21))
        self.menubar.setObjectName("menubar")
        self.menuFILE = QtWidgets.QMenu(self.menubar)
        self.menuFILE.setObjectName("menuFILE")
        self.menuBright_Contrast = QtWidgets.QMenu(self.menubar)
        self.menuBright_Contrast.setObjectName("menuBright_Contrast")
        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        self.menuRotate = QtWidgets.QMenu(self.menubar)
        self.menuRotate.setObjectName("menuRotate")
        self.menuCrop = QtWidgets.QMenu(self.menubar)
        self.menuCrop.setObjectName("menuCrop")
        self.menuPanorama = QtWidgets.QMenu(self.menubar)
        self.menuPanorama.setObjectName("menuPanorama")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.Brightness = QtWidgets.QAction(MainWindow)
        self.Brightness.setObjectName("Brightness")
        self.Contrast = QtWidgets.QAction(MainWindow)
        self.Contrast.setObjectName("Contrast")
        self.Open = QtWidgets.QAction(MainWindow)
        self.Open.setObjectName("Open")
        self.Save = QtWidgets.QAction(MainWindow)
        self.Save.setObjectName("Save")
        self.Exit = QtWidgets.QAction(MainWindow)
        self.Exit.setObjectName("Exit")
        self.Gray = QtWidgets.QAction(MainWindow)
        self.Gray.setObjectName("Gray")
        self.Black_White = QtWidgets.QAction(MainWindow)
        self.Black_White.setObjectName("Black_White")
        self.Blur = QtWidgets.QAction(MainWindow)
        self.Blur.setObjectName("Blur")
        self.Cartoon = QtWidgets.QAction(MainWindow)
        self.Cartoon.setObjectName("Cartoon")
        self.Pencil = QtWidgets.QAction(MainWindow)
        self.Pencil.setObjectName("Pencil")
        self.Undo = QtWidgets.QAction(MainWindow)
        self.Undo.setObjectName("Undo")
        self.Left = QtWidgets.QAction(MainWindow)
        self.Left.setObjectName("Left")
        self.Right = QtWidgets.QAction(MainWindow)
        self.Right.setObjectName("Right")
        self.Flip = QtWidgets.QAction(MainWindow)
        self.Flip.setObjectName("Flip")
        self.crop = QtWidgets.QAction(MainWindow)
        self.crop.setObjectName("Crop")
        self.create=QtWidgets.QAction(MainWindow)
        self.create.setObjectName("Create")
        self.menuFILE.addAction(self.Open)
        self.menuFILE.addAction(self.Save)
        self.menuFILE.addAction(self.Exit)
        self.menuBright_Contrast.addAction(self.Brightness)
        self.menuBright_Contrast.addAction(self.Contrast)
        self.menuFilter.addAction(self.Gray)
        self.menuFilter.addAction(self.Black_White)
        self.menuFilter.addAction(self.Blur)
        self.menuFilter.addAction(self.Cartoon)
        self.menuFilter.addAction(self.Pencil)
        self.menuFilter.addAction(self.Undo)
        self.menuRotate.addAction(self.Left)
        self.menuRotate.addAction(self.Right)
        self.menuRotate.addAction(self.Flip)
        self.menuCrop.addAction(self.crop)
        self.menuPanorama.addAction(self.create)
        self.menubar.addAction(self.menuFILE.menuAction())
        self.menubar.addAction(self.menuBright_Contrast.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menubar.addAction(self.menuRotate.menuAction())
        self.menubar.addAction(self.menuCrop.menuAction())
        self.menubar.addAction(self.menuPanorama.menuAction())

        self.Slider1.valueChanged.connect(self.change_bright)
        self.Slider2.valueChanged.connect(self.change_contrast)
        self.retranslateUi(MainWindow)
        self.Brightness.triggered.connect(self.Slider1.show)
        self.Contrast.triggered.connect(self.Slider2.show)
        self.Open.triggered.connect(self.open)
        self.Save.triggered.connect(self.save)
        self.Exit.triggered.connect(QApplication.instance().quit)
        self.Gray.triggered.connect(self.gray)
        self.Blur.triggered.connect(self.blur)
        self.Black_White.triggered.connect(self.bw)
        self.Pencil.triggered.connect(self.pencil)
        self.Cartoon.triggered.connect(self.cartoon)
        self.Undo.triggered.connect(self.undo)
        self.Left.triggered.connect(self.left)
        self.Right.triggered.connect(self.right)
        self.Flip.triggered.connect(self.flip)
        self.crop.triggered.connect(self.crop_image)
        self.create.triggered.connect(self.panorama)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        


        #value
        self.filename = None   #File path image
        self.tmp = None 
        self.coords=[]
        self.drawing = False
        self.image_now=None
        self.bias = 0 #Value change bright
        self.gain = 0 #Value change contrast 

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Ứng dụng chỉnh sửa ảnh đơn giản"))
        self.label.setText(_translate("MainWindow", ""))
        self.menuFILE.setTitle(_translate("MainWindow", "File"))
        self.menuBright_Contrast.setTitle(_translate("MainWindow", "Adjust"))
        self.menuFilter.setTitle(_translate("MainWindow", "Filter"))
        self.menuRotate.setTitle(_translate("MainWindow", "Rotate"))
        self.menuCrop.setTitle(_translate("MainWindow", "Crop"))
        self.menuPanorama.setTitle(_translate("MainWindow", "Panorama"))
        self.Brightness.setText(_translate("MainWindow", "Brightness"))
        self.Contrast.setText(_translate("MainWindow", "Contrast"))
        self.Open.setText(_translate("MainWindow", "Open"))
        self.Save.setText(_translate("MainWindow", "Save"))
        self.Exit.setText(_translate("MainWindow", "Exit"))
        self.Gray.setText(_translate("MainWindow", "Gray"))
        self.Black_White.setText(_translate("MainWindow", "BlackWhite"))
        self.Blur.setText(_translate("MainWindow", "Blur"))
        self.Cartoon.setText(_translate("MainWindow", "Cartoon"))
        self.Pencil.setText(_translate("MainWindow", "Pencil"))
        self.Undo.setText(_translate("MainWindow","Reset"))
        self.Left.setText(_translate("MainWindow", "Left"))
        self.Right.setText(_translate("MainWindow", "Right"))
        self.crop.setText(_translate("MainWindow","Crop"))
        self.Flip.setText(_translate("MainWindow","Flip"))
        self.create.setText(_translate("MainWindow","Create"))


    #Open file
    def open(self):
        self.filename = QFileDialog.getOpenFileName(None,'Open file','E:','All File(*.*);;JPG(*.jpg);;PNG(*.png)')[0]
        self.image = cv2.imread(self.filename)
        self.image_now=self.image
        self.setPhoto(self.image)

    #Save file
    def save(self):
        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        cv2.imwrite(filename,self.tmp)
    
    


    #Set photo to label
    def setPhoto(self,image):
        self.tmp = image
        image = imutils.resize(image,width=581,height=311)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    #Change Bright
    def change_bright(self,value):
        
        #if self.image_now is None:
         #   if self.bias>value:
         #       self.bias=value
         #       image=cv2.convertScaleAbs(self.image,alpha=1,beta=-self.bias)
                
          #      self.setPhoto(image)
          #  elif self.bias<value:
          #      self.bias=value
          #      image=cv2.convertScaleAbs(self.image_now,alpha=1,beta=self.bias)
                
         #       self.setPhoto(image)
       # else:
         #   if self.bias>value:
         #       self.bias=value
         #       image=cv2.convertScaleAbs(self.image_now,alpha=1,beta=-self.bias)
         #       self.setPhoto(image)
                
           # elif self.bias<value:
           #     self.bias=value
          #      image=cv2.convertScaleAbs(self.image_now,alpha=1,beta=self.bias)
                #self.setPhoto(image)
        #self.bias=value
        #image=cv2.convertScaleAbs(self.image_now,alpha=1,beta=self.bias)
        #self.label_3.setText(str(value))
        #self.setPhoto(image)
        #hsv = cv2.cvtColor(self.image_now,cv2.COLOR_BGR2HSV)
        #h,s,v = cv2.split(hsv)
        #lim = 255 - value
        #v[v>lim] = 255
        #v[v<=lim] += value
        #final_hsv = cv2.merge((h,s,v))
        #img = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
        self.label_3.setText(str(value))
        self.bias=value
        image=cv2.convertScaleAbs(self.image_now,alpha=1,beta=self.bias)
        self.setPhoto(image)
    #Change contrast
    def change_contrast(self,value):
        self.label_4.setText(str(value))
        self.gain = float(value)
        image=cv2.convertScaleAbs(self.image_now,alpha=self.gain,beta=0)
        self.setPhoto(image)

    #Filter gray  
    def gray(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        gray_image=cv2.cvtColor(self.image_now,cv2.COLOR_RGB2GRAY)
        self.image_now=gray_image
        self.setPhoto(gray_image)

    #Filter blur
    def blur(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        image = cv2.blur(self.image_now,(5,5))
        self.image_now=image
        self.setPhoto(image)

    #Filter black&white
    def bw(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        if len(self.image_now.shape)==2:
            (thresh,bw)=cv2.threshold(self.image_now,127,255,cv2.THRESH_BINARY)
            self.image_now=bw
            self.setPhoto(bw)
        elif len(self.image_now.shape)==3:
            gray_image=cv2.cvtColor(self.image_now,cv2.COLOR_RGB2GRAY)
            (thresh,bw)=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
            self.image_now=bw
            self.setPhoto(bw)

    #Filter cartoon
    def cartoon(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        if len(self.image_now.shape)==2:
            gray = cv2.medianBlur(self.image_now,3)
            edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9, 9)
            color = cv2.bilateralFilter(self.image_now, 9, 250, 250)
            cartoon_image = cv2.bitwise_and(color, color, mask=edges)
            self.image_now=cartoon_image
            self.setPhoto(cartoon_image)
        elif len(self.image_now.shape)==3:
            gray = cv2.cvtColor(self.image_now,cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray,3)
            edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9, 9)
            color = cv2.bilateralFilter(self.image_now, 9, 250, 250)
            cartoon_image = cv2.bitwise_and(color, color, mask=edges)
            self.image_now=cartoon_image
            self.setPhoto(cartoon_image)

    #Filter pencil
    def pencil(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        if len(self.image_now.shape)==2:
            img_invert = cv2.bitwise_not(self.image_now)
            #cv2.imshow("",img_invert)
            blur=cv2.GaussianBlur(img_invert,(21,21),0)
            #cv2.imshow("",blur)
            invert_blur=255-blur
            pencil_image=cv2.divide(self.image_now,invert_blur,scale=256.0)
            self.image_now=pencil_image
            self.setPhoto(pencil_image)
        elif len(self.image_now.shape)==3:
            gray=cv2.cvtColor(self.image_now,cv2.COLOR_BGR2GRAY)
            img_invert = cv2.bitwise_not(gray)
            #cv2.imshow("",img_invert)
            blur=cv2.GaussianBlur(img_invert,(21,21,),0)
            #cv2.imshow("blur",blur)
            invert_blur=255-blur
            #cv2.imshow("invert_blur",invert_blur)
            pencil_image=cv2.divide(gray,invert_blur,scale=256.0)
            self.image_now=pencil_image
            self.setPhoto(pencil_image)

    def undo(self):
        self.label_3.setText("")
        self.label_4.setText("")
        image=cv2.imread(self.filename)
        self.image_now=image
        self.setPhoto(image)
        
    #Rotate left
    def left(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        rotate_image=cv2.rotate(self.image_now,cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.image_now=rotate_image
        self.setPhoto(rotate_image)
    #Rotate right
    def right(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        rotate_image=cv2.rotate(self.image_now,cv2.ROTATE_90_CLOCKWISE)
        self.image_now=rotate_image
        self.setPhoto(rotate_image)
    #Rotate flip
    def flip(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        rotate_image=cv2.rotate(self.image_now,cv2.ROTATE_180)
        self.image_now=rotate_image
        self.setPhoto(rotate_image)

    def crop_image(self):
        self.label_3.setText("")
        self.label_4.setText("")
        self.Slider1.hide()
        self.Slider2.hide()
        cv2.imshow("Crop",self.image_now)
        cv2.setMouseCallback('Crop',self.click_and_crop,self.image_now)
        cv2.waitKey(0)

        

    def click_and_crop(self,event,x,y,flag,image):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.coords=[(x,y)]
            print(self.coords)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing is True:
                clone = image.copy()
                cv2.rectangle(clone,self.coords[0],(x,y),(0,255,0),2)
        elif event ==  cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.coords.append((x,y))
            if len(self.coords)==2:
                ty, by, tx, bx =  self.coords[0][1], self.coords[1][1], self.coords[0][0], self.coords[1][0]
                roi = image[ty:by, tx:bx]
                height, width = roi.shape[:2]
                if width > 0 and height > 0:
                    self.image_now=roi
                    self.setPhoto(roi)
    #create panorama
    def panorama(self):
        self.label_3.setText("")
        self.label_4.setText("")
        image1 = cv2.imread("Images/1/1.jpg")
        img1 = cv2.resize(image1,(500,400))
        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        image2 = cv2.imread("Images/1/2.jpg")
        img2 = cv2.resize(image2,(500,400))
        img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        image3 = cv2.imread("Images/1/3.jpg")
        img3 = cv2.resize(image3,(500,400))
        img3_gray=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
        keypoints3, descriptors3 = orb.detectAndCompute(img3, None)
        bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descriptors1, descriptors2,k=2)
        # Ve cac diem chinh giong nhau giua 2 anh
        #def draw_matches(img1, keypoints1, img2, keypoints2, matches):
          #r, c = img1.shape[:2]
          #r1, c1 = img2.shape[:2]
          # Create a blank image with the size of the first image + second image
          #output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
          #output_img[:r, :c, :] = np.dstack([img1, img1, img1])
          #output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])
          # Go over all of the matching points and extract them
          #for match in matches:
           # img1_idx = match.queryIdx
            #img2_idx = match.trainIdx
            #(x1, y1) = keypoints1[img1_idx].pt
            #(x2, y2) = keypoints2[img2_idx].pt
            # Draw circles on the keypoints
           # cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
            #cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)
            # Connect the same keypoints
            #cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
          #return output_img

        #all_matches = []
        #for m, n in matches:
            #all_matches.append(m)
        #img12 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
        #cv2.imshow("Draw matches",img12)
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)

        def warpImages(img1, img2, H):
            rows1, cols1 = img1.shape[:2]
            rows2, cols2 = img2.shape[:2]
            list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
            temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
            list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
            list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)
            [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
            translation_dist = [-x_min,-y_min]
            H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
            output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
            output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
            return output_img


        MIN_MATCH_COUNT = 10

        if len(good) > MIN_MATCH_COUNT:
            # Lấy các keypoint từ img1 và img2 để làm tham số tìm ma trận đồng nhất
            src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            # Tìm ma trận đồng nhất
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            img1_2 = warpImages(img2, img1, M)

           
        
        img1_2_gray = cv2.cvtColor(img1_2,cv2.COLOR_BGR2GRAY)
        keypoints1_2, descriptors1_2 = orb.detectAndCompute(img1_2, None)
        matches1 = bf.knnMatch(descriptors1_2, descriptors3,k=2)
        #all_matches_1=[]
        #for m, n in matches1:
            #all_matches_1.append(m)
        #img4 = draw_matches(img1_2_gray, keypoints1_2, img3_gray, keypoints3, all_matches_1[:30])
        #cv2.imshow("Draw_matches",img4)

        good_1 = []
        for m, n in matches1:
            if m.distance < 0.6 * n.distance:
                good_1.append(m)

        if len(good_1) > MIN_MATCH_COUNT:
            src_pts = np.float32([ keypoints1_2[m.queryIdx].pt for m in good_1]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints3[m.trainIdx].pt for m in good_1]).reshape(-1,1,2)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            panorama = warpImages(img3, img1_2, M)
            cv2.imshow("Panorama",panorama)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
