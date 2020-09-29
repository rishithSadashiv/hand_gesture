import cv2
import numpy as np
import math

hand_cascade = cv2.CascadeClassifier('training/data/hand_cascade_stage16.xml') # cascade trained for images of size (20x20) 1400 pos and 700 neg
palm_cascade = cv2.CascadeClassifier('palm_haar_cascade1.xml')
fist_cascade = cv2.CascadeClassifier('fist_haar_cascade1.xml')


cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)
    hands = hand_cascade.detectMultiScale(gray,6,6) # DETECTING hand IN gray IMAGE
    # This detect multiscale is found to work well with values 2.5,3 or 6,6
    count_defects = 0
#     hands = hand_cascade.detectMultiScale(gray, 11, 11) 
    for (x,y,w,h) in hands:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 0)
        roi_colour = img[y:y+h+10, x:x+w+10]
        roi = gray[y:y+h+10, x:x+w+10]
        cv2.imshow('roi', roi)
        
        # blurring the roi to remove noise
        value = (11, 11)
        blurred = cv2.GaussianBlur(roi, value, 0)
        cv2.imshow('blurred', blurred)
        
        # thresholdin: Otsu's Binarization method
        _, thresh1 = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        cv2.imshow('thresh', thresh1)
        
        # check OpenCV version to avoid unpacking error
        (version, _, _) = cv2.__version__.split('.')
        if version == '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version == '2' or version == '4':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                   cv2.CHAIN_APPROX_NONE)
            
         # find contour with max area
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        # create bounding rectangle around the contour (can skip below two lines)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.imshow('rec', cv2.rectangle(roi_colour, (x, y), (x+w, y+h), (0, 0, 255), 0))
        
        # finding convex hull
        hull = cv2.convexHull(cnt)

        # drawing contours
        drawing = np.zeros(roi_colour.shape,np.uint8)
        cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

        # finding convex hull
        hull = cv2.convexHull(cnt, returnPoints=False)
        
        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
        
        
        # applying Cosine Rule to find angle for all defects (between fingers)
        # with angle > 90 degrees and ignore defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]

            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
            # ignore angles > 90 and highlight rest with red dots
            if angle <= 90:
                count_defects += 1
                cv2.circle(roi_colour, far, 1, [0,0,255], -1)
            #dist = cv2.pointPolygonTest(cnt,far,True)
        
            # draw a line from start to end i.e. the convex points (finger tips)
            # (can skip this part)
            cv2.line(roi_colour,start, end, [0,255,0], 2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)
            
            
            # define actions required
        if count_defects == 1:
            cv2.putText(img,"This is two", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            #print('This is two')
        elif count_defects == 2:
            str = "This is three"
            cv2.putText(img, str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            #print('This is three')
        elif count_defects == 3:
            cv2.putText(img,"This is  four", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            #print('This is four')
        elif count_defects == 4:
            cv2.putText(img,"This is five", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            #print('This is five')
        else:
            palm = palm_cascade.detectMultiScale(img, 1.3, 5) # DETECTING PALM IN THE THRESHOLD IMAGE
            fist = fist_cascade.detectMultiScale(img, 1.3, 5) # DETECTING FIST IN THE THRESHOLD IMAGE
            if len(palm) and len(fist):
                cv2.putText(img,"Both palm and fist are present!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            elif len(palm):
                cv2.putText(img,"This is palm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            elif len(fist):
                cv2.putText(img,"This is fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)
            else:
                cv2.putText(img,"This is one", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,cv2.LINE_AA)

        # show appropriate images in windows
#         cv2.imshow('Gesture', img)
#         all_img = np.hstack((drawing, roi_colour))
        cv2.imshow('Contours', drawing)
        cv2.imshow('roi', roi_colour)
        

    
    
    cv2.imshow('img', img )
    
    
    k = cv2.waitKey(10)
    if k == 27:
        cap.release()
        break
        
cv2.destroyAllWindows()