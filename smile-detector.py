from tkinter import Frame
import cv2

#face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

#grab webcam feed
webcam = cv2.VideoCapture(0)
while True:

    #read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    #change frame to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)



    #run face detection for all faces
    for (x, y, w, h) in faces:
        #draw rectangle around faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        #get the subframe (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        #change just the_face to greyscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #detect smiles
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        #detect eyes
        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.3, minNeighbors=10)

        #find all smiles in the face
        for (x_, y_, w_, h_) in smiles:

            #draw all the rectangles around the smile on the_face (not the entire frame)
            cv2.rectangle(the_face, (x_, y_), (x_+ w_, y_+ h_), (50, 50, 200), 4)

        #find all eyes in the face
        for (x_, y_, w_, h_) in eyes:

            #draw all the rectangles around the smile on the_face (not the entire frame)
            cv2.rectangle(the_face, (x_, y_), (x_+ w_, y_+ h_), (255, 255, 255), 4)
        
        
        #label face as smiling
        if len(smiles) > 0:
            cv2.putText(frame,'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
            
        
    #show the current frame
    cv2.imshow("Smile Detector", frame)

    key = cv2.waitKey(1)
    # press q or Q to quit
    if key == 81 or key == 113:
        break

#clean up
webcam.release()
cv2.destroyAllWindows

print("code completed")