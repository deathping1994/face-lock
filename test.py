import numpy
import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.createLBPHFaceRecognizer()
video_capture = cv2.VideoCapture(1)
labels=[]
images=[]

subjects = input("Enter number of subjects")
subjects = int(subjects)
while subjects:
	for i in range(1,30):
	    # Capture frame-by-frame
	    ret, frame = video_capture.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	    )

	    # Draw a rectangle around the faces
	    for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		images.append(gray[y: y + h, x: x + w])
		labels.append(subjects)
	    # Display the resulting frame
	    cv2.imshow('Training', frame)

	    if cv2.waitKey(5) & 0xFF == ord('q'):
		break
	cv2.destroyAllWindows()
	print "Change subject... "
	y=input("press 1 when ready")
	y = int(y)
	if (y!=1):
		break
	subjects = subjects - 1

print "Training complete, saving data"
recognizer.train(images, numpy.array(labels))
recognizer.save("gaurav.yml")
print "Training Data saved ..."
print "Now run predict.py"

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

