import cv2
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("test.jpg")
# DISPLAY
cv2.imshow("test",img)
cv2.waitKey(0)