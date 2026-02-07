import cv2, ultralytics
img=cv2.imread("photos/drone img1.jpg")
cv2.imshow("pic", img)
cv2.waitKey(0)

capture=cv2.VideoCapture("videos/Berghouse Leopard Jog.mp4")
while True:
    isTrue, frame=capture.read()
    cv2.imshow("vid", frame)
    if (cv2.waitKey(20) and 0xFF==ord("d")):
        break

capture.release()
cv2.destroyAllWindows()
