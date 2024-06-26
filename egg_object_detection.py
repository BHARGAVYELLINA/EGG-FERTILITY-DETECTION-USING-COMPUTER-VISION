import cv2

def reScaleFrame(frame, percent=75):
    width = int(frame.shape[1] * percent // 100)
    height = int(frame.shape[0] * percent // 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

frame = cv2.imread('eggimg1.jpeg')
cv2.imshow("Original Image", frame)
cv2.waitKey(0)

frame50 = reScaleFrame(frame, percent=50)
cv2.imshow("Rescaled Image", frame50)
cv2.waitKey(0)
gray = cv2.cvtColor(frame50, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray)
cv2.waitKey(0)

_, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("Binary Image", bw)
cv2.waitKey(0)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_contour_area = cv2.contourArea(contours[0])
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

if filtered_contours:
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    cv2.rectangle(frame50, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("Detected Egg", frame50)
cv2.waitKey(0)
cv2.destroyAllWindows()