from PIL import Image
import pytesseract
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# function to transform image to four points
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

#find largest contours method
def findLargestCountours(cntList, cntWidths):
    newCntList = []
    newCntWidths = []

    # finding 1st largest rectangle
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    # removing it from old
    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    # finding second largest rectangle
    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    # removing it from old
    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)

    return newCntList, newCntWidths

def convertToScan(image):
    #resize the image
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #apply bilateral filter to smooth the image, reduce noise, and preserve edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    #smooth the edges some with medialBlur
    gray = cv2.medianBlur(gray, 5)

    #detect the edges with canny algorithm
    edged = cv2.Canny(gray, 30, 400)

    # find contours in the edged image, keep only the largest ones, and initialize our screen contour
    countours, hierarcy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    imageCopy = image.copy()
    # approximate the contour

    #sort them in reverse order so the largest contours are in the front
    cnts = sorted(countours, key=cv2.contourArea, reverse=True)

    #create lists of the contours
    screenCntList = []
    scrWidths = []
    for cnt in cnts:
        #specifies that it is a closed shape
        peri = cv2.arcLength(cnt, True) 
        # you want square but you got bad one so you need to approximate
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        screenCnt = approx
        if len(screenCnt) == 4:
            #determine the width and length of the rectangle
            (X, Y, W, H) = cv2.boundingRect(cnt)
            screenCntList.append(screenCnt)
            scrWidths.append(W)

    #find the largest contours largest should be our document
    screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)

    if not len(screenCntList) >= 2:  # there is no rectangle found
        print("No rectangle found")
    elif scrWidths[0] != scrWidths[1]:  # mismatch in rect
        print("Mismatch in rectangle")

    #reshape to 4*2 so that 4 points each one has x,y
    pts = screenCntList[0].reshape(4, 2)

    #define our rectangle 
    rect = order_points(pts)

    #resize back to original size
    warped = four_point_transform(orig, screenCntList[0].reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #define threshold to get B&W images
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    cv2.imshow("Original", image)
    imutils.resize(warped, height = 650)
    cv2.imshow("warp", warped )
    cv2.waitKey(0)
    cv2.imwrite('scanned.jpg', warped)

    return warped

def convertToPDF(doc):
    print(pytesseract.image_to_string(doc))

    #Get a searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(doc, extension='pdf')
    with open('test.pdf', 'w+b') as f:
        f.write(pdf)

#sets executable path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
while True:
    pathToImage = input("Please enter path to document: ")
    try:
        image = cv2.imread(pathToImage)
    except Exception:
        print("invalid path")
        continue
    while True:
        scanOrImage = input("please enter whether your document is a scan or image")

        if scanOrImage.lower() == "scan":
            convertToPDF(image)
            break
        elif scanOrImage.lower() == "image":
            doc = convertToScan(image)
            convertToPDF(doc)
            break
    break