# importing the module
import cv2
import numpy as np


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        a = '(' + str(x) + ', ' + str(y) + ')'
        with open(r'./lane_coords.txt', 'a') as f:
            f.write(a)
        f.close()

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)


    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)


# driver function
if __name__ == "__main__":
    # reading the image
    img = cv2.imread('../inputs/frames/frame0.png', 1)

    pts = np.array([[55, 272], [277, 136], [284, 139], [100, 293]], np.int32)
    pts2 = np.array([[284, 139], [100, 293], [170, 302], [297, 142]], np.int32)
    pts3 = np.array([[170, 302], [297, 142], [312, 142], [265, 307]], np.int32)
    pts4 = np.array([[380, 307], [331, 140], [345, 140], [467, 308]], np.int32)
    pts5 = np.array([[345, 140], [467, 308], [543, 303], [358, 140]], np.int32)
    pts6 = np.array([[358, 140], [543, 303], [602, 290], [371, 139]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    pts2 = pts2.reshape((-1, 1, 2))
    pts3 = pts3.reshape((-1, 1, 2))
    pts4 = pts4.reshape((-1, 1, 2))
    pts5 = pts5.reshape((-1, 1, 2))
    pts6 = pts6.reshape((-1, 1, 2))
    parallel = cv2.polylines(img, [pts], True, (0, 255, 255))
    parallel2 = cv2.polylines(img, [pts2], True, (0, 255, 255))
    parallel3 = cv2.polylines(img, [pts3], True, (0, 255, 255))
    parallel4 = cv2.polylines(img, [pts4], True, (0, 255, 255))
    parallel5 = cv2.polylines(img, [pts5], True, (0, 255, 255))
    parallel6 = cv2.polylines(img, [pts6], True, (0, 255, 255))

    # displaying the image
    # cv2.imshow('image', parallel)
    # cv2.imshow('image', parallel2)
    # cv2.imshow('image', parallel3)
    # cv2.imshow('image', parallel4)
    # cv2.imshow('image', parallel5)
    # cv2.imshow('image', parallel6)
    cv2.imshow('image', img)
    # cv2.imshow('image_crop', img[272:293, 55:284])

    # setting mouse hadler for the image
    # and calling the click_event() function
    # cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
