import cv2

def resizeImage(path):
    ogimg = cv2.imread(path)
    width = float(ogimg.shape[1])
    height = float(ogimg.shape[0])
    
    pivot = max(width, height)
    ratio = float(512/pivot)
    new_width = int(width*ratio)
    new_height = int(height*ratio)
    ogimg = cv2.resize(ogimg, (new_width, new_height), interpolation=cv2.INTER_AREA)

    top = (512 - new_height) // 2
    bottom = 512 - new_height - top
    left = (512 - new_width) // 2
    right = 512 - new_width - left
    borderedImg= cv2.copyMakeBorder(ogimg,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[255, 255, 255])
    cv2.imwrite('512'+path, borderedImg)