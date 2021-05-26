import cv2
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No folder specified")
        exit()
    folderName = sys.argv[1]

    _, _, filenames = next(os.walk(folderName))

    for f in filenames:
        path = os.path.join(folderName, f)
        if os.path.isfile(path):
            image = cv2.imread(path)

            image90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image180 = cv2.rotate(image90, cv2.ROTATE_90_CLOCKWISE)
            image270 = cv2.rotate(image180, cv2.ROTATE_90_CLOCKWISE)

            name90 = f[:-4] + "_90.jpg"
            name180 = f[:-4] + "_180.jpg"
            name270 = f[:-4] + "_270.jpg"

            cv2.imwrite(os.path.join(folderName, name90), image90)
            cv2.imwrite(os.path.join(folderName, name180), image180)
            cv2.imwrite(os.path.join(folderName, name270), image270)

            print(f, "done")

        else:
            print("File", path, "does not exist")


    
