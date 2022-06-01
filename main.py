from obejct_classifier import *

if __name__ == "__main__":
    categories = ["ant", "chair", "bass"]
    oc = object_classifier(categories)
    oc.fit()

    for category in categories:
        fileName = os.listdir("101_ObjectCategories/" + category)

        score = 0
        for name in fileName:
            img = cv2.imread("101_ObjectCategories/"+category+"/" +name)
            tag = oc.predict(img)
            if(category == tag):
                score+=1
        print(category,"accuracy:",score / len(fileName))