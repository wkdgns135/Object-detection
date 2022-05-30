from create_features import *
from training import *
from classify_data import *

def CreateFeatures(categorys):
    CLS = []
    for category in categorys:
        CLS.append([category, '101_ObjectCategories/'+category])

    codebook_file = "model/cb.pkl"
    feature_map_file = "model/fm.pkl"

    input_map = [] 
    for cls in CLS:
        assert len(cls) >= 2, "Format for classes is `<label> file`" 
        label = cls[0] 
        input_map += load_input_map(label, cls[1])
    
    # Building the codebook 
    print("===== Building codebook =====")
    kmeans, centroids = FeatureExtractor().get_centroids(input_map) 
    if codebook_file: 
        with open(codebook_file, 'wb') as f: 
            print('kmeans', kmeans)
            print('centroids', centroids)
            pickle.dump((kmeans, centroids), f)
 
    # Input data and labels 
    print("===== Building feature map =====")
    feature_map = extract_feature_map(input_map, kmeans, 
     centroids) 
    if  feature_map_file: 
        with open(feature_map_file, 'wb') as f: 
            pickle.dump(feature_map, f)

def Training():
    svm_file = "model/svm.pkl"
    feature_map_file = "model/fm.pkl"
 
    # Load the feature map 
    with open(feature_map_file, 'rb') as f: 
        feature_map = pickle.load(f) 
 
    # Extract feature vectors and the labels 
    labels_words = [x['label'] for x in feature_map] 
 
    # Here, 0 refers to the first element in the 
    # feature_map, and 1 refers to the second 
    # element in the shape vector of that element 
    # (which gives us the size) 
    dim_size = feature_map[0]['feature_vector'].shape[1] 
 
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map] 
    
    # Train the SVM 
    svm = ClassifierTrainer(X, labels_words) 

    if svm_file: 
        with open(svm_file, 'wb') as f: 
            pickle.dump(svm, f)

def ClassifyData(img):
    svm_file = "model/svm.pkl"
    codebook_file = "model/cb.pkl"
    input_image = cv2.imread(img) 
 
    tag = ImageClassifier(svm_file, codebook_file).getImageTag(input_image)
    # print("Output class:", tag)
    return tag

if __name__ == "__main__":
    categories = ["ant", "chair", "bass"]
    # If categories is change need restart this functions
    # CreateFeatures(categorys)
    # Training()
    
    for category in categories:
        fileName = os.listdir("101_ObjectCategories/" + category)

        score = 0
        for name in fileName:
            tag = ClassifyData("101_ObjectCategories/"+category+"/" +name)
            if(category == tag):
                score+=1
        print(category,"accuracy:",score / len(fileName))