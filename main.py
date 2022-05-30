from objectDetection import *
from create_features import *
from training import *
from classify_data import *

def CreateFeatures():
    CLS = [['chair','101_ObjectCategories/chair'],['ant','101_ObjectCategories/ant']]
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
        # args = build_arg_parser().parse_args() 

    svm_file = "model/svm.pkl"
    feature_map_file = "model/fm.pkl"

    # feature_map_file = args.feature_map_file 
    # svm_file = args.svm_file 
 
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

    # if args.svm_file: 
    #     with open(args.svm_file, 'wb') as f: 
    #         pickle.dump(svm, f) 

    if svm_file: 
        with open(svm_file, 'wb') as f: 
            pickle.dump(svm, f)

def ClassifyData():
    # args = build_arg_parser().parse_args() 
    # svm_file = args.svm_file 
    # codebook_file = args.codebook_file 
    svm_file = "model/svm.pkl"
    codebook_file = "model/cb.pkl"
    input_image_file = "101_ObjectCategories/ant/image_0041.jpg"
    input_image = cv2.imread(input_image_file) 
 
    tag = ImageClassifier(svm_file, codebook_file).getImageTag(input_image)
    print("Output class:", tag)

if __name__ == "__main__":
    