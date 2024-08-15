from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import cv2
from imutils import paths

# Capture image from webcam
width, height = 640, 360
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Create MTCNN instance for extracting faces
mtcnn = MTCNN(keep_all=True)

# Directory with images
path = r"C:\Users\Hp\OneDrive\Desktop\faces"
imagePaths = list(paths.list_images(path)) #extract the imagepaths




# Process each image and extract face embeddings
def encode_face(imagePaths):
    embedding_list = [] 
    for imagepath in imagePaths:
        image = cv2.imread(imagepath) #read the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to rgb from bgr
        
        faces, conf = mtcnn.detect(image) #detect faces 
        if faces is not None:
            aligned_faces = mtcnn(image)
            embeddings = resnet(aligned_faces).detach().numpy() #get embeddings
            embedding_list.append((imagepath, embeddings)) # append embeddings to list

    # Convert embedding list to numpy array for easier manipulation
    embedding_array = np.vstack([emb for _, emb in embedding_list])
    return embedding_array,embedding_list


while True:
    success, frame = cam.read()
    if not success:
        break
    cv2.imshow('mywindow', frame)

    # Press 'q' to capture and process the frame
    if cv2.waitKey(1) == ord('q'):
        photo = frame
        break

cam.release()
cv2.destroyAllWindows()

# Process captured frame
image = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
faces, confidence = mtcnn.detect(image)

if faces is not None:
    aligned_faces = mtcnn(image)
    embeddings = resnet(aligned_faces).detach().numpy()
    embedding_array,embedding_list = encode_face(imagePaths)
    data = []
    
    # Calculate distances between the captured face and stored embeddings
    for emb in embeddings:
        dist = euclidean_distances(embedding_array, [emb]).flatten()
        
        # Find indices where distance is less than 0.8
        close_indices = np.where(dist < 0.8)[0]
        
        # Append corresponding image paths to data
        for idx in close_indices:
            data.append(embedding_list[idx][0])

    # Display the matched images
    data = np.array(data)
    
    
    if len(data) == 0:
        print("No matching faces found.")
    else:
        for img_path in data:
            match_image = cv2.imread(img_path)
            cv2.imshow('Matching Face', match_image)
            if cv2.waitKey(0) == ord('q'):
                print(data)
                break
        cv2.destroyAllWindows()

else:
    print("No face detected in the captured image.")

