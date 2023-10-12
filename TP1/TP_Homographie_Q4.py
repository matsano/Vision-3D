import numpy as np
import cv2

print("Version d'OpenCV: ",cv2. __version__)

# Ouverture de l'image
PATH_IMG = './Images_Homographie/'

img1 = np.uint8(cv2.imread(PATH_IMG+"Amst-3.jpg"))
img2 = np.uint8(cv2.imread(PATH_IMG+"Amst-2.jpg"))

(h1,w1,c1) = img1.shape
print("Dimension de l'image :",h1,"lignes x",w1,"colonnes x",c1,"couleurs")

(h2,w2,c2) = img2.shape
print("Dimension de l'image :",h2,"lignes x",w2,"colonnes x",c2,"couleurs")

# Selectionner les points de l'image 1
def select_points1(event, x, y, flags, param):
	global points_selected1, X_init
	global img1, clone1
	if (event == cv2.EVENT_FLAG_LBUTTON):
		x_select,y_select = x,y
		points_selected1 += 1
		cv2.circle(img1,(x_select,y_select),8,(0,255,255),1)
		cv2.line(img1,(x_select-8,y_select),(x_select+8,y_select),(0,255,0),1)
		cv2.line(img1,(x_select,y_select-8),(x_select,y_select+8),(0,255,0),1)
		X_init.append( [x_select,y_select] )
	elif event == cv2.EVENT_FLAG_RBUTTON:
		points_selected1 = 0
		img1 = clone1.copy()
		
clone1 = img1.copy()
points_selected1 = 0
X_init = []
cv2.namedWindow("Image initiale 1")
cv2.setMouseCallback("Image initiale 1",select_points1)

while True:
	cv2.imshow("Image initiale 1",img1)
	key = cv2.waitKey(1) & 0xFF
	if (key == ord("q")) & (points_selected1 >= 4):
		break

# Selectionner les points de l'image 2
def select_points2(event, x, y, flags, param):
	global points_selected2, X_final
	global img2, clone2
	if (event == cv2.EVENT_FLAG_LBUTTON):
		x_select,y_select = x,y
		points_selected2 += 1
		cv2.circle(img2,(x_select,y_select),8,(0,255,255),1)
		cv2.line(img2,(x_select-8,y_select),(x_select+8,y_select),(0,255,0),1)
		cv2.line(img2,(x_select,y_select-8),(x_select,y_select+8),(0,255,0),1)
		X_final.append( [x_select,y_select] )
	elif event == cv2.EVENT_FLAG_RBUTTON:
		points_selected2 = 0
		img2 = clone2.copy()
		
clone2 = img2.copy()
points_selected2 = 0
X_final = []
cv2.namedWindow("Image initiale 2")
cv2.setMouseCallback("Image initiale 2",select_points2)

while True:
	cv2.imshow("Image initiale 2",img2)
	key = cv2.waitKey(1) & 0xFF
	if (key == ord("q")) & (points_selected2 >= 4):
		break
		
# Conversion en array numpy
X_init = np.asarray(X_init,dtype = np.float32) 		
print("X_init =",X_init)

X_final = np.asarray(X_final, dtype = np.float32) 		
print("X_final =",X_final)

if (points_selected1 != points_selected2):
	raise ValueError("Different number of points selected in the images.")


#### Votre code d'estimation de H ici

# Coordonnees au format (x, y, 1)
X_init_norm = np.hstack((X_init, np.ones((X_init.shape[0], 1))))
X_final_norm = np.hstack((X_final, np.ones((X_final.shape[0], 1))))

# Normalisation des coordonnees
T_norm = np.array([[1/w1, 0, -1],
                  [0, 1/h1, -1],
                  [0, 0, 1]])
X_init_norm = np.dot(T_norm, np.transpose(X_init_norm))
X_init_norm = np.transpose(X_init_norm)

X_final_norm = np.dot(T_norm, np.transpose(X_final_norm))
X_final_norm = np.transpose(X_final_norm)

# Retourner les coordonnees au format (x, y)
X_init_norm = X_init_norm[:, :-1]
X_final_norm = X_final_norm[:, :-1]

# Calculer la matrice A
A = []
for i in range(points_selected1):
    ax = np.array([-X_init_norm[i][0], -X_init_norm[i][1], -1, 0, 0, 0, X_final_norm[i][0]*X_init_norm[i][0], X_final_norm[i][0]*X_init_norm[i][1], X_final_norm[i][0]])
    ay = np.array([0, 0, 0, -X_init_norm[i][0], -X_init_norm[i][1], -1, X_final_norm[i][1]*X_init_norm[i][0], X_final_norm[i][1]*X_init_norm[i][1], X_final_norm[i][1]])
    A.append(ax)
    A.append(ay)

A = np.vstack(A)

# Obtenir h (derniere ligne de V = dernier vecteur propre)
U, S, V = np.linalg.svd(A)
h_homographie = V[-1, :]

# Obtenir la matrice d'homographie H
H = h_homographie.reshape(3, 3)

# De-normaliser la solution
H = np.dot(np.linalg.inv(T_norm), H)
H = np.dot(H, T_norm)
H = H/H[-1, -1]

# Fonction qui génère la matrice d'homographie
# H = cv2.getPerspectiveTransform(X_init,X_final)
# print("H_func =", H)

### Votre code d'estimation de H ici

# Apliquer la matrice d'homographie
img_warp = cv2.warpPerspective(clone1, H, (clone1.shape[1]+clone2.shape[1],clone1.shape[0]))

# Fusion des images
img_warp[0:clone2.shape[0], 0:clone2.shape[1]] = clone2 
panorama = img_warp
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
