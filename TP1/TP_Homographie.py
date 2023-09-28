import numpy as np
import cv2

print("Version d'OpenCV: ",cv2. __version__)

# Ouverture de l'image
PATH_IMG = './Images_Homographie/'

img = np.uint8(cv2.imread(PATH_IMG+"Pompei.jpg"))

(h,w,c) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"couleurs")

def select_points(event, x, y, flags, param):
	global points_selected,X_init
	global img,clone
	if (event == cv2.EVENT_FLAG_LBUTTON):
		x_select,y_select = x,y
		points_selected += 1
		cv2.circle(img,(x_select,y_select),8,(0,255,255),1)
		cv2.line(img,(x_select-8,y_select),(x_select+8,y_select),(0,255,0),1)
		cv2.line(img,(x_select,y_select-8),(x_select,y_select+8),(0,255,0),1)
		X_init.append( [x_select,y_select] )
	elif event == cv2.EVENT_FLAG_RBUTTON:
		points_selected = 0
		img = clone.copy()
		
clone = img.copy()
points_selected = 0
X_init = []
cv2.namedWindow("Image initiale")
cv2.setMouseCallback("Image initiale",select_points)

while True:
	cv2.imshow("Image initiale",img)
	key = cv2.waitKey(1) & 0xFF
	if (key == ord("q")) & (points_selected >= 4):
		break
		
# Conversion en array numpy
X_init = np.asarray(X_init,dtype = np.float32) 		
print("X_init =",X_init)
X_final = np.zeros((points_selected,2),np.float32)
for i in range(points_selected):
	string_input = "Correspondant de {} ? ".format(X_init[i])
	X_final[i] = input(string_input).split(" ",2)
print("X_final =",X_final)


#### Votre code d'estimation de H ici

# Coordonnees au format (x, y, 1)
X_init_norm = np.hstack((X_init, np.ones((X_init.shape[0], 1))))
X_final_norm = np.hstack((X_final, np.ones((X_final.shape[0], 1))))

# Normalisation des coordonnees
T_norm = np.array([[1/w, 0, -1],
                  [0, 1/h, -1],
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
for i in range(points_selected):
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

#### Votre code d'estimation de H ici

# Juste un exemple pour afficher quelque chose
img_warp = cv2.warpPerspective(clone, H, (w,h))
cv2.imshow("Image rectifiee",img_warp)
cv2.waitKey(0)
