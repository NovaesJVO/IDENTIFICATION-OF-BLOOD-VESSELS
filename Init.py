import numpy as np
from matplotlib import use, pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage import morphology
from PIL import Image,ImageCms
from scipy import ndimage
from sklearn.metrics import precision_recall_curve,average_precision_score
import os

use("TkAgg")


def RGB2LAB(img):
    
    # Create an object to do the RGB to LAB conversion
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    # Generates the image in the LAB channel
    lab_im = ImageCms.applyTransform(img, rgb2lab_transform)

    # Return to LAB image
    return(lab_im)
##################################################################################
def Histogram(img):

    # Create a color histogram
    histogram = []
    Map = {} # Map, to map the colors to the histogram positions
    key = 0
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):

            # If the color has already been entered in the histogram, the count value increases.
            if int(img[x,y]) in Map:
                temp = histogram[Map[int(img[x,y])]]
                histogram[Map[int(img[x,y])]] = (temp[0],temp[1] + 1)
            # If not, enter the color in the histogram, with count value 1
            else:
                Map[int(img[x,y])] = int(key)
                key += 1
                histogram.append((img[x,y],1))

        # Histogram tuple: 0-> with 1 -> number of occurrences
    return(histogram)
##################################################################################
def Binarizacao(img):
    
    # If image for float converts to uint8
    if img.dtype != 'uint8':
        img = Normaliza(img)

    # Calculate the threshold using the Otsu method
    Th = Otsu(img)
   
    img_out = np.zeros(img.shape,dtype = 'bool')
    
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            
            # If the pixel value is less than the threshold, it becomes False (Zero)
            if img[x,y] < Th:
                img_out[x,y] = False
            # If not, True (1)
            else:
                img_out[x,y] = True
    return(img_out)
##################################################################################
def Mean_Filter(img,kernel_size = 9):

    img_out = np.copy(img)

    k = int(kernel_size/2)
    # Apply a Mean Filter to all pixels (minus the edges) using a window of (n x n)
    for x in np.arange(k,img.shape[0]-k):
        for y in np.arange(k,img.shape[1]-k):

            # Generates a submatrix and calculates the mean
            sub_img = img[ x-k : x+k , y-k :y+k ]
            img_out[x,y] = np.mean(sub_img) 

    return(img_out)
##################################################################################
def Luminance_1D(img):

    img_out = np.zeros((img.shape[0],img.shape[1]))
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            # Convert the RGB image to GrayScale using the formula below
            img_out[x,y] = 0.299 * img[x,y,0] + 0.587 * img[x,y,1] + 0.114 * img[x,y,2]
    
    return(img_out)
##################################################################################
def Otsu(img):

    # Generates an image color histogram
    histogram = Histogram(img)
    h_len =len(histogram)
    # Sort the histogram by color value [0]
    histogram.sort()
    
    limiar = 0
    # The threshold is initially infinite
    Th = np.infty

    # Go through all the colors of the histogram, looking for the one that generates the best division of classes
    for l in histogram:
        
        soma_h_a = np.float64(0)
        soma_h_b = np.float64(0)
        soma_a = np.float64(0)
        soma_b = np.float64(0)
        mean_a = np.float64(0)
        mean_b = np.float64(0)
        var_b = np.float64(0)
        var_a = np.float64(0)
        Th_temp = np.float64(0)

        
        # Calculates the value of Wa and Wb
        for E in histogram:

            if E[0] < l[0]:
               soma_h_a += E[1]
               soma_a +=  (E[0]) * E[1]
            else:
                soma_h_b += E[1]
                soma_b +=  (E[0]) * E[1]

        if soma_h_a == 0:
            mean_a = 0
        else:
            mean_a = soma_a/soma_h_a
        if soma_h_b == 0:
            mean_b = 0
        else:
            mean_b = soma_b/soma_h_b
        # Calculate the variance of a and b
        for E in histogram:
            if E[0] < l[0]:
                
                var_a += ((E[0]-mean_a)**2) * E[1]
            else:
                var_b += ((E[0]-mean_b)**2) * E[1]

        if soma_h_a == 0:
            var_a = 0
        else:
            var_a = (var_a/soma_h_a)
        if soma_h_b == 0:
            var_b = 0
        else:
            var_b = (var_b/soma_h_b)

        Th_temp = (soma_h_a/h_len) * var_a
        Th_temp += (soma_h_b/h_len) * var_b
        
        # If the new threshold is better than current, change the value
        if Th_temp < Th:
            Th = Th_temp
            limiar = l[0]

    return(limiar)
##################################################################################
def Normaliza(img):

    v_max = np.max(img)

    img_out = np.zeros(img.shape,dtype=np.uint8)
    # Normalizes the image to a range between 0 and 255 and converts to integer
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            img_out[x,y] = np.uint8(((img[x,y] )/ (v_max )) *255)

    return(img_out)
##################################################################################
def Comparar(img, img_ref,cor = [0,0,255],):


    img_ref = np.array(Image.open(img_ref),dtype='bool')
    img_out = np.zeros((img_ref.shape[0],img_ref.shape[1],3),dtype=np.uint8)

    n = np.float(img_ref.shape[0] * img_ref.shape[1])
    count = 0.0
    for x in np.arange(img_ref.shape[0]):
        for y in np.arange(img_ref.shape[1]):

            if img_ref[x,y] != 0 and img[x,y] == 0:
                img_out[x,y] = cor
            elif img_ref[x,y] == 0 and img[x,y] != 0:
                img_out[x,y] = cor
            else:
                count +=1


    return(img_out,float(count/n),img_ref)
##################################################################################
def Segment_blood_vessel(image_name,img_ref):
    
    # Read the image 
    img = Image.open(image_name)
    # If it is not in RGB, it converts to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Converts the image to LAB
    img_Lab = np.array(RGB2LAB(img))
    # Take LAB channel L
    L = img_Lab[:, :, 0]
    img = np.array(img)

    # Converts the image to GrayScale using the luminance formula, as the return value can be float a normalization applies.
    G = Normaliza(Luminance_1D(img))

    # Shows incoming image and converted images
    plt.figure()
    plt.suptitle("Input, L and Grayscale")
    plt.subplot(131); plt.imshow(img); plt.title("Input")
    plt.subplot(132); plt.imshow(L , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G , cmap="gray"); plt.title("GrayScale")
    plt.show()

    # Apply a contrast adjustment using an adaptive histogram
    L_CLAHE = equalize_adapthist(L,kernel_size=64, clip_limit=0.09, nbins=128)
    G_CLAHE = equalize_adapthist(G,kernel_size=64, clip_limit=0.09, nbins=128)
    
    # Displays the adjustment result
    plt.figure(figsize=(10,10))
    plt.suptitle("CLAHE")
    plt.subplot(131); plt.imshow(L_CLAHE , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_CLAHE , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Applies a Mean Filter to the adjusted image
    L_mean = Mean_Filter(L_CLAHE,13)
    G_mean = Mean_Filter(G_CLAHE,13)

    # Shows the result of Mean Filter
    plt.figure(figsize=(10,10))
    plt.suptitle("Mean Filter")
    plt.subplot(131); plt.imshow(L_mean , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_mean , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Subtract from the filtered image, the image with contrast adjustment, so the vases should stand out more than the background of the image
    L_diff = L_mean - L_CLAHE
    L_diff = np.where(L_diff < 0,0,L_diff) # The result of the subtraction may be negative, so an adjustment is made

    G_diff = G_mean - G_CLAHE
    G_diff = np.where(G_diff < 0,0,G_diff) # The result of the subtraction may be negative, so an adjustment is made
    
    # Mostra o resultado da remoção do fundo
    plt.figure(figsize=(10,10))
    plt.suptitle("Removal of the Fund")
    plt.subplot(131); plt.imshow(L_diff , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_diff , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Normalizes the image without the background, the adjustment using adpathic histogram returns a matrix float
    G_diff = Normaliza(G_diff)
    G_diff_CLAHE = equalize_adapthist(G_diff) # Applies a new ad-hoc histogram to further enhance the vessels
    L_diff = Normaliza(L_diff)
    L_diff_CLAHE = equalize_adapthist(L_diff) # Applies a new ad-hoc histogram to further enhance the vessels
    
    # Displays the result of the re-adjustment
    plt.figure(figsize=(10,10))
    plt.suptitle("Removal of the Fund - CLAHE")
    plt.subplot(131); plt.imshow(L_diff_CLAHE , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_diff_CLAHE , cmap="gray"); plt.title("Grayscale")
    plt.show()
    
    # Normalizes the image again and applies a binarization
    G_diff_CLAHE = Normaliza(G_diff_CLAHE)
    G_bin = Binarizacao(G_diff_CLAHE)
    L_diff_CLAHE = Normaliza(L_diff_CLAHE)
    L_bin = Binarizacao(L_diff_CLAHE)

    # Shows the result of binarization
    plt.figure(figsize=(10,10))
    plt.suptitle("Binarization")
    plt.subplot(131); plt.imshow(L_bin , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_bin , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Apply a morphological operation of area opening, thus removing any noise generated by binarization
    G_op = morphology.remove_small_objects(G_bin, min_size=25)
    L_op = morphology.remove_small_objects(L_bin, min_size=30)

    # Shows the result of the morphological operation
    plt.figure(figsize=(10,10))
    plt.suptitle("Area Opening ")
    plt.subplot(131); plt.imshow(L_op , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(G_op , cmap="gray"); plt.title("Grayscale")
    plt.show()


    # Calculate an image histogram in GrayScale
    hist = Histogram(G)
    hist.sort(key=lambda tup: tup[1])
    # Calculates the threshold between the retina and the black background of the image
    limiar = max(hist[1:100],key=lambda item:item[1])[0]
    mask_G = np.where(G <= limiar*1.5,1,0)
    mask_G = ndimage.binary_dilation(mask_G)

    hist = Histogram(L)
    hist.sort(key=lambda tup: tup[1])
    limiar = max(hist[1:100],key=lambda item:item[1])[0]

    mask_L = np.where(L <= limiar*1.5,1,0)
    mask_L = ndimage.binary_dilation(mask_L) # To reduce mask shape errors, a swelling operation is performed

    # Shows the background mascara  
    plt.figure(figsize=(10,10))
    plt.suptitle("Background Mask")
    plt.subplot(131); plt.imshow(mask_L , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(mask_G, cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Subtract the background mask from the image, thus removing the edge noise from the image  
    out = L_op.astype(np.float) - mask_L
    out_L = np.where(out < 0,0,out) # The result of the subtraction may be negative, so an adjustment is made

    # Subtract the background mask from the image, thus removing the edge noise from the image  
    out = G_op.astype(np.float) - mask_G
    out_G = np.where(out < 0,0,out)  

    # Displays the image after removing the border
    plt.figure(figsize=(10,10))
    plt.suptitle("Remove Border")
    plt.subplot(131); plt.imshow(out_L , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(out_G , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Compare the generated images with those of the manual image
    img_out_L,taxa_L,_ = Comparar(out_L,img_ref,[0,255,0])
    img_out_G,taxa_G,img_ref = Comparar(out_G,img_ref,[255,0,0])

    # Shows the result of the comparison
    plt.figure(figsize=(10,10))
    plt.suptitle("Comparation")
    plt.subplot(131); plt.imshow(img_ref , cmap="gray"); plt.title("Manual")
    plt.subplot(132); plt.imshow(img_out_L , cmap="gray"); plt.title("L")
    plt.subplot(133); plt.imshow(img_out_G , cmap="gray"); plt.title("Grayscale")
    plt.show()

    # Plot the percentage of hits between images
    plt.bar(1,taxa_L,color='g')
    plt.bar(2,taxa_G,color='r')
    plt.title('Average Score Rate')
    plt.xticks([1,2],['L', 'GrayScale'])
    plt.yticks(np.arange(0, 1, 0.09))
    plt.ylabel('Average Score')
    plt.xlabel('Channel')
    plt.legend()


    plt.show()
##################################################################################

image_name = str(input("Enter the path to input image: "))
img_ref = str(input("Enter the path to reference image: "))

Segment_blood_vessel(image_name,img_ref)
