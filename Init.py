
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage import filters
from PIL import Image,ImageCms
from scipy.fftpack import fftn, ifftn, fftshift


def RGB2LAB(img):

    if img.mode != "RGB":
        img = img.convert("RGB")

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile  = ImageCms.createProfile("LAB")

    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    lab_im = ImageCms.applyTransform(img, rgb2lab_transform)

    return(lab_im)
##################################################################################
def Histogram(img):

    h_len =  int(np.max(img)+1)
    hist = [0] * h_len

    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            hist[int(img[x][y])] += 1
  
    return(hist,h_len)
##################################################################################
def AdaptHistEq(img,k,CLIPLIMIT = 0.01):
    # Direto do artigo

    img_out = np.matrix(img)
    eqcount = np.zeros(img.shape)
    for x in np.arange(k,img.shape[0]-k):
        for y in np.arange(k,img.shape[1]-k):

            for xi in np.arange(x-k, x+k):
                for yi in np.arange(y-k ,y+k):
                    if img[x,y] == img[xi,yi]:
                        eqcount[x,y] +=1

    for x in np.arange(k,img.shape[0]-k):
        for y in np.arange(k,img.shape[1]-k):
            cliptotal = 0
            partialrank = 0
            incr = 0
            for xi in np.arange(x-k, x+k):
                for yi in np.arange(y-k ,y+k):
                    if eqcount[xi,yi] > CLIPLIMIT:
                        incr = CLIPLIMIT / eqcount[xi,yi]
                    else:
                        incr = 1
            
                cliptotal = cliptotal + (1 - incr)
                if img[x,y] > img[xi,yi]:
                    partialrank = partialrank + incr

            redistr = (cliptotal / (k*k)) * img[x,y]
            img_out[x,y] = partialrank + redistr

    return(img_out)
##################################################################################
def Contrast_Modulation(img,d,c = 0):
    
    a = np.min(img)
    b = np.max(img)
    img_out = np.zeros(img.shape,dtype=np.float64)
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            img_out[x,y] = (img[x,y] - a) * ((d - c) / (b - a)) + c

    return img_out
##################################################################################
def Gamma_Adjustment(img,gamma = 0.04,c = 1):

    img_out = np.zeros(img.shape,dtype=np.float64)

    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):
            img_out[x,y] = c*(img[x,y]**(gamma))
    return(img_out)
##################################################################################
def Binarizacao(img):
    
    Th = filters.threshold_otsu(img)
    img_out = np.zeros(img.shape)
    
    for x in np.arange(img.shape[0]):
        for y in np.arange(img.shape[1]):

            if img[x,y] <= Th:
                img_out[x,y] = 0
            else:
                img_out[x,y] = 1
##################################################################################
def Mean_Filter(img,k):

    img_out = np.copy(img)

    for x in np.arange(k,img.shape[0]-k):
        for y in np.arange(k,img.shape[1]-k):
            sub_img = img_out[ x-k : x+k , y-k :y+k ]
            img_out[x,y] = np.mean(sub_img) 

    return(img_out)
##################################################################################
def Test_Lab(image_name):

    # Converter para LAB e extrair o canal L
    img = np.array(RGB2LAB(Image.open(image_name)))
    L = img[:, :, 0]
    
    # Calcula onde começa a retina do fundo
    hist,h_len = Histogram(L)
    mediana = np.argsort(hist)[round(h_len/2)]
    mask = np.where(L < hist[mediana], L , 255)

    L_C = Contrast_Modulation(L,120) # Aplica um ajuste de constrast
    L_Gamma = Gamma_Adjustment(L,1.0,0.3) # Aplica um ajuste de constrast

    # Mostra a imagem no canal L e, as imagem depois do ajuste
    plt.figure(figsize=(15,15))
    plt.suptitle("Teste de Ajuste usando o canal L")

    plt.subplot(131); plt.imshow(L, cmap="gray"); plt.title("L")
    plt.subplot(132); plt.imshow(L_C , cmap="gray"); plt.title("Contrast Modulation")
    plt.subplot(133); plt.imshow(L_Gamma, cmap="gray"); plt.title("Gamma Adjustment")
    plt.show()
    
    # Converte o fundo preto para branco
    L = np.where(L < hist[mediana],255,L) 

    #Faz um ajuste de contrast utilizando o CLAHE
    L_CLAHE = equalize_adapthist(L,kernel_size=64, clip_limit=0.09, nbins=128)
    # Aplica um mean filter no resultado do CLAHE
    L_mean = Mean_Filter(L_CLAHE,5)

    # remove o fundo imagem
    L_diff = L_mean - L_CLAHE
    L_diff = np.where(L_diff < 0,0,L_diff) # O resultado da subtração pode ser negativo, logo é feito um ajuste
    
    # Mostra o resultado CLAHE, do mean filter e da remoção do fundo
    
    plt.figure(figsize=(15,15))
    plt.suptitle("Teste de Ajuste usando CLAHE e Mean Filter")
    plt.subplot(131); plt.imshow(L_CLAHE, cmap="gray"); plt.title("CLAHE")
    plt.subplot(132); plt.imshow(L_mean , cmap="gray"); plt.title("Mean")
    plt.subplot(133); plt.imshow(L_diff, cmap="gray"); plt.title("Remoção do Fundo")
    
    # Realizar algum ajuste da imagem sem o fundo, aplicar a binarização e, talvez algum operador morfologico
    plt.show()
##################################################################################
def Test_RGB(image_name):

    # Converter para LAB e extrair o canal L
    img = Image.open(image_name)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = np.array(img)
    G = img[:, :, 1]
    
    # Calcula onde começa a retina do fundo
    hist,h_len = Histogram(G)
    mediana = np.argsort(hist)[round(h_len/2)]
    mask = np.where(G < hist[mediana], G , 255)
    
    G_C = Contrast_Modulation(G,32) # Aplica um ajuste de constrast
    G_Gamma = Gamma_Adjustment(G,0.9,0.3) # Aplica um ajuste de constrast

     # Mostra a imagem no canal L e, as imagem depois do ajuste
    plt.figure(figsize=(15,15))
    plt.suptitle("Teste de Ajuste usando o canal G")
    plt.subplot(131); plt.imshow(G, cmap="gray"); plt.title("G")
    plt.subplot(132); plt.imshow(G_C , cmap="gray"); plt.title("Contrast Modulation")
    plt.subplot(133); plt.imshow(G_Gamma, cmap="gray"); plt.title("Gamma Adjustment")
    plt.show()

    # Converte o fundo preto para branco
    G = np.where(G < hist[mediana],255,G) 
    # Faz um ajuste de contrast utilizando o CLAHE
    G_CLAHE = equalize_adapthist(G,kernel_size=64, clip_limit=0.09, nbins=128)
    # Aplica um mean filter no resultado do CLAHE
    G_mean = Mean_Filter(G_CLAHE,5)
    
    # Remove o fundo imagem
    G_diff = G_mean - G_CLAHE
    G_diff = np.where(G_diff < 0,0,G_diff) # O resultado da subtração pode ser negativo, logo é feito um ajuste
    
    # Mostra o resultado CLAHE, do mean filter e da remoção do fundo
    plt.figure(figsize=(15,15))
    plt.suptitle("Teste de Ajuste usando CLAHE e Mean Filter")
    plt.subplot(131); plt.imshow(G_CLAHE, cmap="gray"); plt.title("CLAHE")
    plt.subplot(132); plt.imshow(G_mean , cmap="gray"); plt.title("Mean")
    plt.subplot(133); plt.imshow(G_diff, cmap="gray"); plt.title("Remoção do Fundo")

    # Realizar algum ajuste da imagem sem o fundo, aplicar a binarização e, talvez algum operador morfologico
    plt.show()
##################################################################################

Test_Lab("01_original.jpg")
Test_RGB("01_original.jpg")