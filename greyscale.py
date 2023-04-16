#   Rostislav Kucera
#   12.4.2023
#   Greyscale conversion to maintain discriminability
#   Original work: "Color-to-grayscale conversion to maintain discriminability" - Raja Bala, Karen M. Braun

import sys, getopt
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import imageio
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976
import numpy


numpy.set_printoptions(threshold=sys.maxsize)

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)


def main(argv):
    inputfile = 'dia.jpg'
    outputfile = 'def_out.jpg'
    clusters = 20
    mode = equal
    try:
        opts, args = getopt.getopt(argv,"n:m:i:o:",["mode=", "colors=", "ifile=","ofile="])
    except getopt.GetoptError:
        print ('greyscale.py -m <mode> -n <num_colors> -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--ifile"):
            inputfile = arg
        if opt in ("-o", "--ofile"):
            outputfile = arg
        if opt in ("-n", "--clusters"):
            clusters = int(arg)
        if opt in ("-m", "--mode"):
            mode = arg
    if mode == "weighted":
        col_diff(clusters, inputfile, outputfile)
    elif mode == "equal":
        equal(clusters, inputfile, outputfile)
        
#funtion implementing weighted lightness spacing
#that is described in "Color-to-grayscale conversion to maintain discriminability"
#by Raja Bala, Karen M. Braun

def col_diff(cluster_num, input, out):
    print('reading image')
    im = Image.open(input)
    ar = np.asarray(im)
    shape = ar.shape #save the original 3d shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float) #reshape to 2d array

    #finding color clusters for removing color noise
    print('finding clusters')
    cluster_cents, _ = scipy.cluster.vq.kmeans(ar, cluster_num)

    centroid, _= scipy.cluster.vq.vq(ar, cluster_cents) #to which centroid does it belong

    c = ar.copy()

    gray_arr = np.dot(cluster_cents[...,:3], [0.2126, 0.7152, 0.0722]) #create array of lumiance of every color for sorting

    for i, cent in enumerate(gray_arr):
        c[scipy.r_[np.where(centroid==i)],:] = cent #change color to one of the greyscale

    imageio.imwrite('std_' + out, c.reshape(*shape).astype(np.uint8)) #save standard greyscale picture
    
    tmp = sorted(enumerate(gray_arr), key=lambda i: i[1]) #sort colors and save their orig. indexes

    res = [0]*len(gray_arr) #preallocate result array

    for n in range(0, len(tmp)):
        #darkest color
        if n == 0:
            res[tmp[n][0]] = 0.0;
        else:
            sum_n = 0;
            sum_N = 0;
            for i in range(1, n+1):
                rgb_arr = cluster_cents[tmp[i][0]].tolist() #create list from input RGB format
                col_act = sRGBColor(rgb_arr[0], rgb_arr[1], rgb_arr[2]) #convert color to sRGB format
                rgb_arr = cluster_cents[tmp[i-1][0]].tolist()
                col_prev = sRGBColor(rgb_arr[0], rgb_arr[1], rgb_arr[2])
                col_act_lab = convert_color(col_act, LabColor) #convert sRGB to LabColor (Yxy model)
                col_prev_lab = convert_color(col_prev, LabColor)
                delta_e = delta_e_cie1976(col_act_lab, col_prev_lab) #distance between 2 colors in CIE1976 model
                sum_n += delta_e

            for i in range(1, len(tmp)):
                rgb_arr = cluster_cents[tmp[i][0]].tolist()
                col_act = sRGBColor(rgb_arr[0], rgb_arr[1], rgb_arr[2])
                rgb_arr = cluster_cents[tmp[i-1][0]].tolist()
                col_prev = sRGBColor(rgb_arr[0], rgb_arr[1], rgb_arr[2])
                col_act_lab = convert_color(col_act, LabColor)
                col_prev_lab = convert_color(col_prev, LabColor)
                delta_e = delta_e_cie1976(col_act_lab, col_prev_lab) 
                sum_N += delta_e

            res[tmp[n][0]] = 255.0 * (sum_n / sum_N);
    #compute new img array
    for i, code in enumerate(res):
        c[scipy.r_[np.where(centroid==i)],:] = code

    imageio.imwrite(out, c.reshape(*shape).astype(np.uint8)) #reshape to 3d array and save enhanced greyscale picture
    print('saved greyscale images')

#funtion implementing equal lightness spacing
#that is described in "Color-to-grayscale conversion to maintain discriminability"
#by Raja Bala, Karen M. Braun

def equal(cluster_num, input, out):
    print('reading image')
    im = Image.open(input)
    ar = np.asarray(im)
    im.close()
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    print('finding clusters')
    cluster_cents, _ = scipy.cluster.vq.kmeans(ar, cluster_num)

    vecs, _ = scipy.cluster.vq.vq(ar, cluster_cents)

    c = ar.copy()
    gray_arr = np.dot(cluster_cents[...,:3], [0.2126, 0.7152, 0.0722])


    for i, code in enumerate(gray_arr):
        c[scipy.r_[np.where(vecs==i)],:] = code

    imageio.imwrite('std_' + out, c.reshape(*shape).astype(np.uint8))
    
    tmp = sorted(enumerate(gray_arr), key=lambda i: i[1])
    ind = [0]*len(gray_arr)

    x = 255.0/len(gray_arr)

    for i in range(0, len(tmp)):
        ind[tmp[i][0]] = x*i

    #compute new img array
    for i, code in enumerate(ind):
        c[scipy.r_[np.where(vecs==i)],:] = code

    imageio.imwrite(out, c.reshape(*shape).astype(np.uint8))

    print('saved greyscale images')


if __name__ == "__main__":
   main(sys.argv[1:])