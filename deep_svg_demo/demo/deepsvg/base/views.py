from django.shortcuts import render
# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from cairosvg import svg2png
import time
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
from vaegan.experiments import query
import os
from PIL import Image
import numpy as np



@api_view(['GET'])
def getRoutes(request, pk):
    file = readFile(pk)
    svg_str = file.readlines()[0]

    ## inputFile is the file path for the intput png for the model
    inputFile = '/Users/yy/Downloads/output.png'
    svg2png(svg_str,write_to=inputFile)

    im = Image.open(inputFile)
    
    
    im = im.convert('RGB')
    im.save('audacious.jpg')
    im.save(r'/Users/yy/Downloads/output.jpg')

    inputFile = '/Users/yy/Downloads/output.jpg'
    
    ### generatedFile is the file path of the result of the model
    generatedFile = '/Users/yy/Downloads/output2.gif'
    ##########################################################################################
    ############### Load the model here, and evaluate the inputFile ##########################
    ##########################################################################################

    import os 
    print("views.py", os.getcwd())

    path = '../../../vaegan'
    png_path = Path('..')
    model = query.load_vae(path=path)
    img = plt.imread(inputFile)
    img = np.asarray(Image.open(inputFile).convert('RGB')).astype(np.float32)
    # print(img[:3])
    # print(img[..., :3].max(), img.min())
    # print(img[..., :3].mean())
    img *= 255.
    file_list = query.img2dataset(img, model, k=1, fmt='gif', path=path)
    file_list = list(map(lambda x: str(png_path / x), file_list))

    print(file_list)

    img_frames = []
    src_file_path = '/Users/yy/courses/csci2470/DeepAnimation/icons/icon_gif/'
    dest_file_path = '/Users/yy/courses/csci2470/DeepAnimation/deep_svg_demo/demo/frontend/public/'
    for x in file_list:
        p = Path(x)
        filename = p.name.split('\\')[-1]
        img_frames += [filename]
        srcfile = src_file_path + filename
        outfile = dest_file_path + filename
        newPath = shutil.copy(srcfile, outfile)

    file_to_del = '/Users/yy/Downloads/' + pk
    os.remove(file_to_del)
    file_to_del = inputFile
    os.remove(inputFile)

    ##########################################################################################
    ##########################################################################################

    return Response(img_frames[0])

def readFile(pk):
    try:
        file = open('/Users/yy/Downloads/'+pk,"r")
    except:
        time.sleep(2)
        file = readFile(pk)
    return file