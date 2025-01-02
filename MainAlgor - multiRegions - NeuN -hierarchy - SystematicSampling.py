import subprocess
#subprocess.call(['C:\\Program Files\\Bitplane\\Imaris 10.1.1\\Imaris.exe', 'id1001'])
import ImarisLib
import numpy as np
import math as m
import random
import nrrd
from tifffile import imwrite
import sys
sys.path.insert(0, 'k:/workstation/code/shared/pipeline_utilities/imaris')
#K:/workstation/code/shared/pipeline_utilities/imaris/imaris_hdr_update.py
import imaris_hdr_update
from skimage.io import imread
import os
from time import sleep
import shutil
import imagej
ij = imagej.init(r'K:\CIVM_Apps\Fiji.app',mode='interactive')


exec(open(r"K:\ProjectSpace\yt133\codes\Compilation of cell counting\5xFAD\CountingCodes\Auto_ver\ij_classifier_test.py").read())

fiji=r"K:\CIVM_Apps\Fiji.app\ImageJ-win32.exe".replace('\\','/')
macro_path="S:/yt133/To_delete_Statistics/macroscript.ijm" #where you save your macro

specimen_name = '220114-23_1'
#root needs to be updated with different specimens
root="S:/yt133/To_delete_Statistics/history/data_5xFAD_NeuNDensity_NewRegions/" + specimen_name + "/" #working folder

#find the label file
folder_path = "B:/20.5xfad.01/BXD77/" + specimen_name + "/Aligned-Data/labels/RCCF/"
for file in os.listdir(folder_path):
    if file.endswith('_labels.nhdr'):
        filename = os.path.join(folder_path, file)

im, header = nrrd.read(filename)
im = im[::-1, ::-1, :].astype(int) # this im will be sent as a variable


def GetServer():
   vImarisLib = ImarisLib.ImarisLib()
   vServer = vImarisLib.GetServer()
   return vServer;

vServer=GetServer()
vImarisLib=ImarisLib.ImarisLib()
v = vImarisLib.GetApplication(1001)
img = v.GetImage(0)#load NeuN
img2 = v.GetImage(1)#load labelmap data
#print(type(img2))
vExtentMin=[img.GetExtendMinX(),img.GetExtendMinY(),img.GetExtendMinZ()]
vExtentMax=[img.GetExtendMaxX(),img.GetExtendMaxY(),img.GetExtendMaxZ()]

vExtentMin2 =[img2.GetExtendMinX(),img2.GetExtendMinY(),img2.GetExtendMinZ()]
print('vExtentMin:',vExtentMin)
print('vExtentMax:',vExtentMax)
aChannel=0


"""
classifiers={
"K:\abababa.classifier" : [ 1, 2, 3 ],
}
for classer_file in classifiers:
  label=classifiers[classer_file]
"""

"""
Yuqi thinks it's better to make this below:
Write the script as a function, and make the regions as a large dictionary
"""
def mainAlgor(label, classifier, step=20, volume_bar=20, volume_avgbar=100):
    ### label: list, classifier: path str, volume_bar: a num that represents the average volume size
    ### step: the sampling step size for systematic sampling

    if len(label) == 1:
        newdir = root + "/" + str(label[0]) + "/"
    elif len(label) > 1:
        newdir = root + "/" + str(label[0]) + '-' + str(label[-1]) + '/'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    folder = newdir + "Tif/"
    output = newdir + "tif_out/"
    processed = newdir + "tif_processed/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(output):
        os.makedirs(output)
    if not os.path.exists(processed):
        os.makedirs(processed)

    if len(label) == 1:
        out_csv_file = root + "/" + str(label[0]) + "_counts.csv"
    elif len(label) > 1:
        out_csv_file = root + "/" + str(label[0]) + '-' + str(label[-1]) + "_counts.csv"

    if os.path.isfile(out_csv_file):
        print("SKIPPING. Already found output csv file here: {}".format(out_csv_file), flush=True)
        return

    print(f" folder={folder} output={output} processed={processed} macro_path={macro_path}")

    s = [6, 6, 6]  # Subvolume size (in pixels)
    indices = np.argwhere(np.isin(im, label))  # Check if elements in 'im' are in 'label'
    x_min, x_max = min(indices[:, 0]), max(indices[:, 0])
    y_min, y_max = min(indices[:, 1]), max(indices[:, 1])
    z_min, z_max = min(indices[:, 2]), max(indices[:, 2])

    # Calculate number of valid subvolumes per dimension
    x_range = range(x_min, x_max - s[0] + 1, step)
    y_range = range(y_min, y_max - s[1] + 1, step)
    z_range = range(z_min, z_max - s[2] + 1, step)

    # Calculate N dynamically
    #N = len(x_range) * len(y_range) * len(z_range) #this might not be final counts of N

    # Initialize output array
    arr = np.empty((0, 3), int)

    # Systematic sampling over the valid grid
    for z in z_range:
        for y in y_range:
            for x in x_range:
                # Check if the subvolume contains only the target label
                subvolume = im[x:x+s[0], y:y+s[1], z:z+s[2]]
                if np.all(np.isin(subvolume, label)):
                    arr = np.append(arr, np.array([[x, y, z]]), axis=0)
    N=arr.shape[0] #agree with the subvol number
    ij_macro, args = save_classifier_macro(folder, output, processed, classifier, macro_path,N=N)
    print(f"Number of valid subvolumes: {arr.shape[0]}")
    print("Subvolume starting points:", arr)

    coordinate_filename = os.path.join(root, f'{label[0]}_coordinates.csv')  # Save subregions' coordinates
    np.savetxt(coordinate_filename, arr, delimiter=",", fmt="%d")

    loc2 = (arr * [25, 25, 25] + vExtentMin2 - vExtentMin) / np.array([1.8, 1.8, 4])  # Relative position to NeuN frame

    # Generate TIFF subvolumes
    for i in range(arr.shape[0]):
        subvol = img.GetDataSubVolumeShorts(loc2[i, 0], loc2[i, 1], loc2[i, 2], 0, 0, 56, 56, 25)
        subvol = np.transpose(subvol, [2, 1, 0])  # Swap dimensions (Imaris exports X and Z swapped)
        filename = folder + 'Region_' + str(i) + '.tif'
        imwrite(filename, np.float32(subvol), imagej=True,
                metadata={'spacing': 4, 'unit': 'um', 'axes': 'ZYX'},
                resolution=(1/1.8, 1/1.8))  # ZYX axis required by ImageJ

    ij.ui().showUI()
    result = ij.py.run_macro(ij_macro, args)

    print("Use ImageJ now!")
    num_neuron = []
    for i in range(arr.shape[0]):
        image = imread(processed + "morpho_" + str(i) + ".tif")
        N_n = image.max()
        for j in range(1, image.max() + 1):
            if image.size - np.count_nonzero(image-j) > 10 * volume_avgbar:
                N_n -= 1
            elif image.size - np.count_nonzero(image-j) < volume_bar:
                N_n -= 1
            elif image.size - np.count_nonzero(image-j) > volume_avgbar:
                N_n += np.floor((image.size - np.count_nonzero(image-j)) / volume_avgbar) - 1
        num_neuron.append(N_n)

    np.savetxt(out_csv_file, num_neuron, fmt='%d', delimiter="\n")

    shutil.move(processed, newdir + "tif_processed/")
    shutil.move(output, newdir + "tif_out/")
    shutil.move(folder, newdir + "Tif/")



#Above is the function. Below is defining all regions and call function
regions = {
    # "AVT-anteroventral thalamic nucleus": {
        # "labels": [61],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },
    # "PFT-Parafasclatericular thalamic nucleus": {
        # "labels": [74],
        # "classifier":  "K:/ProjectSpace/yt133/Labelmap/200316auditory/24_19.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },
    # "POT-Posterior nucleus of the thalamus": {
        # "labels": [78],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },
    # "MGD-Medial geniculate nucleus": {
        # "labels": [79],
        # "classifier":  "K:/ProjectSpace/yt133/Labelmap/200316auditory/24_19.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },
    # "RED-Red nucleus": {
        # "labels": [110],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 15,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },
    # "TMN-Trigeminal motor nucleus": {
        # "labels": [126],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/24_19.classifier",
        # "volume_bar": 15,
        # "volume_avgbar": 100,
        # "group_name": "VolumeDecreased"
    # },

    # "AON-Anterior olfactory nucleus": {
        # "labels": [4],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "PIC-Piriform cortex": {
        # "labels": [12],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    "VCS-Visual Cortex Secondary": {
        "labels": [25],
        "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        "volume_bar": 20,
        "volume_avgbar": 100,
        "group_name": "VolumeIncreased"
    },
    "VCP-Visual Cortex Primary": {
        "labels": [26],
        "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        "volume_bar": 20,
        "volume_avgbar": 100,
        "group_name": "VolumeIncreased"
    },
    # "SUB-Subiculum": {
        # "labels": [28],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/5xFAD/03subiculum.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "CA1": {
        # "labels": [31],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/5xFAD/03subiculum.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "CA3": {
        # "labels": [32],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/5xFAD/03subiculum.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "BLA-Basolateral amygdala": {
        # "labels": [41],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/5xFAD/03subiculum.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "EPF-Endopiriform nucleus": {
        # "labels": [45],
        # "classifier":"K:/ProjectSpace/yt133/Labelmap/200316auditory/24_19.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },
    # "NAC-Nucleus accumbens": {
        # "labels": [48],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "VolumeIncreased"
    # },

    # "ACC-Anterior Cingulate ": {
        # "labels": [9],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "FaDecreasedGray"
    # },
    # "LSN-Lateral septal nucleus": {
        # "labels": [55],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "FaDecreasedGray"
    # },
    # "VPT-Ventral Posterior Thalamic Complex": {
        # "labels": [65],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "FaDecreasedGray"
    # },
    # "SUT-Subthalamic nucleus": {
        # "labels": [99],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "FaDecreasedGray"
    # },
    # "HGN-Hypoglossal nucleus": {
        # "labels": [151],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/200316auditory/short1.classifier",
        # "volume_bar": 20,
        # "volume_avgbar": 100,
        # "group_name": "FaDecreasedGray"
    # },


    # "Foc-fornix": {
        # "labels": [162],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/191209_BLA/BLAc_.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "FADecreasedWhite"
    # },
    # "cst-corticospinal tract": {
        # "labels": [168],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/191209_BLA/BLAc_.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "FADecreasedWhite"
    # },
    # "mel-medial lemniscus": {
        # "labels": [169],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/5xFAD/169.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "FADecreasedWhite"
    # },
    # "mlf-medial longitudinal fasciculus": {
        # "labels": [171],
        # "classifier": "K:/ProjectSpace/yt133/Labelmap/191209_BLA/BLAc_.classifier",
        # "volume_bar": 10,
        # "volume_avgbar": 100,
        # "group_name": "FADecreasedWhite"
    # }
}



# for region_name in regions:
  # label = regions[region_name]["labels"]
  # classifier = regions[region_name]["classifier"]
  # group_name = regions[region_name]['group_name']
  # volume_bar = regions[region_name]["volume_bar"]
  # volume_avgbar = regions[region_name]["volume_avgbar"]
  # mainAlgor(label, classifier, group_name, N = 30, volume_bar = volume_bar, volume_avgbar = volume_avgbar)
# Update to call function without group_name
for region_name in regions:
    label = regions[region_name]["labels"]
    classifier = regions[region_name]["classifier"]
    volume_bar = regions[region_name]["volume_bar"]
    volume_avgbar = regions[region_name]["volume_avgbar"]
    mainAlgor(label, classifier,step = 20, volume_bar=volume_bar, volume_avgbar=volume_avgbar)

