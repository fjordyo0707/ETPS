import ReadColMap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics
import PIL.Image

class Superpixel:
    def __init__(self):
        self.idx = -1
        self.rows = []
        self.cols = []
        self.featureRows = []
        self.featureCols = []
        self.featureDepth = []
        self.featureNumber = 0
        self.agentPixel = [0,0]
        self.superDepth = -1
        self.label = 0

    def Print(self):
        print('Patxh Number: ', self.idx)
        print('Number of pixel: ', len(self.rows))
        print('Number of featureRows: ', len(self.featureRows))
        print('Number of featureDepth: ', len(self.featureDepth))
        print('agentPixel: ', self.agentPixel)
        print('superDepth: ', self.superDepth)
        print('Label: ', self.label)

class Patch:
    def __init__(self):
        self.nonfeatureSuperpixel = []
        self.featureSuperpixel = []
    def Print(self):
        print('Number nonfeatureSuperpixel: ', len(self.nonfeatureSuperpixel))
        print('Number featrueSuperpixel: ', len(self.featureSuperpixel))


def Process_Label():
    label_png = './sample_data/frames/000026_json/label.png'
    lbl = np.asarray(PIL.Image.open(label_png))
    label_idx = np.unique(lbl)
    return lbl, label_idx

# I use 000029.png to demonstrate
# The conresponding idx in Views is 28
# 26.png -> idx 24
# 29.png -> idx 28
def initialTest(input_frames, recon):
    testFrameidx = 28
    testfile = "000029"
    testCase = recon.views[testFrameidx]
    testCase.Print()
    testImg = cv2.imread(input_frames+testfile+".png")
    h, w, ch = testImg.shape
    global global_h
    global global_w
    global_h = h
    global_w = w
    sparseDepthImg = np.zeros((h, w, ch))
    #print(sparseDepthImg.shape)
    for key, value in testCase.points2d.items():
        cv2.circle(sparseDepthImg, (int(value[0]), int(value[1])), 15, (0,0,255), -1)

    cv2.imwrite(testfile+"_sparse.png",sparseDepthImg)
    depthSparse = recon.GetSparseDepthMap(testFrameidx)
    return depthSparse, h, w

def constructSuperpixels(inputSparseDepth, seg_lbl, seg_idx):
    print('**************start construct Superpixels*****************')
    superPixelArray = []
    #label_withSuper = [[] for _ in range(len(seg_idx))]
    idxNoneZeroRowSparse, idxNoneZeroColSparse = np.nonzero(inputSparseDepth)
    fs = cv2.FileStorage("label_1500.xml", cv2.FILE_STORAGE_READ)
    fn_label_matrix = fs.getNode("label_matrix")
    fn_label_number = fs.getNode("label_number")
    label_matrix = fn_label_matrix.mat()
    label_number = int(fn_label_number.real())
    print('label number: ', label_number)

    for i in range(label_number+1):
        tempSuper = Superpixel()
        tempSuper.idx = i
        tempRow, tempCol = np.nonzero(label_matrix == i)
        tempSuper.rows =  tempRow
        tempSuper.cols = tempCol
        tempSuper.agentPixel[0] = int(np.median(tempRow))
        tempSuper.agentPixel[1] = int(np.median(tempCol))
        tempSuper.label = seg_lbl[tempSuper.agentPixel[0]][tempSuper.agentPixel[1]]
        superPixelArray.append(tempSuper)
        #label_withSuper[int(tempSuper.label)].append(tempSuper)

    print('')
    print('Number of Superpixel: ', len(superPixelArray))

    for i,j in zip(idxNoneZeroRowSparse, idxNoneZeroColSparse):
        tempIdxSupers = label_matrix[i][j]
        nowSuper = superPixelArray[tempIdxSupers]
        nowSuper.featureRows.append(i)
        nowSuper.featureCols.append(j)
        nowSuper.featureDepth.append(inputSparseDepth[i,j])
        nowSuper.superDepth = inputSparseDepth[i,j]
        #nowSuper.Print()
    return superPixelArray

def constructPatches(sortSuperPixels, seg_idx):
    label_withSuper = [[] for _ in range(len(seg_idx))]
    patchArray = []
    for oneSuper in sortSuperPixels:
        label_withSuper[int(oneSuper.label)].append(oneSuper)

    for onelabel in label_withSuper:
        tempPatch = Patch()
        for oneSuper in onelabel:
            if oneSuper.superDepth != -1:
                tempPatch.featureSuperpixel.append(oneSuper.idx)
            else:
                tempPatch.nonfeatureSuperpixel.append(oneSuper.idx)
        patchArray.append(tempPatch)
        tempPatch.Print()
    return patchArray

def propogateDepth(sortSuperPixels, patchArray):
    DenseDepthMap = np.zeros((global_h,global_w))
    for onePatch in patchArray:
        ctpoint_rows = np.zeros(len(onePatch.featureSuperpixel))
        ctpoint_cols = np.zeros(len(onePatch.featureSuperpixel))
        ctpoint_depth = np.zeros(len(onePatch.featureSuperpixel))
        for i in range(len(onePatch.featureSuperpixel)):
            ctpoint_rows[i] = sortSuperPixels[onePatch.featureSuperpixel[i]].agentPixel[0]
            ctpoint_cols[i] = sortSuperPixels[onePatch.featureSuperpixel[i]].agentPixel[1]
            ctpoint_depth[i] = sortSuperPixels[onePatch.featureSuperpixel[i]].superDepth
        for i in range(len(onePatch.nonfeatureSuperpixel)):
            udpoint_row = sortSuperPixels[onePatch.nonfeatureSuperpixel[i]].agentPixel[0]
            udpoint_col = sortSuperPixels[onePatch.nonfeatureSuperpixel[i]].agentPixel[1]
            weighted = np.reciprocal(np.sqrt((ctpoint_rows - udpoint_row)**2 + (ctpoint_cols - udpoint_col)**2))
            weighted_normalize = weighted/weighted.sum()
            sortSuperPixels[onePatch.nonfeatureSuperpixel[i]].superDepth = np.multiply(weighted_normalize,ctpoint_depth).sum()
        print('Finish one patch')

    for onesuper in sortSuperPixels:
        for row, col in zip(onesuper.rows, onesuper.cols):
            DenseDepthMap[row][col] = onesuper.superDepth
    return DenseDepthMap












'''
plt.imshow(CoarseDepthMap, cmap='gray')
plt.show()
cv2.namedWindow("CoarseDepthMap", cv2.WINDOW_NORMAL)
cv2.imshow("CoarseDepthMap",CoarseDepthMap)
cv2.waitKey(10000)
cv2.destroyWindow("CoarseDepthMap")

cv2.imwrite("CoarseDepthMap.png",CoarseDepthMap)
'''
def main():
    input_frames = "sample_data/frames/"
    input_colmap = "sample_data/reconstruction/"
    output_folder = "output/"
    recon = ReadColMap.ReadColmap(input_colmap, input_frames)

    global_h = 0
    global_w = 0
    seg_lbl, seg_idx = Process_Label()
    sparseDepthMap,h ,w = initialTest(input_frames, recon)
    sortSuperPixels = constructSuperpixels(sparseDepthMap, seg_lbl, seg_idx)
    patchArray = constructPatches(sortSuperPixels, seg_idx)
    DenseDepthMap = propogateDepth(sortSuperPixels, patchArray)
    plt.imsave("SuperDense.png", DenseDepthMap, cmap = 'viridis')


if __name__ == '__main__':
    main()