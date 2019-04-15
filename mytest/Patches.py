import ReadColMap
import cv2
import numpy as np
import matplotlib.pyplot as plt

class patch:
    def __init__(self):
        self.idx = -1
        self.rows = []
        self.cols = []
        self.featureRows = []
        self.featureCols = []
        self.featureDepth = []
        self.featureNumber = 0

    def Print(self):
        print('Patxh Number: ', self.idx)
        print('Number of pixel: ', len(self.rows))
        print('Number of featureRows: ', len(self.featureRows))
        print('Number of featureDepth: ', len(self.featureDepth))


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

def constructPatches(inputSparseDepth):
    print('**************start construct patches*****************')
    patches = []
    idxNoneZeroRowSparse, idxNoneZeroColSparse = np.nonzero(inputSparseDepth)
    fs = cv2.FileStorage("label_500.xml", cv2.FILE_STORAGE_READ)
    fn_label_matrix = fs.getNode("label_matrix")
    fn_label_number = fs.getNode("label_number")
    label_matrix = fn_label_matrix.mat()
    label_number = int(fn_label_number.real())
    print('label number: ', label_number)

    for i in range(label_number+1):
        tempPatch = patch()
        tempPatch.idx = i
        tempRow, tempCol = np.nonzero(label_matrix == i)
        tempPatch.rows =  tempRow
        tempPatch.cols = tempCol
        #tempPatch.Print()
        patches.append(tempPatch)

    print('')
    print('Number of patch: ', len(patches))

    for i,j in zip(idxNoneZeroRowSparse, idxNoneZeroColSparse):
        tempIdxPatches = label_matrix[i][j]
        nowPatch = patches[tempIdxPatches]
        nowPatch.featureRows.append(i)
        nowPatch.featureCols.append(j)
        nowPatch.featureDepth.append(inputSparseDepth[i,j])
        #nowPatch.Print()
    return patches


def constructCoarseDepthMap(patches):
    CoarseDepthMap = np.zeros((global_h,global_w))
    for currentPatch in patches:

        for row, col in zip(currentPatch.rows, currentPatch.cols):
            if len(currentPatch.featureDepth) is not 0:
                CoarseDepthMap[row,col] = currentPatch.featureDepth[0]
            else:
                CoarseDepthMap[row,col] = 0
    #print(CoarseDepthMap[np.nonzero(CoarseDepthMap)])
    return CoarseDepthMap




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
    sparseDepthMap,h ,w = initialTest(input_frames, recon)
    sortPatches = constructPatches(sparseDepthMap)
    CoarseDepthMap = constructCoarseDepthMap(sortPatches)
    #plt.imsave("CoarseDepthMap.png", CoarseDepthMap, cmap = 'seismic')

if __name__ == '__main__':
    main()