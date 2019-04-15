import ReadColMap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import Patches 
from sklearn.neighbors import KNeighborsClassifier
def Process_Label():
    label_png = './sample_data/frames/000026_json/label.png'
    lbl = np.asarray(PIL.Image.open(label_png))
    label_idx = np.unique(lbl)
    return lbl, label_idx

def constructPatches(label_matrix, label_number, inputSparseDepth):
    print('**************start construct patches*****************')
    patches = []
    idxNoneZeroRowSparse, idxNoneZeroColSparse = np.nonzero(inputSparseDepth)
    for i in label_number:
        tempPatch = Patches.patch()
        tempPatch.idx = i
        tempRow, tempCol = np.nonzero(label_matrix == i)
        tempPatch.rows =  tempRow
        tempPatch.cols = tempCol
        #tempPatch.Print()
        patches.append(tempPatch)
    for i,j in zip(idxNoneZeroRowSparse, idxNoneZeroColSparse):
        tempIdxPatches = label_matrix[i][j]
        nowPatch = patches[tempIdxPatches]
        nowPatch.featureRows.append(i)
        nowPatch.featureCols.append(j)
        nowPatch.featureDepth.append(inputSparseDepth[i,j]*100)
    for one_patch in patches:
        one_patch.featureNumber = len(one_patch.featureDepth)

    return patches

def constructDenseDepthMap(sortPatches):
    DenseDepthMap = np.zeros((global_h,global_w))
    for onePatch in sortPatches:
        p_rows = np.asarray(onePatch.rows, )
        p_cols = np.asarray(onePatch.cols)
        p_data = np.vstack((p_rows,p_cols)).transpose()
        f_rows = np.asarray(onePatch.featureRows)
        f_cols = np.asarray(onePatch.featureCols)
        f_data = np.vstack((f_rows,f_cols)).transpose()
        f_label = np.asarray(onePatch.featureDepth, dtype=np.uint8)
        knn = KNeighborsClassifier(n_neighbors=2)
        knn.fit(f_data, f_label)
        p_label = knn.predict(p_data)
        print(p_label.shape)
        print(p_data.shape)
        for i in range(p_data.shape[0]):
            DenseDepthMap[p_data[i][0], p_data[i][1]] = p_label[i]

    return DenseDepthMap







def main():
    input_frames = "sample_data/frames/"
    input_colmap = "sample_data/reconstruction/"
    output_folder = "output/"
    global global_h
    global global_w
    recon = ReadColMap.ReadColmap(input_colmap, input_frames)
    sparseDepthMap, global_h, global_w = Patches.initialTest(input_frames, recon)
    lbl, label_idx = Process_Label()
    sortPatches = constructPatches(lbl, label_idx, sparseDepthMap)
    DenseDepthMap = constructDenseDepthMap(sortPatches)
    DenseDepthMap = DenseDepthMap.astype('uint8')
    #CoarseDepthMap = Patches.constructCoarseDepthMap(sortPatches)
    DenseDepthMap = cv2.GaussianBlur(DenseDepthMap,(9,9),0)
    plt.imsave("DenseDepthMap.png", DenseDepthMap, cmap = 'viridis')
    #plt.imshow(DenseDepthMap, cmap='viridis')
    #plt.show()




if __name__ == '__main__':
    main()