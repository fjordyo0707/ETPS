/*
    ===== DESCRIPTION =====

    This is an implementation of the algorithms in

    Real-Time Coarse-to-fine Topologically Preserving Segmentation
    by Jian Yao, Marko Boben, Sanja Fidler, Raquel Urtasun

    published in CVPR 2015.

    http://www.cs.toronto.edu/~urtasun/publications/yao_etal_cvpr15.pdf

    Code is available from https://bitbucket.org/mboben/spixel

    ===== LICENSE =====

    Copyright(C) 2015  Jian Yao, Marko Boben, Sanja Fidler and Raquel Urtasun

    This program is free software : you can redistribute it and / or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include "stdafx.h"
#include "segengine.h"
#include "functions.h"
#include "utils.h"
#include <fstream>
#include <cstdlib>
#include "contrib/SGMStereo.h"

using namespace cv;
using namespace std;

void ConvertOCVToPNG(const Mat& ocvImg, png::image<png::rgb_pixel>& pngImg)
{
    pngImg.resize(ocvImg.cols, ocvImg.rows);

    for (int r = 0; r < ocvImg.rows; r++) {
        for (int c = 0; c < ocvImg.cols; c++) {
            const Vec3b& bgrColor = ocvImg.at<Vec3b>(r, c);
            pngImg.set_pixel(c, r, png::rgb_pixel(bgrColor[2], bgrColor[1], bgrColor[0]));
        }
    }
}

Mat ConvertFloatToOCV(int width, int height, const float* data)
{
    Mat1w result(height, width);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            result(y, x) = (ushort)(data[width*y + x] * 256.0f + 0.5);
        }
    }
    return result;
}

double SGMPreprocessing(const Mat& leftImage, const Mat& rightImage, Mat& dispImage)
{
    png::image<png::rgb_pixel> leftImageSGM, rightImageSGM;

    ConvertOCVToPNG(leftImage, leftImageSGM);
    ConvertOCVToPNG(rightImage, rightImageSGM);

    size_t width = leftImageSGM.get_width();
    size_t height = leftImageSGM.get_height();

    if (width != rightImageSGM.get_width() || height != rightImageSGM.get_height()) {
        dispImage = Mat1w();
        return 0.0;
    }

    float* dispImageFloat = (float*)malloc(width*height*sizeof(float));

    SGMStereo sgm;
    Timer t;

    sgm.compute(leftImageSGM, rightImageSGM, dispImageFloat);
    t.Stop();
    dispImage = ConvertFloatToOCV(width, height, dispImageFloat);

    free(dispImageFloat);
    
    return t.GetTimeInSec();
}

void ProcessFilesBatch(SPSegmentationParameters& params, const vector<string>& files, const string& fileDir)
{
    MkDir(fileDir + "out");
    MkDir(fileDir + "seg");

    int nProcessed = 0;
    double totalTime = 0.0;

    for (const string& f : files) {
        string fileName = fileDir + f;
        Mat image = imread(fileName, CV_LOAD_IMAGE_COLOR);

        if (image.rows == 0 || image.cols == 0) {
            cout << "Failed reading image '" << fileName << "'" << endl;
            continue;
        }

        cout << "Processing: " << fileName << endl;

        SPSegmentationEngine engine(params, image);

        engine.ProcessImage();

        engine.PrintPerformanceInfo();
        totalTime += engine.ProcessingTime();


        string outImage = ChangeExtension(fileDir + "out/" + f, "_sp.png");
        string outImageSeg = ChangeExtension(fileDir + "seg/" + f, ".png");

        double min, max;
        minMaxLoc(engine.GetSegmentation(), &min, &max);
        string filename = "./examples/mytest/label.xml";
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "label_number" << (int)max;
        fs <<"label_matrix"<< engine.GetSegmentation();
        fs.release(); 



        Mat img_seg;
        engine.GetSegmentation().convertTo(img_seg, CV_64F);
        imwrite(outImage, engine.GetSegmentedImage());
        imwrite(outImageSeg, img_seg);
        

        nProcessed++;
    }

    if (nProcessed > 1 && params.timingOutput) {
        cout << "Processed " << nProcessed << " files in " << totalTime << " sec. ";
        cout << "Average per image " << (totalTime / nProcessed) << " sec." << endl;
    }
}

// If dispPattern is empty, dispDir is the whole file name (in case we call this
// function to process only one file)
void ProcessFilesStereoBatch(SPSegmentationParameters& params, const vector<string>& files, const string& fileDir,
    const string& dispDir, const string& dispPattern)
{
    MkDir(fileDir + "out");
    MkDir(fileDir + "seg");
    MkDir(fileDir + "disp");

    int nProcessed = 0;
    double totalTime = 0.0;

    for (const string& f : files) {
        string fileName = fileDir + f;
        string dispFileName = dispPattern.empty() ? dispDir : ChangeExtension(dispDir + f, dispPattern);
        Mat image = imread(fileName, CV_LOAD_IMAGE_COLOR);
        Mat dispImage = imread(dispFileName, CV_LOAD_IMAGE_ANYDEPTH);

        if (image.rows == 0 || image.cols == 0) {
            cout << "Failed reading image '" << fileName << "'" << endl;
            continue;
        }
        if (dispImage.rows == 0 || dispImage.cols == 0) {
            cout << "Failed reading disparity image '" << dispFileName << "'" << endl;
            continue;
        }
        cout << "Processing: " << fileName << endl;

        SPSegmentationEngine engine(params, image, dispImage);

        engine.ProcessImageStereo();
        engine.PrintDebugInfoStereo();
        engine.PrintPerformanceInfo();
        totalTime += engine.ProcessingTime();

        string outImage = ChangeExtension(fileDir + "out/" + f, "_sp.png");
        string outImageSeg = ChangeExtension(fileDir + "seg/" + f, ".png");
        string outImageDisp = ChangeExtension(fileDir + "disp/" + f, ".png");

        imwrite(outImage, engine.GetSegmentedImage());
        imwrite(outImageSeg, engine.GetSegmentation());
        imwrite(outImageDisp, engine.GetDisparity());

        nProcessed++;
    }

    if (nProcessed > 1 && params.timingOutput) {
        cout << "Processed " << nProcessed << " files in " << totalTime << " sec. ";
        cout << "Average per image " << (totalTime / nProcessed) << " sec." << endl;
    }
}

void ProcessFilesStereoBatchSGM(SPSegmentationParameters& params, const vector<string>& files, const string& leftFileDir,
    const string& rightFileDir, bool rightIsName)
{
    MkDir(leftFileDir + "out");
    MkDir(leftFileDir + "seg");
    MkDir(leftFileDir + "disp");

    int nProcessed = 0;
    double totalTime = 0.0;

    for (const string& f : files) {
        string leftFileName = leftFileDir + f;
        string rightFileName = rightIsName ? rightFileDir : rightFileDir + f;
        Mat leftImage = imread(leftFileName, CV_LOAD_IMAGE_COLOR);
        Mat rightImage = imread(rightFileName, CV_LOAD_IMAGE_COLOR);

        if (leftImage.empty()) {
            cout << "Failed reading left image '" << leftImage << "'" << endl;
            continue;
        }
        if (rightImage.empty()) {
            cout << "Failed reading right image '" << rightImage << "'" << endl;
            continue;
        }
        cout << "Processing: " << leftFileName << "/" << rightFileName << endl;

        Mat dispImage;
        double sgmTime;
        
        sgmTime = SGMPreprocessing(leftImage, rightImage, dispImage);
        totalTime += sgmTime;

        if (dispImage.empty()) {
            cout << "Failed creating SGM image for '" << leftImage << "'/'" << rightImage << "' pair" << endl;
            continue;
        }

        if (params.timingOutput) {
            cout << "SGM processing time: " << sgmTime << " sec." << endl;
        }

        SPSegmentationEngine engine(params, leftImage, dispImage);

        engine.ProcessImageStereo();
        engine.PrintDebugInfoStereo();
        engine.PrintPerformanceInfo();
        totalTime += engine.ProcessingTime();

        string outImage = ChangeExtension(leftFileDir + "out/" + f, "_sp.png");
        string outImageSeg = ChangeExtension(leftFileDir + "seg/" + f, ".png");
        string outImageDisp = ChangeExtension(leftFileDir + "disp/" + f, ".png");

        imwrite(outImage, engine.GetSegmentedImage());
        imwrite(outImageSeg, engine.GetSegmentation());
        imwrite(outImageDisp, engine.GetDisparity());

        nProcessed++;
    }

    if (nProcessed > 1 && params.timingOutput) {
        cout << "Processed " << nProcessed << " files in " << totalTime << " sec. ";
        cout << "Average per image " << (totalTime / nProcessed) << " sec." << endl;
    }
}

void ProcessFilesStereoBatchSGM(SPSegmentationParameters& params, string leftDir, string rightDir,
    const string& filePattern)
{
    vector<string> files;

    FindFiles(leftDir, filePattern, files, false);
    EndDir(leftDir);
    EndDir(rightDir);
    ProcessFilesStereoBatchSGM(params, files, leftDir, rightDir, false);
}

void ProcessFileStereoSGM(SPSegmentationParameters& params, string leftFile, string rightFile,
    const string& filePattern)
{
    vector<string> files;
    string leftDir = FilePath(leftFile);
    string leftName = FileName(leftFile);

    files.push_back(leftName);
    ProcessFilesStereoBatchSGM(params, files, leftDir, rightFile, true);
}

void ProcessFilesBatch(SPSegmentationParameters& params, const string& dirName, const string& pattern)
{
    vector<string> files;
    string fileDir = dirName;

    FindFiles(fileDir, pattern, files, false);
    EndDir(fileDir);
    ProcessFilesBatch(params, files, fileDir);
}

void ProcessFile(SPSegmentationParameters& params, const string& file)
{
    vector<string> files;
    string fileDir = FilePath(file);
    string fileName = FileName(file);

    files.push_back(fileName);
    ProcessFilesBatch(params, files, fileDir);
}

void ProcessFilesStereoBatch(SPSegmentationParameters& params, const string& dirName, const string& pattern,
    const string& dispPattern)
{
    vector<string> files;
    string fileDir = dirName;

    FindFiles(fileDir, pattern, files, false);
    EndDir(fileDir);
    ProcessFilesStereoBatch(params, files, fileDir, fileDir, dispPattern);
}

void ProcessFileStereo(SPSegmentationParameters& params, const string& file, const string& dispFile)
{
    vector<string> files;
    string fileDir = FilePath(file);
    string fileName = FileName(file);

    files.push_back(fileName);
    ProcessFilesStereoBatch(params, files, fileDir, dispFile, "");
}

void ProcessFiles(const string& paramFile, const string& name1, const string& name2,
    const string& name3)
{
    SPSegmentationParameters params = ReadParameters(paramFile);

    if (params.randomSeed > 0) {
        srand(params.randomSeed);
    }

    if (params.stereo) {
        if (params.batchProcessing) {
            if (params.computeSGM) ProcessFilesStereoBatchSGM(params, name1, name2, name3);
            else ProcessFilesStereoBatch(params, name1, name2, name3);
        } else { // !batchProcessing
            if (params.computeSGM) ProcessFileStereoSGM(params, name1, name2, name3);
            else ProcessFileStereo(params, name1, name2);
        }
    } else {    // !stereo
        if (params.batchProcessing) ProcessFilesBatch(params, name1, name2);
        else ProcessFile(params, name1);
    }
}

int main(int argc, char* argv[])
{
    if (argc == 3) {
        ProcessFiles(argv[1], argv[2], "", "");
    } else if (argc == 4) {
        ProcessFiles(argv[1], argv[2], argv[3], "");
    } else if (argc == 5) {
        ProcessFiles(argv[1], argv[2], argv[3], argv[4]);
    } else {
        cout << argc << endl;
        for (int i = 1; i < argc; i++) {
            cout << argv[i] << "  ";
        }
        cout << endl;
        cout << "Real-Time Coarse-to-fine Topologically Preserving Segmentation, CVPR 2015" << endl;
        cout << "Built on: " << __DATE__ << " " << __TIME__ << endl;
        cout << "Usage (stereo = 0 & batchProcessing = 0): spixel config_file.yml file_name" << endl;
        cout << "   or (stereo = 0 & batchProcessing = 1): spixel config_file.yml file_dir file_pattern" << endl;
        cout << "   or (stereo = 1 & batchProcessing = 0 & computeSGM = 0): spixel config_file.yml file_name disparity_file_name" << endl;
        cout << "   or (stereo = 1 & batchProcessing = 1 & computeSGM = 0): spixel config_file.yml file_dir file_pattern disparity_extension" << endl;
        cout << "   or (stereo = 1 & batchProcessing = 0 & computeSGM = 1): spixel config_file.yml left_file_name right_file_name" << endl;
        cout << "   or (stereo = 1 & batchProcessing = 1 & computeSGM = 1): spixel config_file.yml left_file_dir right_file_dir file_pattern" << endl;
    }
    return 0;
}

