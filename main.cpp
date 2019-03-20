#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

//Global varialbes 
vector<string> classNames;
vector<Scalar> colours;
//scalar is a template class for a 4-element vector

//Thresholding Parameters
float confidenceThreshold = 0.4;
float maskThreshold = 0.4;

void postprocess(Mat& M_image, const vector<Mat>& vM_outs);
void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask); 


int main(){
//load the coco class names
string classNameFile ="mscoco_labels.names";
//Common Objects in Context (COCO) labels

ifstream NameFile(classNameFile);
string line;
while(getline(NameFile,line)) classNames.push_back(line);

//create a colours vectors, storing diferent colours used later 
//for assigning classes to coloured boundingboxes
string coloursFile = "colours.txt";
ifstream coloursF(coloursFile);
while(getline(coloursF,line)) {
double r,g,b;
char* pEnd;
r = strtod(line.c_str(), &pEnd);
g = strtod(pEnd, &pEnd);
b = strtod(pEnd, NULL);
colours.push_back(Scalar(r,g,b,255.0));
}

//Setting up DNN in opencv using weights and graph  
string textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
string modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

Net net = readNetFromTensorflow(modelWeights, textGraph);
net.setPreferableBackend(DNN_BACKEND_OPENCV);
net.setPreferableTarget(DNN_TARGET_CPU);

//load image for processing

Mat M_image;

M_image = imread("FCMViewofTraffic.jpg");

if(! M_image.data){
	cout<<"Could not open the image"<<endl;
	return -1;
	}

Mat M_blob;
blobFromImage(M_image, M_blob, 1.0, Size(M_image.cols, M_image.rows),Scalar(),true,false);
//NN need the image to be in blob format
/*
Binary Large Object refers to a group of connected pixels in a binary image 

Blob representation is converting each BLOB into a few representatie number.
Ignoring non relevant data.
*/

net.setInput(M_blob);
vector<string> outNames;

outNames.push_back("detection_out_final"); //predicting bounding boxes

outNames.push_back("detection_masks"); //predicting object masks

vector<Mat> vM_outs;

net.forward(vM_outs,outNames);

postprocess(M_image, vM_outs);

imshow("The output", M_image);

imwrite("FCMViewofTraffic_output.jpg",M_image);
waitKey(0);

return 0;
}


void postprocess(Mat& M_image, const vector<Mat>& vM_outs){
/*  The purpose of this function is to extract the bounding box and masks 
    for the objects detected 
    M_image = M_Blob
*/
    Mat M_BoundingBox = vM_outs[0];
    Mat M_Masks = vM_outs[1];

    /* Information aout the masks
	size of masks: NxCxHxW
	N - number of dectected boxes
	C - number of classes (excluding background)
	HxW - segmentation shape
	*/

    const int numDetections = M_BoundingBox.size[2]; //size in x 12
    const int numClasses = M_Masks.size[1]; //

	M_BoundingBox = M_BoundingBox.reshape(1,M_BoundingBox.total()/7);

    for (int i =0; i<numDetections; i++){

        float score = M_BoundingBox.at<float>(i,2);

        if(score >confidenceThreshold){
            //bounding box properties 
            int classId = static_cast<int>(M_BoundingBox.at<float>(i, 1));
            int left = static_cast<int>(M_image.cols * M_BoundingBox.at<float>(i, 3));
            int top = static_cast<int>(M_image.rows * M_BoundingBox.at<float>(i, 4));
            int right = static_cast<int>(M_image.cols * M_BoundingBox.at<float>(i, 5));
            int bottom = static_cast<int>(M_image.rows * M_BoundingBox.at<float>(i, 6));

            left = max(0, min(left, M_image.cols - 1));
            top = max(0, min(top, M_image.rows - 1));
            right = max(0, min(right, M_image.cols - 1));
            bottom = max(0, min(bottom, M_image.rows - 1));
            Rect box = Rect(left, top, right - left + 1, bottom - top + 1);

            //Mask for the object
            Mat objectMask(M_Masks.size[2], M_Masks.size[3],CV_32F, M_Masks.ptr<float>(i,classId));
            drawBox(M_image, classId, score, box, objectMask);

        }
    }
   
}

void drawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x+box.width, box.y+box.height), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classNames.empty())
    {
        CV_Assert(classId < (int)classNames.size());
        label = classNames[classId] + ":" + label;

    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
    rectangle(frame, Point(box.x, box.y - round(1.5*labelSize.height)), Point(box.x + round(1.5*labelSize.width), box.y + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
    Scalar colour = colours[classId%colours.size()];
    
    // Resize the mask, threshold, color and apply it on the image

    resize(objectMask, objectMask, Size(box.width, box.height));
        //error between b2 nd b6;
    //let's check size of frame

    Mat mask = (objectMask > maskThreshold);
    Mat coloredRoi = (0.3 * colour + 0.7 * frame(box)); //error here
    coloredRoi.convertTo(coloredRoi, CV_8UC3);
    

    //Draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mask.convertTo(mask, CV_8U);
    findContours(mask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(coloredRoi, contours, -1, colour, 5, LINE_8, hierarchy, 100);
    coloredRoi.copyTo(frame(box), mask);

}

