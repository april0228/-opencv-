#include <iostream>
#include <opencv2/opencv.hpp>
#include "MiniCv.h" 

using namespace std;
using namespace cv;


vector<vector<double>> ker = {		//用于检测y方向边缘的算子
	{-1,0,1},
	{-2,0,2},
	{-1,0,1}
};

static int n = 0;		//计数器，与show配合食用
void show(Mat& img) {		//显示图片

	n++;
	imshow("pic " + to_string(n), img); waitKey(0);
}

const string resultPath = "E:\\001sources\\carLineDataset\\result\\"; //保存结果文件路径
const string path = "E:\\001sources\\carLineDataset\\*.jpg";          //读取文件位置

static int num = 0;		//计数器，与deal配合食用
void deal(string loc) {		//核心处理单元

	Mat img = imread(loc);                //读取图片
	img = _getGray(img);                  //转灰度，减少后续运算
	img = _blur(img, 3, 3);               //平滑滤波，去除噪音
	img = _gammaGrayReform(img, 2, 1);    //有的照片上阴天拍摄或者其他原因导致教暗，需要增量（效果比直方图均衡化好）。
										  //至关重要，还关系到阈值选择的普适性

	int h = img.rows;                     //背景处理，减少噪音与运算量
	int w = img.cols;
	for (int i = 0; i < h * 2 / 5; i++) {
		for (int j = 0; j < w; j++) {
			img.at<uchar>(i, j) = 0;
		}
	}

	img = _lpFilter(img, 1);              //拉普拉斯锐化
	img = _myFilter(img, ker);            //检测y方向边缘

	for (int i = 0; i < h; i++) {         //两侧处理，减少噪音与运算量
		for (int j = 0; j < w; j++) {
			if ((double)i <= -0.8 * j + (double)600 || (double)i <= 0.8 * j - (double)440) {
				img.at<uchar>(i, j) = 0;
			}
		}
	}

	img = _thredShold(img, 120);           //阈值分割
	img = _houghDetectline(img);           //霍夫直线检测-核心

	Mat img0 = imread(loc);                //获取结果
	for (int i = h * 2 / 5; i < h - 1; i++) {
		for (int j = 1; j < w - 1; j++) {
			if (img.at<uchar>(i, j) == 255) {
				img0.at<Vec3b>(i, j)[0] = 0;
				img0.at<Vec3b>(i, j)[1] = 255;
				img0.at<Vec3b>(i, j)[2] = 0;
				img0.at<Vec3b>(i, j + 1)[0] = 0;
				img0.at<Vec3b>(i, j + 1)[1] = 255;
				img0.at<Vec3b>(i, j + 1)[2] = 0;
				img0.at<Vec3b>(i, j - 1)[0] = 0;
				img0.at<Vec3b>(i, j - 1)[1] = 255;
				img0.at<Vec3b>(i, j - 1)[2] = 0;
			}
		}
	}

	imwrite(resultPath + to_string(num)+ ".jpg", img0);  //写入结果
	num++;
	cout << "pic " << num << " done..." << endl;
	//show(img0);

}
int main() {


	vector<string> loc;
	glob(path, loc);      //将文件夹下所有图片路径保存到loc

	for (int i = 0, n = loc.size(); i < n; i++) {  //遍历处理，如果数据集庞大，更好的办法是开多线程处理
		deal(loc[i]);
	}
	cout << "all done !" << endl;

	return 0;
}



