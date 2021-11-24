#pragma once

#ifndef MINICV_H
#define MINICV_H

#include <opencv2/opencv.hpp> // 仅用于读入与写入
#include <vector>
using namespace std;

    const double PI = 3.1415926;
	const vector<vector<double>> lpKernel0 = { {0,1,0 }, { 1,-4,1 }, { 0,1,0 } }; //拉普拉斯算子0
	const vector<vector<double>> lpKernel1 = { {1,1,1} , {1,-8,1},{1,1,1}  };     // 拉普拉斯算子1

	cv::Mat _getGray(const cv::Mat& src);  //转灰度图

	cv::Mat _revert(const cv::Mat& src);   //像素翻转，指获得负片

	cv::Mat _lineGrayReform(const cv::Mat& src, const double k = 1, const double b = 0); //线性灰度转换

	cv::Mat _logGrayReform(const cv::Mat& src,const double c = 1);  //对数灰度转换

	cv::Mat _gammaGrayReform(const cv::Mat& src, const double c = 1, const double gamma = 1.0);//幂指灰度转换

	cv::Mat _boxFilter(const cv::Mat& src,const int ker_h = 3,const int ker_w = 3 ,const bool flag = 0); //方框滤波

	cv::Mat _blur(const cv::Mat& src, const int ker_h = 3, const int ker_w = 3); //均值滤波

	cv::Mat _gaussFilter(const cv::Mat& src,const int n, const double sigma = 1); //高斯滤波

	cv::Mat _medianFilter(const cv::Mat& src, const int ker_h = 1, const int ker_w = 1); //中值滤波

	cv::Mat _minMaxFilter(const cv::Mat& src, const int ker_h = 1,const int ker_w = 1,const bool flag = 0); //最值滤波，膨胀，腐蚀，对应opencv api erode 与 dilate

	cv::Mat _myFilter(const cv::Mat& src, const vector<vector<double>> kernel,const double biss = 0); //自定义滤波器,可以自实现candy，Sabel等算子

	cv::Mat _lpFilter(const cv::Mat& src, const int flag = 0, const double biss = 0); //拉普拉斯锐化

	void _floodFill(cv::Mat& src, const int i, const int j,const  int seedPx, const int newPx ,const int flag  = 0);   //漫水填充法

	cv::Mat _thredShold(const cv::Mat& src, const int shred,const int maxVal = 255,const int flag = 0); //阈值化

	cv::Mat _adaptiveThreshold(const cv::Mat& src, const int maxVal = 255, const int flag = 0, const int blockSize = 3, const int C = 0); //自适应阈值化

	cv::Mat _candy( cv::Mat& src, const int low,const int hight); //candy边缘检测

	cv::Mat _edgeDetective(const cv::Mat& src,const int flag = 0);      //其他边缘检测算子

	bool cmp_value(const pair<vector<int>, int> left, const pair<vector<int>, int> right);//霍夫检测中需要的自定义比较函数
	cv::Mat _houghDetectline(const cv::Mat& img); //霍夫直线检测

	cv::Mat _blance(cv::Mat img); //直方图均衡化

#endif // !MINICV_H

/*

为了与opencv原有函数区分，故加了下划线开头，也可以自定义自己的名字空间

Mat 对象较大，传入引用较快，其他基本类对象差别不大效率上

以下 g(x,y)为位置为（x,y）处转换后的像素，f(x,y)为转换前

getGray：彩色转灰度
	无说明

revert：像素翻转
    g(x,y) = 255 - f(x,y)

lineGrayReform ：线性灰度转换
	g(x,y) = k*f(x,y) + b

logGrayReform ： 对数灰度转换
	g(x,y) = log(1 + f(x,y) ) / c

gammaGrayReform : 幂指灰度转换
	g(x,y) = c*[f(x,y)]^gamma

boxFilter ： 方框滤波
	src:输入图像 
	ker_h : 卷积核高,奇数
	ker_w : 卷积核宽，奇数
	flag  : 是否归一化，归一化后就是blur

blur ： 均值滤波
	参数参考boxFilter

gaussFilter ： 高斯滤波器，实际上是带有权值的滤波，均值滤波默认权值相等
	g(x,y) = f(x,y)*guass(x,y)
	guass(x,y) = e^[-(x^2 + y^2)/ (2sigma^2)] / (2 PI * sigma^2)

medianFilter:中值滤波，复杂度较高，也参考快速中值滤波降低复杂度
	后两个参数为核尺寸

minMaxFilter :最值滤波
	0最小值滤波
	1最大值滤波

myFilter ： 自定义滤波器
    需要传入核
	biss 偏置

lpFilter ： 拉普拉斯滤波
	本质是内部调用myfilter
	flag = 0   {  {0,1,0} ,{1,-4,1} ,{0,1,0} }
	flag = 1   {  {1,1,1} , {1,-8,1},{1,1,1} }

floodFill ： 洪水填充
	x,y是种子点
	oldPx,newPx 分别为洪泛前后像素
	flag 0四邻域，1八邻域
	可以递归实现，也可以mask滤波器方式实现，还可以基于扫描线实现

thredShold ： 阈值分割
	shred 阈值大小
	flag 
	    0   g(x,y) = 255     f(x,y) > shred ;  g(x,y) = 0          f(x,y) < shred ;
		1   g(x,y) = 0       f(x,y) > shred ;  g(x,y) = 255        f(x,y) < shred ;
		2   g(x,y) = shred   f(x,y) > shred ;  g(x,y) = f(x,y)     f(x,y) < shred ;
		3   g(x,y) = 0       f(x,y) > shred ;  g(x,y) = f(x,y)     f(x,y) < shred ;
		4   g(x,y) = f(x,y)  f(x,y) > shred ;  g(x,y) = 0          f(x,y) < shred ;

adaptiveThreshold ： 自定义阈值分割
	阈值均值
	blocksize 邻域大小
	flag同上，不过只支持前两种

candy ： candy边缘检测
	low 与 hight 分别为candy检测最后一步滞后阈值的端点

edgeDetective ： 其他边缘检测算子

	0 sabel;
	1 robert;
	2 scharr;

blance : 直方图均衡化
	无说明
*/
