#pragma once

#ifndef MINICV_H
#define MINICV_H

#include <opencv2/opencv.hpp> // �����ڶ�����д��
#include <vector>
using namespace std;

    const double PI = 3.1415926;
	const vector<vector<double>> lpKernel0 = { {0,1,0 }, { 1,-4,1 }, { 0,1,0 } }; //������˹����0
	const vector<vector<double>> lpKernel1 = { {1,1,1} , {1,-8,1},{1,1,1}  };     // ������˹����1

	cv::Mat _getGray(const cv::Mat& src);  //ת�Ҷ�ͼ

	cv::Mat _revert(const cv::Mat& src);   //���ط�ת��ָ��ø�Ƭ

	cv::Mat _lineGrayReform(const cv::Mat& src, const double k = 1, const double b = 0); //���ԻҶ�ת��

	cv::Mat _logGrayReform(const cv::Mat& src,const double c = 1);  //�����Ҷ�ת��

	cv::Mat _gammaGrayReform(const cv::Mat& src, const double c = 1, const double gamma = 1.0);//��ָ�Ҷ�ת��

	cv::Mat _boxFilter(const cv::Mat& src,const int ker_h = 3,const int ker_w = 3 ,const bool flag = 0); //�����˲�

	cv::Mat _blur(const cv::Mat& src, const int ker_h = 3, const int ker_w = 3); //��ֵ�˲�

	cv::Mat _gaussFilter(const cv::Mat& src,const int n, const double sigma = 1); //��˹�˲�

	cv::Mat _medianFilter(const cv::Mat& src, const int ker_h = 1, const int ker_w = 1); //��ֵ�˲�

	cv::Mat _minMaxFilter(const cv::Mat& src, const int ker_h = 1,const int ker_w = 1,const bool flag = 0); //��ֵ�˲������ͣ���ʴ����Ӧopencv api erode �� dilate

	cv::Mat _myFilter(const cv::Mat& src, const vector<vector<double>> kernel,const double biss = 0); //�Զ����˲���,������ʵ��candy��Sabel������

	cv::Mat _lpFilter(const cv::Mat& src, const int flag = 0, const double biss = 0); //������˹��

	void _floodFill(cv::Mat& src, const int i, const int j,const  int seedPx, const int newPx ,const int flag  = 0);   //��ˮ��䷨

	cv::Mat _thredShold(const cv::Mat& src, const int shred,const int maxVal = 255,const int flag = 0); //��ֵ��

	cv::Mat _adaptiveThreshold(const cv::Mat& src, const int maxVal = 255, const int flag = 0, const int blockSize = 3, const int C = 0); //����Ӧ��ֵ��

	cv::Mat _candy( cv::Mat& src, const int low,const int hight); //candy��Ե���

	cv::Mat _edgeDetective(const cv::Mat& src,const int flag = 0);      //������Ե�������

	bool cmp_value(const pair<vector<int>, int> left, const pair<vector<int>, int> right);//����������Ҫ���Զ���ȽϺ���
	cv::Mat _houghDetectline(const cv::Mat& img); //����ֱ�߼��

	cv::Mat _blance(cv::Mat img); //ֱ��ͼ���⻯

#endif // !MINICV_H

/*

Ϊ����opencvԭ�к������֣��ʼ����»��߿�ͷ��Ҳ�����Զ����Լ������ֿռ�

Mat ����ϴ󣬴������ýϿ죬��������������𲻴�Ч����

���� g(x,y)Ϊλ��Ϊ��x,y����ת��������أ�f(x,y)Ϊת��ǰ

getGray����ɫת�Ҷ�
	��˵��

revert�����ط�ת
    g(x,y) = 255 - f(x,y)

lineGrayReform �����ԻҶ�ת��
	g(x,y) = k*f(x,y) + b

logGrayReform �� �����Ҷ�ת��
	g(x,y) = log(1 + f(x,y) ) / c

gammaGrayReform : ��ָ�Ҷ�ת��
	g(x,y) = c*[f(x,y)]^gamma

boxFilter �� �����˲�
	src:����ͼ�� 
	ker_h : ����˸�,����
	ker_w : ����˿�����
	flag  : �Ƿ��һ������һ�������blur

blur �� ��ֵ�˲�
	�����ο�boxFilter

gaussFilter �� ��˹�˲�����ʵ�����Ǵ���Ȩֵ���˲�����ֵ�˲�Ĭ��Ȩֵ���
	g(x,y) = f(x,y)*guass(x,y)
	guass(x,y) = e^[-(x^2 + y^2)/ (2sigma^2)] / (2 PI * sigma^2)

medianFilter:��ֵ�˲������ӶȽϸߣ�Ҳ�ο�������ֵ�˲����͸��Ӷ�
	����������Ϊ�˳ߴ�

minMaxFilter :��ֵ�˲�
	0��Сֵ�˲�
	1���ֵ�˲�

myFilter �� �Զ����˲���
    ��Ҫ�����
	biss ƫ��

lpFilter �� ������˹�˲�
	�������ڲ�����myfilter
	flag = 0   {  {0,1,0} ,{1,-4,1} ,{0,1,0} }
	flag = 1   {  {1,1,1} , {1,-8,1},{1,1,1} }

floodFill �� ��ˮ���
	x,y�����ӵ�
	oldPx,newPx �ֱ�Ϊ�鷺ǰ������
	flag 0������1������
	���Եݹ�ʵ�֣�Ҳ����mask�˲�����ʽʵ�֣������Ի���ɨ����ʵ��

thredShold �� ��ֵ�ָ�
	shred ��ֵ��С
	flag 
	    0   g(x,y) = 255     f(x,y) > shred ;  g(x,y) = 0          f(x,y) < shred ;
		1   g(x,y) = 0       f(x,y) > shred ;  g(x,y) = 255        f(x,y) < shred ;
		2   g(x,y) = shred   f(x,y) > shred ;  g(x,y) = f(x,y)     f(x,y) < shred ;
		3   g(x,y) = 0       f(x,y) > shred ;  g(x,y) = f(x,y)     f(x,y) < shred ;
		4   g(x,y) = f(x,y)  f(x,y) > shred ;  g(x,y) = 0          f(x,y) < shred ;

adaptiveThreshold �� �Զ�����ֵ�ָ�
	��ֵ��ֵ
	blocksize �����С
	flagͬ�ϣ�����ֻ֧��ǰ����

candy �� candy��Ե���
	low �� hight �ֱ�Ϊcandy������һ���ͺ���ֵ�Ķ˵�

edgeDetective �� ������Ե�������

	0 sabel;
	1 robert;
	2 scharr;

blance : ֱ��ͼ���⻯
	��˵��
*/
