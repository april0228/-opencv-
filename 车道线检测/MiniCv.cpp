#include "MiniCv.h"
#include <opencv2/opencv.hpp> 
#include <math.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_map>
using namespace std;
using namespace cv;

cv::Mat _getGray(const cv::Mat& src) { // GRAY = B * 0.114 + G * 0.587 + R * 0.299

	int h = src.rows;
	int w = src.cols;
	Mat dst(h, w, CV_8UC1);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			dst.at<uchar>(i, j) = 0;
			dst.at<uchar>(i, j) += 0.114 * src.at<Vec3b>(i, j)[0] ;
			dst.at<uchar>(i, j) += 0.587 * src.at<Vec3b>(i, j)[1] ;
			dst.at<uchar>(i, j) += 0.229 * src.at<Vec3b>(i, j)[2] ;
		}
	}

	return dst;
}

cv::Mat _revert(const cv::Mat& src) {

	Mat dst(src.rows, src.cols, CV_8UC1);
	for (int i = 0, m = src.rows; i < m; i++) {
		for (int j = 0, n = src.cols; j < n; j++) {
			dst.at<uchar>(i, j) = abs(255 - src.at<uchar>(i, j));
		}
	}

	return dst;
}

cv::Mat _lineGrayReform(const cv::Mat& src, const double k , const double b ) {

	Mat dst = src.clone();
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			dst.at<uchar>(i, j) =  k * src.at<uchar>(i, j) + b;
		}
	}

	return dst;
}

cv::Mat _logGrayReform(const cv::Mat& src, const double c) {

	Mat dst = src.clone();
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			dst.at<uchar>(i, j) = log(1 + src.at<uchar>(i, j))*c ;
		}
	}

	return src;
}

cv::Mat _gammaGrayReform(const cv::Mat& src, const double c,const double gamma) {

	Mat dst = src.clone();
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			int px = c * pow(src.at<uchar>(i, j), gamma);
			dst.at<uchar>(i, j) = px < 255 ? px : 255;
		}
	}
	return dst;
}

cv::Mat _boxFilter( const cv::Mat& src, const int ker_h , const int ker_w ,const  bool flag ) {

	Mat dst = src.clone(); 
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			// for point (i,j)  
			int sum = 0;
			for (int p = i - ker_h / 2; p <= i + ker_h / 2; p++) {
				for (int q = j - ker_w / 2; q <= j + ker_w / 2; q++) {
					
					if (p < 0 || q < 0 || p >= h || q >= w) continue; //边界0填充
					else {
						sum += src.at<uchar>(p, q);
					}
				}
			}
			if (flag) { //blur
				dst.at<uchar>(i, j) = sum / ( ker_h  *  ker_w );
			}
			else { // boxFilter
				dst.at<uchar>(i, j) = sum < 255 ? sum : 255;
			}
		}
	}

	return dst;

}

cv::Mat _blur(const cv::Mat& src,const int ker_h , const int ker_w ) {
	return _boxFilter(src, ker_h, ker_w, 1);
}

cv::Mat _gaussFilter(const cv::Mat& src,const int n, double sigma) {
	
	vector< vector<double> > kernel (n,vector<double>(n) );//卷积核
	int center = n / 2; //卷积核中心
	extern const double PI;
	double sum = 0;

	for (int i = 0; i < n; i++){ //获取卷积核
		for (int j = 0; j < n; j++) {

			kernel[i][j] = (1 / (2 * PI * sigma * sigma)) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
			sum += kernel[i][j];
		}
	}
	if (sum == 0) sum = 1;
	for (int i = 0; i < n; i++) {  //卷积核归一化
		for (int j = 0; j < n; j++) {
			kernel[i][j] /= sum;
		}
	}

	
	Mat dst = src.clone() ; //卷积操作
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			// for point (i,j)  
			int sum = 0;
			int u = 0, v = 0;
			for (int p = i - n / 2; p <= i + n / 2; p++) {
				for (int q = j - n / 2; q <= j + n / 2; q++) {

					if (p < 0 || q < 0 || p >= h || q >= w) {//边界0填充
						;
					}
					else {
						sum += (int)src.at<uchar>(p, q)* kernel[u][v];
					}
					v++;
					if (v == n) {
						v = 0;
						u++;
					}

				}
			}
			dst.at<uchar>(i, j) = sum ;
			
		}
	}

	return dst;
}

cv::Mat _medianFilter(const cv::Mat& src, const int ker_h,const int ker_w) {

	Mat dst = src.clone();
	vector<int> pxlist;
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			// for point (i,j)  
			for (int p = i - ker_h / 2; p <= i + ker_h / 2; p++) {
				for (int q = j - ker_w / 2; q <= j + ker_w / 2; q++) {

					if (p < 0 || q < 0 || p >= h || q >= w) continue; //边界0填充
					else {
						pxlist.push_back((int)src.at<uchar>(p, q));
					}
					
				}
			}
			sort(pxlist.begin(), pxlist.end());
			dst.at<uchar>(i, j) = pxlist[pxlist.size() / 2];
			pxlist.clear();
		}
	}

	return dst;
	
}

cv::Mat _minMaxFilter(const cv::Mat& src, const int ker_h ,const int ker_w ,const bool flag ) {

	Mat dst = src.clone();
	int minpx = INT_MAX;
	int maxpx = INT_MIN;
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			// for point (i,j) 
			for (int p = i - ker_h / 2; p <= i + ker_h / 2; p++) {
				for (int q = j - ker_w / 2; q <= j + ker_w / 2; q++) {

					if (p < 0 || q < 0 || p >= h || q >= w) continue; //边界0填充
					else {
						minpx = ( minpx <= (int)src.at<uchar>(p, q) ? minpx : (int)src.at<uchar>(p, q) );
						maxpx = ( maxpx >= (int)src.at<uchar>(p, q) ? maxpx : (int)src.at<uchar>(p, q) );
							
					}

				}
			}
			if (flag == 0) dst.at<uchar>(i, j) = minpx;
			else           dst.at<uchar>(i, j) = maxpx;
			//std::cout << "[" << i << "," << j << "]" << maxpx << endl;
			minpx = INT_MAX;
			maxpx = INT_MIN;
		}
	}

	return dst;
}

cv::Mat _myFilter(const cv::Mat& src, const vector<vector<double>> kernel,const double biss) {
	
	Mat dst = src.clone(); 
	int m = kernel.size();
	int n = kernel[0].size();
	//卷积操作
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			
			// for point (i,j)  
			int sum = 0;
			int u = 0, v = 0;
			for (int p = i - m / 2; p <= i + m / 2; p++) {
				for (int q = j - n / 2; q <= j + n / 2; q++) {

					if (p < 0 || q < 0 || p >= h || q >= w) {//边界0填充
						;
					}
					else {
						sum += src.at<uchar>(p, q) * kernel[u][v];
					}
					v++;
					if (v == n) {
						v = 0;
						u++;
					}

				}
			}
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;
			dst.at<uchar>(i, j) = sum + biss;

		}
	}

	return dst;
}

cv::Mat _lpFilter(const cv::Mat& src, const int flag , const double biss ) {

	extern const vector<vector<double>> lpKernel0;
	extern const vector<vector<double>> lpKernel1;

	if (flag == 0) {
		return _myFilter(src, lpKernel0, biss);
	}
	return _myFilter(src, lpKernel1, biss);
}

void _floodFill(cv::Mat& src, const int i, const int j, const int seedPx, const int newPx,const int flag ) {

	if (i >= 0 && j >= 0 && i < src.rows && j < src.cols && src.at<uchar>(i,j) == seedPx && src.at<uchar>(i, j) != newPx) {

		src.at<uchar>(i, j) = newPx;
		_floodFill(src, i - 1, j, seedPx, newPx,flag);
		_floodFill(src, i + 1, j, seedPx, newPx, flag);
		_floodFill(src, i , j - 1, seedPx, newPx, flag);
		_floodFill(src, i , j + 1, seedPx, newPx, flag);

		if (flag == 1) {  //8邻域
			_floodFill(src, i - 1, j + 1, seedPx, newPx, flag);
			_floodFill(src, i + 1, j + 1, seedPx, newPx, flag);
			_floodFill(src, i + 1, j - 1, seedPx, newPx, flag);
			_floodFill(src, i - 1, j - 1, seedPx, newPx, flag);
		}
	}

}

cv::Mat _thredShold(const cv::Mat& src, const int shred, const int maxVal ,const int flag) {

	Mat dst = src.clone();
	for (int i = 0, h = src.rows; i < h; i++) {
		for (int j = 0, w = src.cols; j < w; j++) {
			switch (flag) {
			case 0 : dst.at<uchar>(i, j) =  dst.at<uchar>(i, j) >= shred ?   maxVal : 0; break;
			case 1 : dst.at<uchar>(i, j) =  dst.at<uchar>(i, j) >= shred ?   0 : maxVal; break;
			case 2 : dst.at<uchar>(i, j) =  dst.at<uchar>(i, j) >= shred ?   shred : dst.at<uchar>(i, j)  ; break;
			case 3 : dst.at<uchar>(i, j) = dst.at<uchar>(i, j)  >= shred ?    0 : dst.at<uchar>(i, j); break;
			case 4 : dst.at<uchar>(i, j)  = dst.at<uchar>(i, j) <= shred ?    0 : dst.at<uchar>(i, j); break;
			}

		}
	}

	return dst;
}

cv::Mat _adaptiveThreshold(const cv::Mat& src, const int maxVal, const int flag ,const int blockSize, const int C) {

	    Mat dst = src.clone();

	    //均值阈值化
		for (int i = 0, h = src.rows; i < h; i++) {
			for (int j = 0, w = src.cols; j < w; j++) {
				// for point (i,j)  
				int sum = 0;
				for (int p = i - blockSize / 2; p <= i + blockSize / 2; p++) {
					for (int q = j - blockSize / 2; q <= j + blockSize / 2; q++) {
						if (p < 0 || q < 0 || p >= h || q >= w) continue; //边界0填充
						else {
							sum += src.at<uchar>(p, q);
						}
					}
				}

				sum = sum / (blockSize * blockSize) - C;  //sum aka thred
				if (flag == 0 && src.at<uchar>(i, j) >= sum)  dst.at<uchar>(i, j) = maxVal;
				else if (flag == 0 && src.at<uchar>(i, j) < sum)  dst.at<uchar>(i, j) = 0;
				else if (flag == 1 && src.at<uchar>(i, j) >= sum)  dst.at<uchar>(i, j) = 0;
				else if (flag == 1 && src.at<uchar>(i, j) < sum)  dst.at<uchar>(i, j) = maxVal;
			}
		
	    }

		return dst;
	
}

cv::Mat _candy( cv::Mat& src, const int low,const int hight) {

	//step 1 guassFilter;
	Mat dst = src.clone();
	src = _gaussFilter(src, 3, 1);
	

	//step 2 sabel 
	vector<vector<double>> sabel_x = {  //检测竖直方向
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};
	vector<vector<double>> sabel_y = {  //检测水平方向
		{-1,-2,-1},
		{0,0,0},
		{1,2,1}
	};
	Mat temp_x = _myFilter(src, sabel_x, 0); 
	Mat temp_y = _myFilter(src, sabel_y, 0);  
	
	//step 3 nms
	int h = src.rows;
	int w = src.cols;
	for (int y = 1; y < h-1; y++) {
		for (int x = 1; x < w-1; x++) {
			if (temp_x.at<uchar>(y, x) >= temp_x.at<uchar>(y, x - 1) &&
				temp_x.at<uchar>(y, x) >= temp_x.at<uchar>(y , x + 1) && temp_x.at<uchar>(y, x)!=0) { //检测竖直方向，那么nms作用与水平
				;//temp_x.at<uchar>(y, x) = 255;
			}
			else {
				temp_x.at<uchar>(y, x) = 0;
			}
		}
	}
	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			if (temp_x.at<uchar>(y, x) >= temp_x.at<uchar>(y-1, x ) &&
				temp_x.at<uchar>(y, x) >= temp_x.at<uchar>(y+1, x ) && temp_x.at<uchar>(y, x) != 0) { //检测水平方向，那么nms作用于竖直
				;//temp_x.at<uchar>(y, x) = 255;
			}
			else {
				temp_x.at<uchar>(y, x) = 0;
			}
		}
	}
	dst = abs(temp_x) + abs(temp_y);

	//step 4 threshold
	vector<vector<int>> feature(h, vector<int>(w));
	for (int y = 0; y < h ; y++) {
		for (int x = 0; x < w ; x++) {
			if (dst.at<uchar>(y, x) <= low) {
				feature[y][x] = -1; // not edge
				dst.at<uchar>(y, x) = 0;
			}
			else if (dst.at<uchar>(y, x) >= hight) {
				feature[y][x] = 1; //edge
			}
			else {
				feature[y][x] = 0; //not for sure
			}
			
		}
	}
	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			if (feature[y][x] == 0) {
				
				if (feature[y - 1][x] == 1 || feature[y + 1][x] == 1 ||
					feature[y][x - 1] == 1 || feature[y][x + 1] == 1 ||
					feature[y - 1][x - 1] == 1 || feature[y - 1][x + 1] == 1 ||
					feature[y + 1][x - 1] == 1 || feature[y + 1][x + 1] == 1) {
					feature[y][x] = 1;// keep
				}
				else {
					feature[y][x] = -1;
				}

			}
			

		}
	}
	for (int y = 1; y < h - 1; y++) {
		for (int x = 1; x < w - 1; x++) {
			if (feature[y][x] == 1) {
				dst.at<uchar>(y, x) = 255;
			}
			else {
				dst.at<uchar>(y, x) = 0;
			}
		}
	}

	return dst;
}

bool cmp_value(const pair<vector<int>, int> left, const pair<vector<int>, int> right) {
	return left.second < right.second;
}

cv::Mat _houghDetectline(const cv::Mat& src) {

	Mat img = src.clone();

	extern const double PI;
	const int h = img.rows;
	const int w = img.cols;
	const int n = h * w + 1000;

	map<vector<int>, int> vote;    //统计票数 vector0-r , vector1-theta
	
	int max = 0;
	for (int y = 0; y < h; y++) {     //霍夫空间计算票数
		for (int x = 0; x < w; x++) {
			if (img.at<uchar>(y, x) <= 20) { //减少计算量
				continue;
			}
			for (int theta = 20; theta <= 180; theta++) {
				if (abs(theta - 90) <= 10 ) {   //一般车道线都是前方，而不是垂直与前方
					continue;
				}
				int r = x * cos(theta * PI / 180) + y * sin(theta * PI / 180);
				vote[{r,theta}] ++;
				
				if (max <= vote[{r, theta}]) {
					max = vote[{r, theta}];
					//cout<<"vote = "<< max << " r = " << r << " theta = " << theta << endl;
				}
			}
		}
	}

	vector<vector<bool>> mask(h, vector<bool>(w));
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			mask[y][x] = 0;
		}
	}

	int line = 2;
	while (line--) {

		auto it = max_element(vote.begin(), vote.end(), cmp_value);
		int r = it->first[0]; //cout << r << "  ";
		int theta = it->first[1]; //cout << theta << endl;
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int tar = x * cos(theta * PI / 180) + y * sin(theta * PI / 180);
				if (tar == r) {
					mask[y][x] = 1;
				}
			}
		}

		int limit = 500;
		while (limit-- && !vote.empty()) {  //直线聚类

			auto p = max_element(vote.begin(), vote.end(), cmp_value);
			int pr = p->first[0];
			int pth = p->first[1];
			if (abs(r - pr) <= 60 || abs(theta - pth)<=5) {
				vote[{ pr,pth }] = 0;
			}
			else {
				break;
			}
		}
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (mask[y][x]) {
				img.at<uchar>(y, x) = 255;
			}
			else {
				img.at<uchar>(y, x) = 0;
			}
		}
	}
	

	//cv::imshow("hough", img);
	//cv::waitKey(0);
	return img;
}

cv::Mat _edgeDetective(const cv::Mat& src, const int flag) {

	Mat dst = src.clone();

	vector<vector<double>> sabel_x(3, vector<double>(3));
	vector<vector<double>> sabel_y;
	if (flag == 0) { // sobel
		 sabel_x = {  //检测竖直方向
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
		};
		 sabel_y = {  //检测水平方向
			{-1,-2,-1},
			{0,0,0},
			{1,2,1}
		};
	}
	if (flag == 1) { //robert
		sabel_x = {  //检测竖直方向
		{-1,0,1},
		{-1,0,1},
		{-1,0,1}
		};
		 sabel_y = {  //检测水平方向
			{-1,-1,-1},
			{0,0,0},
			{1,1,1}
		};
	}
	if (flag == 2) { //scahrr
		sabel_x = {  //检测竖直方向
		{-3,0,3},
		{-10,0,10},
		{-3,0,3}
		};
		 sabel_y = {  //检测水平方向
			{-3,-10,-3},
			{0,0,0},
			{3,10,3}
		};
	}

	Mat temp_x = _myFilter(src, sabel_x, 0);
	Mat temp_y = _myFilter(src, sabel_y, 0);

	dst = temp_x + temp_y;
	return dst;
}

cv::Mat _blance( cv::Mat& img) {

	Mat dst = img.clone();
	vector<int> gray(256,0);  //灰度分布
	vector<double> density(256, 0);//灰度密度
	vector<double> graySum(256, 0);//累计密度
	vector<int> blance(256, 0); //均衡化后的灰度值
	int h = img.rows;
	int w = img.cols;

	//统计每个灰度下的像素个数
	for (int i = 0; i < h; i++){
		uchar* p = img.ptr<uchar>(i);
		for (int j = 0; j < w; j++){
			gray[p[j]]++;
		}
	}

	//统计灰度频率
	for (int i = 0; i < 256; i++){
		density[i] = ((double)gray[i] / (h * w));
	}

	//计算累计密度
	graySum[0] = density[0];
	for (int i = 1; i < 256; i++){
		graySum[i] = graySum[i - 1] + density[i];
	}

	//重新计算均衡化后的灰度值，四舍五入。
	for (int i = 0; i < 256; i++){
		blance[i] = (uchar)(255 * graySum[i] + 0.5);
	}
	//直方图均衡化,更新原图每个点的像素值
	for (int i = 0; i < h; i++){
		uchar* p = dst.ptr<uchar>(i);
		for (int j = 0; j <w; j++){
			p[j] = blance[p[j]];
		}
	}

	return dst;
}