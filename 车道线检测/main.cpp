#include <iostream>
#include <opencv2/opencv.hpp>
#include "MiniCv.h" 

using namespace std;
using namespace cv;


vector<vector<double>> ker = {		//���ڼ��y�����Ե������
	{-1,0,1},
	{-2,0,2},
	{-1,0,1}
};

static int n = 0;		//����������show���ʳ��
void show(Mat& img) {		//��ʾͼƬ

	n++;
	imshow("pic " + to_string(n), img); waitKey(0);
}

const string resultPath = "E:\\001sources\\carLineDataset\\result\\"; //�������ļ�·��
const string path = "E:\\001sources\\carLineDataset\\*.jpg";          //��ȡ�ļ�λ��

static int num = 0;		//����������deal���ʳ��
void deal(string loc) {		//���Ĵ���Ԫ

	Mat img = imread(loc);                //��ȡͼƬ
	img = _getGray(img);                  //ת�Ҷȣ����ٺ�������
	img = _blur(img, 3, 3);               //ƽ���˲���ȥ������
	img = _gammaGrayReform(img, 2, 1);    //�е���Ƭ�����������������ԭ���½̰�����Ҫ������Ч����ֱ��ͼ���⻯�ã���
										  //������Ҫ������ϵ����ֵѡ���������

	int h = img.rows;                     //������������������������
	int w = img.cols;
	for (int i = 0; i < h * 2 / 5; i++) {
		for (int j = 0; j < w; j++) {
			img.at<uchar>(i, j) = 0;
		}
	}

	img = _lpFilter(img, 1);              //������˹��
	img = _myFilter(img, ker);            //���y�����Ե

	for (int i = 0; i < h; i++) {         //���ദ������������������
		for (int j = 0; j < w; j++) {
			if ((double)i <= -0.8 * j + (double)600 || (double)i <= 0.8 * j - (double)440) {
				img.at<uchar>(i, j) = 0;
			}
		}
	}

	img = _thredShold(img, 120);           //��ֵ�ָ�
	img = _houghDetectline(img);           //����ֱ�߼��-����

	Mat img0 = imread(loc);                //��ȡ���
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

	imwrite(resultPath + to_string(num)+ ".jpg", img0);  //д����
	num++;
	cout << "pic " << num << " done..." << endl;
	//show(img0);

}
int main() {


	vector<string> loc;
	glob(path, loc);      //���ļ���������ͼƬ·�����浽loc

	for (int i = 0, n = loc.size(); i < n; i++) {  //��������������ݼ��Ӵ󣬸��õİ취�ǿ����̴߳���
		deal(loc[i]);
	}
	cout << "all done !" << endl;

	return 0;
}



