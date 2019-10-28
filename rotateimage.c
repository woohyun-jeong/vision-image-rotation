#include <opencv\highgui.h>
#include <opencv\cv.h>

#define Thigh 100
#define Tlow 30

double getRotationAngle(IplImage* src);
void rotateImage(IplImage* src, IplImage* dst, double degree);

int main() {
	double degree;
	IplImage *sImg = cvLoadImage("text2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //ȸ�� ���� ����� ���� ��ų �̹���
	IplImage *smoothing = cvCreateImage(cvGetSize(sImg), sImg->depth, sImg->nChannels); //�޵�� ���͸� ���� ��Ų �̹���
	IplImage *canny = cvCreateImage(cvGetSize(sImg), sImg->depth, sImg->nChannels); //�ɴ� ������ ���� ��Ų �̹���
	IplImage *src = cvLoadImage("text2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //ȸ���� ���� ��ų �̹���
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels); //ȸ���� ������ �̹���

	//��Ʈ ���� �������Ÿ� ���� �޵�� ���� ����
	cvSmooth(sImg, smoothing, CV_MEDIAN, 3);
	//ĳ�� ���� ����
	cvCanny(smoothing, canny, Tlow, Thigh, 3);
	//���� ���
	degree = getRotationAngle(canny);
	//printf("%f", degree);
	//������ �̿��� ȸ��
	rotateImage(src, dst, degree);

	//cvNamedWindow("Source",CV_WINDOW_AUTOSIZE); //�����츦 �������
	//cvNamedWindow("Destination",CV_WINDOW_AUTOSIZE);
	cvShowImage("src", src);
	cvShowImage("Des", dst);

	
	cvWaitKey(0); //â�� �ٷ� �������� ���

	cvDestroyAllWindows(); //

}

double getRotationAngle(IplImage* src)
{
	// Only 1-Channel
	if (src->nChannels != 1)
		return 0;

	// ������ �� ����� �� �ֵ��� ��â
	cvDilate((IplImage*)src, (IplImage*)src);

	// ���念�� ����
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* seqLines;

	// Image�� ���������� seq�� ����(rho, theta)
	seqLines = cvHoughLines2((IplImage*)src, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 50, 30, 3); // ���ڵ� �����ؾ���

    // �������� �Ÿ��� ���ؼ� ���� �� ������ �������� �̹��� 0 or 90���� ȸ��
	double    longDistance = 0;    // ���� �� ���� �� ����
	int        longDistanceIndex = 0;    // ���� �� ���� �� ���� �ε���
	for (int i = 0; i < seqLines->total; i++) {
		CvPoint* lines = (CvPoint*)cvGetSeqElem(seqLines, i);
		double euclideanDistance;        // sequence�� ����� line���� Euclidean distance�� ����
		euclideanDistance = (lines[1].x - lines[0].x) * (lines[1].x - lines[0].x) + (lines[1].y - lines[0].y) * (lines[1].y - lines[0].y);
		euclideanDistance = sqrt(euclideanDistance);

		// ���� �� Euclidean distance�� ���� 
		if (longDistance < euclideanDistance) {
			longDistanceIndex = i;
			longDistance = euclideanDistance;
		}

	}
	// ȸ���� ���� ���
	CvPoint*    lines = (CvPoint*)cvGetSeqElem(seqLines, longDistanceIndex);
	int            dx = lines[1].x - lines[0].x;
	int            dy = lines[1].y - lines[0].y;
	double        rad = atan2((double)dx, (double)dy);    // ȸ���� ����(radian)
	double        degree = rad * 180 / CV_PI;             // ȸ���� ����(degree) ����

	if (dx > 0 && dy > 0) // ȸ���� ��������
		degree *= -1;
	if (degree>90)
		degree -= 90;

	// �޸� ����
	cvClearSeq(seqLines);
	cvReleaseMemStorage(&storage);

	return degree;
}

void rotateImage(IplImage* src, IplImage* dst, double degree)
{
	// Only 1-Channel
	if (src->nChannels != 1)
		return;

	CvPoint2D32f    centralPoint = cvPoint2D32f(src->width / 2, src->height / 2); // ȸ�� ������ ����(�̹����� �߽���)
	CvMat*            rotationMatrix = cvCreateMat(2, 3, CV_32FC1); // ȸ�� ���� ���

	// Rotation ���� ��� ���� �� ����(90������ ������ ������ ���� �����̹���(�ʿ�� ����))
	cv2DRotationMatrix(centralPoint, degree, 1, rotationMatrix);

	// Image Rotation
	cvWarpAffine(src, dst, rotationMatrix, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);

	// Memory ����
	cvReleaseMat(&rotationMatrix);
}

