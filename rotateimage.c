#include <opencv\highgui.h>
#include <opencv\cv.h>

#define Thigh 100
#define Tlow 30

double getRotationAngle(IplImage* src);
void rotateImage(IplImage* src, IplImage* dst, double degree);

int main() {
	double degree;
	IplImage *sImg = cvLoadImage("text2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //회전 각도 계산을 적용 시킬 이미지
	IplImage *smoothing = cvCreateImage(cvGetSize(sImg), sImg->depth, sImg->nChannels); //메디안 필터를 적용 시킨 이미지
	IplImage *canny = cvCreateImage(cvGetSize(sImg), sImg->depth, sImg->nChannels); //케니 엣지를 적용 시킨 이미지
	IplImage *src = cvLoadImage("text2.jpg", CV_LOAD_IMAGE_GRAYSCALE); //회전을 적용 시킬 이미지
	IplImage *dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels); //회전을 적용한 이미지

	//솔트 페퍼 잡음제거를 위한 메디안 필터 적용
	cvSmooth(sImg, smoothing, CV_MEDIAN, 3);
	//캐니 엣지 검출
	cvCanny(smoothing, canny, Tlow, Thigh, 3);
	//각도 계산
	degree = getRotationAngle(canny);
	//printf("%f", degree);
	//각도를 이용한 회전
	rotateImage(src, dst, degree);

	//cvNamedWindow("Source",CV_WINDOW_AUTOSIZE); //윈도우를 만들어줌
	//cvNamedWindow("Destination",CV_WINDOW_AUTOSIZE);
	cvShowImage("src", src);
	cvShowImage("Des", dst);

	
	cvWaitKey(0); //창을 바로 닫지말고 대기

	cvDestroyAllWindows(); //

}

double getRotationAngle(IplImage* src)
{
	// Only 1-Channel
	if (src->nChannels != 1)
		return 0;

	// 직선이 잘 검출될 수 있도록 팽창
	cvDilate((IplImage*)src, (IplImage*)src);

	// 저장영역 생성
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* seqLines;

	// Image의 직선영역을 seq에 저장(rho, theta)
	seqLines = cvHoughLines2((IplImage*)src, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / 180, 50, 30, 3); // 인자들 수정해야함

    // 직선들의 거리를 구해서 가장 긴 직선을 기준으로 이미지 0 or 90도로 회전
	double    longDistance = 0;    // 직선 중 가장 긴 길이
	int        longDistanceIndex = 0;    // 직선 중 가장 긴 길이 인덱스
	for (int i = 0; i < seqLines->total; i++) {
		CvPoint* lines = (CvPoint*)cvGetSeqElem(seqLines, i);
		double euclideanDistance;        // sequence에 저장된 line들의 Euclidean distance를 저장
		euclideanDistance = (lines[1].x - lines[0].x) * (lines[1].x - lines[0].x) + (lines[1].y - lines[0].y) * (lines[1].y - lines[0].y);
		euclideanDistance = sqrt(euclideanDistance);

		// 가장 긴 Euclidean distance를 저장 
		if (longDistance < euclideanDistance) {
			longDistanceIndex = i;
			longDistance = euclideanDistance;
		}

	}
	// 회전된 각도 계산
	CvPoint*    lines = (CvPoint*)cvGetSeqElem(seqLines, longDistanceIndex);
	int            dx = lines[1].x - lines[0].x;
	int            dy = lines[1].y - lines[0].y;
	double        rad = atan2((double)dx, (double)dy);    // 회전된 각도(radian)
	double        degree = rad * 180 / CV_PI;             // 회전된 각도(degree) 저장

	if (dx > 0 && dy > 0) // 회전된 각도보정
		degree *= -1;
	if (degree>90)
		degree -= 90;

	// 메모리 해제
	cvClearSeq(seqLines);
	cvReleaseMemStorage(&storage);

	return degree;
}

void rotateImage(IplImage* src, IplImage* dst, double degree)
{
	// Only 1-Channel
	if (src->nChannels != 1)
		return;

	CvPoint2D32f    centralPoint = cvPoint2D32f(src->width / 2, src->height / 2); // 회전 기준점 설정(이미지의 중심점)
	CvMat*            rotationMatrix = cvCreateMat(2, 3, CV_32FC1); // 회전 기준 행렬

	// Rotation 기준 행렬 연산 및 저장(90도에서 기울어진 각도를 빼야 본래이미지(필요시 수정))
	cv2DRotationMatrix(centralPoint, degree, 1, rotationMatrix);

	// Image Rotation
	cvWarpAffine(src, dst, rotationMatrix, CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS);

	// Memory 해제
	cvReleaseMat(&rotationMatrix);
}

