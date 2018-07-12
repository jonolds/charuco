#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;

namespace {
	const char* about =
		"Calibration using a ChArUco board\n"
		"  To capture a frame for calibration, press 'c',\n"
		"  If input comes from video, press any key for next frame\n"
		"  To finish capturing, press 'ESC' key and calibration starts.\n";
	const char* keys =
		"{w        |       | Number of squares in X direction }"
		"{h        |       | Number of squares in Y direction }"
		"{sl       |       | Square side length (in meters) }"
		"{ml       |       | Marker side length (in meters) }"
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
		"{@outfile |<none> | Output file with calibrated camera parameters }"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{ci       | 0     | Camera id if input doesnt come from video (-v) }"
		"{dp       |       | File of marker detector parameters }"
		"{rs       | false | Apply refind strategy }"
		"{zt       | false | Assume zero tangential distortion }"
		"{a        |       | Fix aspect ratio (fx/fy) to this value }"
		"{pc       | false | Fix the principal point at the center }"
		"{sc       | false | Show detected chessboard corners after calibration }";
}

static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if(!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

static bool saveCameraParams(const string &filename, Size imageSize, float aspectRatio, int flags,
	const Mat &cameraMatrix, const Mat &distCoeffs, double totalAvgErr) {
	FileStorage fs(filename, FileStorage::WRITE);
	if(!fs.isOpened())
		return false;
	time_t tt;
	time(&tt);
	struct tm *t2 = localtime(&tt);
	char buf[1024];
	strftime(buf, sizeof(buf) - 1, "%c", t2);
	fs << "calibration_time" << buf;
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	if(flags & CALIB_FIX_ASPECT_RATIO) fs << "aspectRatio" << aspectRatio;
	if(flags != 0) {
		sprintf(buf, "flags: %s%s%s%s",
			flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
			flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
			flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
			flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
	}
	fs << "flags" << flags;
	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;
	fs << "avg_reprojection_error" << totalAvgErr;
	return true;
}

Mat charImg, Img;

int main() {
	int squaresX = 5;
	int squaresY = 7;
	float squareLength = 0.04;
	float markerLength = 0.02;
	string outputFile = "outFile.txt";
	bool showChessboardCorners = true;

	int calibrationFlags = 0;
	calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
	double aspectRatio = (16 / 9);
	//calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
	//calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;

	Ptr<aruco::DetectorParameters> detect = aruco::DetectorParameters::create();
	if(!readDetectorParameters("detectIn.yml", detect)) {cerr << "Err DetectIn"; return 0;}
	bool refineStrategy = false;

	VideoCapture cap = VideoCapture(0);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);
	int waitTime = 20;

	Ptr<aruco::Dictionary> dictionary =
		getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(10));

	// create charuco board object
	Ptr<aruco::CharucoBoard> charBoard = aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
	charBoard->draw(Size(700, 900), charImg, 50, 1);
	imwrite("charImg.png", charImg);
	Ptr<aruco::Board> board = charBoard.staticCast<aruco::Board>();

	// collect data from each frame
	vector< vector< vector< Point2f > > > allCorners;
	vector< vector< int > > allIds;
	vector< Mat > allImgs;
	Size imgSize;

	while(cap.grab()) {
		Mat image, imageCopy;
		cap.retrieve(image);

		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;

		// detect markers
		detectMarkers(image, dictionary, corners, ids, detect, rejected);

		// refind strategy to detect more markers
		if(refineStrategy) refineDetectedMarkers(image, board, corners, ids, rejected);

		// interpolate charuco corners
		Mat currentCharucoCorners, currentCharucoIds;
		if((int)ids.size() > 0)
			interpolateCornersCharuco(corners, ids, image, charBoard, currentCharucoCorners,
				currentCharucoIds);

		// draw results
		image.copyTo(imageCopy);
		if((int)ids.size() > 0) aruco::drawDetectedMarkers(imageCopy, corners);

		if(currentCharucoCorners.total() > 0)
			aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);

		putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
			Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);

		imshow("out", imageCopy);
		char key = (char)waitKey(waitTime);
		if(key == 27) break;
		if(key == 'c' && (int)ids.size() > 0) {
			cout << "Frame captured" << endl;
			allCorners.push_back(corners);
			allIds.push_back(ids);
			allImgs.push_back(image);
			imgSize = image.size();
			cout << imgSize << "\n";
		}
	}

	if((int)allIds.size() < 1) {
		cerr << "Not enough captures for calibration" << endl;
		return 0;
	}

	Mat cameraMatrix, distCoeffs;
	vector< Mat > rvecs, tvecs;

	if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
		cameraMatrix = Mat::eye(3, 3, CV_64F);
		cameraMatrix.at< double >(0, 0) = aspectRatio;
	}

	// prepare data for calibration
	vector< vector< Point2f > > allCornersConcatenated;
	vector< int > allIdsConcatenated;
	vector< int > markerCounterPerFrame;
	markerCounterPerFrame.reserve(allCorners.size());
	for(unsigned int i = 0; i < allCorners.size(); i++) {
		markerCounterPerFrame.push_back((int)allCorners[i].size());
		for(unsigned int j = 0; j < allCorners[i].size(); j++) {
			allCornersConcatenated.push_back(allCorners[i][j]);
			allIdsConcatenated.push_back(allIds[i][j]);
		}
	}

	// calibrate camera using aruco markers
	double arucoRepErr = calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated,
		markerCounterPerFrame, board, imgSize, cameraMatrix,
		distCoeffs, noArray(), noArray(), calibrationFlags);

	// prepare data for charuco calibration
	int nFrames = (int)allCorners.size();
	vector< Mat > allCharucoCorners;
	vector< Mat > allCharucoIds;
	vector< Mat > filteredImages;
	allCharucoCorners.reserve(nFrames);
	allCharucoIds.reserve(nFrames);

	for(int i = 0; i < nFrames; i++) {
		// interpolate using camera parameters
		Mat currentCharucoCorners, currentCharucoIds;
		interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charBoard,
			currentCharucoCorners, currentCharucoIds, cameraMatrix,
			distCoeffs);

		allCharucoCorners.push_back(currentCharucoCorners);
		allCharucoIds.push_back(currentCharucoIds);
		filteredImages.push_back(allImgs[i]);
	}

	if(allCharucoCorners.size() < 4) {
		cerr << "Not enough corners for calibration" << endl;
		return 0;
	}

	// calibrate camera using charuco
	double repError =
		calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charBoard, imgSize,
			cameraMatrix, distCoeffs, rvecs, tvecs, calibrationFlags);

	bool saveOk = saveCameraParams(outputFile, imgSize, (float)aspectRatio, calibrationFlags,
		cameraMatrix, distCoeffs, repError);
	if(!saveOk) {
		cerr << "Cannot save output file" << endl;
		return 0;
	}

	cout << "Rep Error: " << repError << endl;
	cout << "Rep Error Aruco: " << arucoRepErr << endl;
	cout << "Calibration saved to " << outputFile << endl;

	// show interpolated charuco corners for debugging
	if(showChessboardCorners) {
		for(unsigned int frame = 0; frame < filteredImages.size(); frame++) {
			Mat imageCopy = filteredImages[frame].clone();
			if((int)allIds[frame].size() > 0) {

				if(allCharucoCorners[frame].total() > 0) {
					aruco::drawDetectedCornersCharuco(imageCopy, allCharucoCorners[frame],
						allCharucoIds[frame]);
				}
			}

			imshow("out", imageCopy);
			char key = (char)waitKey(waitTime);
			if(key == 27) break;
		}
	}

	return 0;
}