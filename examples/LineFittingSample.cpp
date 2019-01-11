#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <random>

#include "GRANSAC.hpp"
#include "LineModel.hpp"

GRANSAC::VPFloat Slope(int x0, int y0, int x1, int y1)
{
	return (GRANSAC::VPFloat)(y1 - y0) / (x1 - x0);
}

void DrawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int LineWidth)
{
	GRANSAC::VPFloat slope = Slope(a.x, a.y, b.x, b.y);

	cv::Point p(0, 0), q(img.cols, img.rows);

	p.y = -(a.x - p.x) * slope + a.y;
	q.y = -(b.x - q.x) * slope + b.y;

	cv::line(img, p, q, color, LineWidth, cv::LINE_AA, 0);
}


class RANSAC_LINE
{
	GRANSAC::RANSAC<Line2DModel, 2>* _estimator;

public:
	RANSAC_LINE()
	{
		_estimator = new GRANSAC::RANSAC<Line2DModel, 2>;
	}
	~RANSAC_LINE()
	{
		if (_estimator != NULL) delete _estimator;
	}
public:
	int fit(std::vector<cv::Point>& pt_list, int thresh = 20, int iterations = 100 )
	{
		if (_estimator == NULL)
			return -1;
		_estimator->Initialize(thresh, iterations); // Threshold, iterations

		std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> CandPoints;
		for (auto pt = pt_list.begin(); pt != pt_list.end(); pt++)
		{
			std::shared_ptr<GRANSAC::AbstractParameter> CandPt = std::make_shared<Point2D>(pt->x, pt->y);
			CandPoints.push_back(CandPt);
		}


		int start = cv::getTickCount();
		_estimator->Estimate(CandPoints);
		int end = cv::getTickCount();
		std::cout << "RANSAC took: " << GRANSAC::VPFloat(end - start) / GRANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;

		return 0;

	}
	int get_points_in_line(std::vector<cv::Point>& pt_list)
	{
		if (_estimator == NULL) return -1;

		pt_list.clear();
		auto BestInliers = _estimator->GetBestInliers();
		if (BestInliers.size() > 0)
		{
			for (auto& Inlier : BestInliers)
			{
				auto RPt = std::dynamic_pointer_cast<Point2D>(Inlier);
				cv::Point Pt(floor(RPt->m_Point2D[0]), floor(RPt->m_Point2D[1]));
				pt_list.push_back(Pt);
			}
		}
		return 0;
	}

	int get_line(cv::Point& pt0, cv::Point& pt1)
	{
		if (_estimator == NULL) return -1;
		auto BestLine = _estimator->GetBestModel();
		if (BestLine)
		{
			auto BestLinePt1 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[0]);
			auto BestLinePt2 = std::dynamic_pointer_cast<Point2D>(BestLine->GetModelParams()[1]);
			if (BestLinePt1 && BestLinePt2)
			{
				cv::Point Pt1(BestLinePt1->m_Point2D[0], BestLinePt1->m_Point2D[1]);
				cv::Point Pt2(BestLinePt2->m_Point2D[0], BestLinePt2->m_Point2D[1]);
				pt0 = Pt1;
				pt1 = Pt2;
				return 0;
			}
		}
		return -1; 
	}
};



int main(int argc, char * argv[])
{
	if (argc != 1 && argc != 3)
	{
		std::cout << "[ USAGE ]: " << argv[0] << " [<Image Size> = 1000] [<nPoints> = 500]" << std::endl;
		return -1;
	}

	int Side = 1000;
	int nPoints = 500;
	if (argc == 3)
	{
		Side = std::atoi(argv[1]);
		nPoints = std::atoi(argv[2]);
	}

	cv::Mat Canvas(Side, Side, CV_8UC3);
	Canvas.setTo(255);

	// Randomly generate points in a 2D plane roughly aligned in a line for testing
	std::random_device SeedDevice;
	std::mt19937 RNG = std::mt19937(SeedDevice());

	std::uniform_int_distribution<int> UniDist(0, Side - 1); // [Incl, Incl]
	int Perturb = 25;
	std::normal_distribution<GRANSAC::VPFloat> PerturbDist(0, Perturb);
			 
	RANSAC_LINE ransac;
	std::vector<cv::Point> point_cluster;

	for (int i = 0; i < nPoints; ++i)
	{
		int Diag = UniDist(RNG);
		cv::Point Pt(floor(Diag + PerturbDist(RNG)), floor(Diag + PerturbDist(RNG)));
		cv::circle(Canvas, Pt, floor(Side / 100) + 3, cv::Scalar(0, 0, 0), 2);
		point_cluster.push_back(Pt);
	}

	ransac.fit(point_cluster);

	std::vector<cv::Point> points_in_line;
	int ret = ransac.get_points_in_line(points_in_line);
	if (points_in_line.size() > 0)
	{
		for (auto& pt : points_in_line)
		{
			cv::circle(Canvas, pt, 2, cv::Scalar(0, 255, 0), 1);
		}
	}

	cv::Point pt1, pt2;
	ret = ransac.get_line(pt1, pt2);
	if (ret == 0)
	{
		cv::line(Canvas, pt1, pt2, CV_RGB(255, 0, 0), 2);
	}

	while (true)
	{
		cv::imshow("RANSAC Example", Canvas);

		char Key = cv::waitKey(1);
		if (Key == 27)
			return 0;
		if (Key == ' ')
			cv::imwrite("LineFitting.png", Canvas);
	}

	return 0;
}

