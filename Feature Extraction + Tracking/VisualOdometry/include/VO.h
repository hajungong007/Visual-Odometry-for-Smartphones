#ifndef __VO__H__
#define __VO__H__

#include "iostream"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>
#include <string.h>
#include <boost/lambda/bind.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include "map"
#include <fstream>



namespace VO{

    class featureOperations{
        public:
            //! Constructor for capturing from the camera
            featureOperations(cv::Mat cameraMatrix);

            //! Constructor for processing from a directory
            featureOperations(std::string imgFolderPath,cv::Mat cameraMatrix);

            //! Function for extracting features from an image
            std::vector<cv::Point2f> detectFeatures(cv::Mat img);

            //! Function to track features in consecutive images
            bool trackFeatures(cv::Mat prevImg,cv::Mat currentImg,std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status);

            //! Function to compute trajectory
            bool buildTrajectory();

            //! Plot Trajectory
            bool plotTrajectory();

        private:

            //! Camera Intrinsic Matrix
            cv::Mat m_intrinsicMatrix;
    };

    class RobustMatcher{
        private:
            // pointer to the feature point detector object
            cv::Ptr<cv::FeatureDetector> detector;

            // pointer to the feature descriptor extractor object
            cv::Ptr<cv::DescriptorExtractor> extractor;

            // pointer to the matcher object
            cv::Ptr<cv::DescriptorMatcher> matcher;

            float ratio; // max ratio between 1st and 2nd NN
            bool refineF; // if true, will refine the F matrix
            double distance; // min distance to epipolar
            double confidence; // confidence level ( probability )

        public:
            RobustMatcher() : ratio(0.65f), refineF(true), confidence(0.99), distance(3.0){
                detector = new cv::OrbFeatureDetector();
                extractor = new cv::OrbDescriptorExtractor();
                matcher = new cv::BFMatcher();
            }

            // Set the feature detector
            void setFeatureDetector(cv::Ptr<cv::FeatureDetector>& detect){
                detector=detect;
            }

            // Set the descriptor extractor
            void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor>& desc){
                extractor = desc;
            }

            // Set the matcher
            void setDescriptorMatcher(cv::Ptr<cv::DescriptorMatcher>& match){
                matcher = match;
            }

            // Set confidence level
            void setConfidenceLevel(double conf) {
               confidence = conf;
            }

            //Set MinDistanceToEpipolar
            void setMinDistanceToEpipolar(double dist) {
               distance = dist;
            }

            //Set ratio
            void setRatio(float rat) {
               ratio = rat;
            }

            // Clear matches for which NN ratio is > than threshold
            // return the number of removed points
            // (corresponding entries being cleared,
            // i.e. size will be 0)
            int ratioTest(std::vector<std::vector<cv::DMatch> >& matches);

            // Insert symmetrical matches in symMatches vector
            void symmetryTest(const std::vector<std::vector<cv::DMatch> >& matches1,const std::vector<std::vector<cv::DMatch> >& matches2,std::vector<cv::DMatch>& symMatches);

            // Identify good matches using RANSAC
            // Return fundemental matrix
            cv::Mat ransacTest(const std::vector<cv::DMatch>& matches,const std::vector<cv::KeyPoint>& keypoints1,const std::vector<cv::KeyPoint>& keypoints2,std::vector<cv::DMatch>& outMatches);

            // Match feature points using symmetry test and RANSAC
            // returns fundamental matrix
            cv::Mat match(cv::Mat& image1,cv::Mat& image2, // input images
               std::vector<cv::DMatch>& matches,std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2); // output matches and keypoints

    };
}

cv::Mat readCameraIntrinsic(std::string pathToIntrinsic);

#endif
