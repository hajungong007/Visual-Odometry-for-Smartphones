#include "VO.h"

VO::featureOperations::featureOperations(cv::Mat cameraMatrix){ // Loading images from a camera
    this->m_intrinsicMatrix=cameraMatrix;

}

VO::featureOperations::featureOperations(std::string imgFolderPath, cv::Mat cameraMatrix){ // Loading images from a directory

    this->m_intrinsicMatrix=cameraMatrix;
    boost::filesystem::path imgDir(imgFolderPath);
    boost::filesystem::directory_iterator end_iter;

    // Dataset sorted by Time

    typedef std::multimap<std::time_t, boost::filesystem::path> imageSet;
    imageSet imgStream;

    if ( boost::filesystem::exists(imgDir) && boost::filesystem::is_directory(imgDir))
    {
      for( boost::filesystem::directory_iterator dir_iter(imgDir) ; dir_iter != end_iter ; ++dir_iter)
      {
        if (boost::filesystem::is_regular_file(dir_iter->status()) )
        {
          imgStream.insert(imageSet::value_type(boost::filesystem::last_write_time(dir_iter->path()), *dir_iter));
        }
      }
    }


    std::multimap<std::time_t, boost::filesystem::path>::iterator it = imgStream.begin();
    std::multimap<std::time_t, boost::filesystem::path>::key_compare comparator = imgStream.key_comp();

    std::time_t last = imgStream.rbegin()->first;
    std::time_t bigBang = imgStream.begin()->first;


    // First two images
    std::string currentImg,nextImg;
    cv::Mat firstImage,secondImage;



    if(!comparator((*it).first,bigBang)){
        currentImg = ((*it).second).string();
//        std::cout<<"currentImg : "<<currentImg<<std::endl;
        *it++;
        nextImg = ((*it).second).string();
//        std::cout<<"nextImg : "<<nextImg<<std::endl;
        firstImage = cv::imread(currentImg,CV_LOAD_IMAGE_ANYCOLOR);
        secondImage = cv::imread(nextImg,CV_LOAD_IMAGE_ANYCOLOR);

        if ( !firstImage.data || !secondImage.data ) {
          std::cout<< " --(!) Error reading images " << std::endl;
        }

//        cv::imshow("Current Image",firstImage);
//        cv::imshow("Next Image",secondImage);
        *it++;
        std::vector<uchar> status;
        std::vector<cv::Point2f> points1,points2;
        points1 = this->detectFeatures(firstImage);
//        points2 = this->detectFeatures(secondImage);
        this->trackFeatures(firstImage,secondImage,points1,points2,status);
//        cv::waitKey(0);

        // Change these accordingly.

        double focal = 718.8560;
        cv::Point2d pp(607.1928, 185.2157);

        //recovering the pose and the essential matrixs

        // Will compute the R and T matrix, homography from Essential Matrix yahan and then call the build trajectory here.
        cv::Mat E, R, t, mask;
        E = cv::findEssentialMat(points2, points1, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
        cv::recoverPose(E, points2, points1, R, t, focal, pp, mask);

        cv::Mat prevImage = secondImage;
        cv::Mat currImage;
        std::vector<cv::Point2f> prevFeatures = points2;
        std::vector<cv::Point2f> currFeatures;

        cv::Mat R_f,t_f; // Final Rotation and Translation Vectors

        R_f = R.clone();
        t_f = t.clone();

//        std::cout<<t_f<<std::endl;
//        std::cout<<R_f<<std::endl;

        std::clock_t begin = clock();

        cv::namedWindow("Camera",CV_WINDOW_AUTOSIZE);
        cv::Mat trajectory = cv::Mat::zeros(600,600,CV_8UC3);




        // The image stream from the 2nd image onwards
        int fileIndex = 0;
        std::string remainingImgs;
        do {
            ++fileIndex;
            std::cout<<"Frame # : "<<fileIndex<<std::endl;
            remainingImgs= ((*it).second).string();
            currImage = cv::imread(remainingImgs,CV_LOAD_IMAGE_ANYCOLOR);

            this->trackFeatures(prevImage,currImage,prevFeatures,currFeatures,status);
            E = cv::findEssentialMat(currFeatures, prevFeatures, focal, pp, cv::RANSAC, 0.999, 1.0, mask);
            cv::recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

            ///////////////////////////////////
            double scale = 1;
            t_f = t_f + scale*(R_f*t);
            R_f = R*R_f;
            ///////////////////////////////////

            prevImage = currImage.clone();
            prevFeatures = currFeatures;

            cv::imshow("Camera", currImage);
            cv::waitKey(30);
            if(this->plotTrajectory(trajectory,t_f,R_f)){
                std::cout<<"Trajectory Plotted"<<std::endl;
            }


        } while ( comparator((*it++).first, last) );

    }

}

std::vector<cv::Point2f> VO::featureOperations::detectFeatures(cv::Mat img){
    cv::Mat distortCoeffs;
//    cv::Mat undistortedImg;

    cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);
    // Undistort

//    cv::undistort(img,undistortedImg,this->m_intrinsicMatrix,distortCoeffs);

//    cv::Mat imgForOrb = undistortedImg.clone();

    // Detect features on this image
    std::vector<cv::Point2f> pointsFAST;
    std::vector<cv::KeyPoint> keypoints_FAST;

    // FAST Detector
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    cv::FAST(img,keypoints_FAST,fast_threshold,nonmaxSuppression);
    cv::KeyPoint::convert(keypoints_FAST,pointsFAST,std::vector<int>());
    assert(pointsFAST.size() > 0);
    return pointsFAST;
}

bool VO::featureOperations::trackFeatures(cv::Mat prevImg, cv::Mat currentImg, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status){

    cv::Mat prevImg_gray,currentImg_gray;
    cv::cvtColor(prevImg,prevImg_gray,CV_BGR2GRAY);
    cv::cvtColor(currentImg,currentImg_gray,CV_BGR2GRAY);

    std::vector<float> err;
    cv::Size winSize=cv::Size(21,21);
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);


    cv::calcOpticalFlowPyrLK(prevImg_gray, currentImg_gray, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++)
       {  cv::Point2f pt = points2.at(i- indexCorrection);
          if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                if((pt.x<0)||(pt.y<0))	{
                  status.at(i) = 0;
                }
                points1.erase (points1.begin() + (i - indexCorrection));
                points2.erase (points2.begin() + (i - indexCorrection));
                indexCorrection++;
          }

       }


}


bool VO::featureOperations::plotTrajectory(cv::Mat trajectory,cv::Mat translation,cv::Mat rotation){

    cv::namedWindow("Trajectory",CV_WINDOW_AUTOSIZE);

    int x = int(translation.at<double>(0)) + 300;
    int y = int(translation.at<double>(2)) + 100;
    cv::circle(trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

    char text[100];
    int fontFace = CV_FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    cv::rectangle( trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", translation.at<double>(0), translation.at<double>(1), translation.at<double>(2));
    cv::putText(trajectory, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
    cv::imshow("Trajectory",trajectory);
    cv::waitKey(30);

    return true;

}

cv::Mat readCameraIntrinsic(std::string pathToIntrinsic){
   float cameraIntrinsicMatrix[3][3];
   cv::Mat cameraMatrix;
   std::string line;
   std::ifstream matrixReader(pathToIntrinsic.c_str());
   if (matrixReader.is_open())
     {
       for(int i=0;i<3;i++){
           for(int j=0;j<3;j++){
               matrixReader>>cameraIntrinsicMatrix[i][j];
           }
       }
     }
    else{
       std::cout << "Unable to open file"<<std::endl;
   }
   cameraMatrix=cv::Mat(3,3,CV_32FC1,&cameraIntrinsicMatrix);
   std::cout<<"\n ******Camera Intrinsic Matrix********** \n"<<std::endl;
//   std::cout<<cameraMatrix.at<float>(1,1)<<std::endl;


   return cameraMatrix;
}

//int VO::RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch> > &matches){
//    int removed=0;
//    // for all matches
//    for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator= matches.begin();matchIterator!= matches.end(); ++matchIterator) {
//          // if 2 NN has been identified
//          if (matchIterator->size() > 1) {
//              // check distance ratio
//              if ((*matchIterator)[0].distance/(*matchIterator)[1].distance > ratio) {
//                 matchIterator->clear(); // remove match
//                 removed++;
//              }
//          }
//          else { // does not have 2 neighbours
//              matchIterator->clear(); // remove match
//              removed++;
//          }
//       }
//       return removed;
//}

//void VO::RobustMatcher::symmetryTest(const std::vector<std::vector<cv::DMatch> > &matches1, const std::vector<std::vector<cv::DMatch> > &matches2, std::vector<cv::DMatch> &symMatches){
//    // for all matches image 1 -> image 2
//       for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1) {
//          // ignore deleted matches
//          if (matchIterator1->size() < 2)
//              continue;
//          // for all matches image 2 -> image 1
//          for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end(); ++matchIterator2) {
//              // ignore deleted matches
//              if (matchIterator2->size() < 2)
//                 continue;
//              // Match symmetry test
//              if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx && (*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
//                  // add symmetrical match
//                    symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance));
//                    break; // next match in image 1 -> image 2
//              }
//          }
//       }
//}

//cv::Mat VO::RobustMatcher::ransacTest(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &outMatches){

//      // Convert keypoints into Point2f
//      std::vector<cv::Point2f> points1, points2;
//      cv::Mat fundamental;

//      for (std::vector<cv::DMatch>::const_iterator it= matches.begin();it!= matches.end(); ++it) {

//          // Get the position of left keypoints
//          float x= keypoints1[it->queryIdx].pt.x;
//          float y= keypoints1[it->queryIdx].pt.y;
//          points1.push_back(cv::Point2f(x,y));
//          // Get the position of right keypoints
//          x= keypoints2[it->trainIdx].pt.x;
//          y= keypoints2[it->trainIdx].pt.y;
//          points2.push_back(cv::Point2f(x,y));
//       }

//      // Compute F matrix using RANSAC
//      std::vector<uchar> inliers(points1.size(),0);

//      if (points1.size()>0&&points2.size()>0){

//         cv::Mat fundamental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2), // matching points
//             inliers,       // match status (inlier or outlier)
//             CV_FM_RANSAC, // RANSAC method
//             distance,      // distance to epipolar line
//             confidence); // confidence probability

//         // extract the surviving (inliers) matches

//         std::vector<uchar>::const_iterator itIn= inliers.begin();
//         std::vector<cv::DMatch>::const_iterator itM= matches.begin();

//         // for all matches

//         for ( ;itIn!= inliers.end(); ++itIn, ++itM) {
//            if (*itIn) { // it is a valid match
//                outMatches.push_back(*itM);
//             }
//          }

//          if (refineF) { // The F matrix will be recomputed with all accepted matches. Convert keypoints into Point2f for final F computation

//             points1.clear();
//             points2.clear();

//             for (std::vector<cv::DMatch>::const_iterator it= outMatches.begin();it!= outMatches.end(); ++it) {
//                 // Get the position of left keypoints
//                 float x= keypoints1[it->queryIdx].pt.x;
//                 float y= keypoints1[it->queryIdx].pt.y;
//                 points1.push_back(cv::Point2f(x,y));
//                 // Get the position of right keypoints
//                 x= keypoints2[it->trainIdx].pt.x;
//                 y= keypoints2[it->trainIdx].pt.y;
//                 points2.push_back(cv::Point2f(x,y));
//             }

//             // Compute 8-point F from all accepted matches
//             if (points1.size()>0&&points2.size()>0){
//                fundamental= cv::findFundamentalMat(
//                   cv::Mat(points1),cv::Mat(points2), // matches
//                   CV_FM_8POINT); // 8-point method
//             }
//          }

//       }

//       return fundamental;
//}

//cv::Mat VO::RobustMatcher::match(cv::Mat &image1, cv::Mat &image2, std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2){

//        // 1a. Detection of the features
//       detector->detect(image1,keypoints1);
//       detector->detect(image2,keypoints2);
//       // 1b. Extraction of the descriptors
//       cv::Mat descriptors1, descriptors2;
//       extractor->compute(image1,keypoints1,descriptors1);
//       extractor->compute(image2,keypoints2,descriptors2);
//       // 2. Match the two image descriptors. Construction of the matcher cv::BruteForceMatcher<cv::L2<float>> matcher
//       // from image 1 to image 2
//       // based on k nearest neighbours (with k=2)

//       std::vector<std::vector<cv::DMatch> > matches1;
//       matcher->knnMatch(descriptors1,descriptors2,
//           matches1, // vector of matches (up to 2 per entry)
//           2);        // return 2 nearest neighbours

//        // from image 2 to image 1
//        // based on k nearest neighbours (with k=2)

//        std::vector<std::vector<cv::DMatch> > matches2;
//        matcher->knnMatch(descriptors2,descriptors1,
//           matches2, // vector of matches (up to 2 per entry)
//           2);        // return 2 nearest neighbours
//        // 3. Remove matches for which NN ratio is > than threshold

//        // clean image 1 -> image 2 matches
//        int removed= ratioTest(matches1);
//        // clean image 2 -> image 1 matches
//        removed= ratioTest(matches2);
//        // 4. Remove non-symmetrical matches
//        std::vector<cv::DMatch> symMatches;
//        symmetryTest(matches1,matches2,symMatches);
//        // 5. Validate matches using RANSAC
//        cv::Mat fundamental= ransacTest(symMatches,
//                    keypoints1, keypoints2, matches);
//        // return the found fundamental matrix
//        return fundamental;

//}
