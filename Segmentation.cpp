#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <string>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#define RED     "\033[31m"      /* Red */
#define RESET   "\033[0m"
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */

typedef pcl::PointXYZRGB PointT;

void viewCloud(pcl::PointCloud <PointT>::Ptr cloud1,std::string const &viewerName);
pcl::PointCloud<PointT>::Ptr filterNANS(pcl::PointCloud<PointT>::Ptr cloud);
pcl::PointCloud<PointT>::Ptr  extractInliers(pcl::PointCloud<PointT>::Ptr cloud_filtered,pcl::PointIndices::Ptr inliers,bool flag);
pcl::PointCloud<PointT>::Ptr cylinderSegmentation(pcl::PointCloud<PointT>::Ptr cloud);
pcl::PointCloud<PointT>::Ptr voxelGridFiltering(pcl::PointCloud<PointT>::Ptr cloud);
bool customCondition(const PointT& seedPoint, const PointT& candidatePoint, float squaredDistance);
void generateClusters(std::vector<pcl::PointIndices> clusters,pcl::PointCloud<PointT>::Ptr cloud,std::string const &clusterType);
void conditionalEucludieanClustering(pcl::PointCloud<PointT>::Ptr cloud);
void eucludieanClustering (pcl::PointCloud<PointT>::Ptr cloud);
pcl::PointCloud<PointT>::Ptr planarSegmentation(pcl::PointCloud<PointT>::Ptr cloud);
void regionBasedClustering(pcl::PointCloud<PointT>::Ptr cloud);
void colorBasedClustering(pcl::PointCloud<PointT>::Ptr cloud);
void minCutBasedClustering(pcl::PointCloud<PointT>::Ptr cloud);

void viewCloud(pcl::PointCloud <PointT>::Ptr cloud1,std::string const &viewerName){
  pcl::visualization::CloudViewer viewer (viewerName);
  viewer.showCloud (cloud1);
  while (!viewer.wasStopped ())
  {
    boost::this_thread::sleep (boost::posix_time::microseconds (100));
  }  
}


pcl::PointCloud<PointT>::Ptr filterNANS(pcl::PointCloud<PointT>::Ptr cloud){
  // Read in the cloud data
 // reader.read ("table_scene_mug_stereo_textured.pcd", *cloud);
  std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;
  pcl::PassThrough<PointT> pass;
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  // Build a passthrough filter to remove spurious NaNs
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, 1.5);
  pass.filter (*cloud_filtered);
  std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;
  return cloud_filtered;
}
pcl::PointCloud<PointT>::Ptr  extractInliers(pcl::PointCloud<PointT>::Ptr cloud_filtered,pcl::PointIndices::Ptr inliers,bool flag){
  pcl::ExtractIndices<PointT> extract;
  extract.setNegative(flag);//true for other than segmented
                            //false for only segmented
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers);
  extract.filter (*cloud_plane);
  return cloud_plane; 
}
 //In this function, will learn  segment arbitrary plane models from a given point cloud dataset.
pcl::PointCloud<PointT>::Ptr cylinderSegmentation(pcl::PointCloud<PointT>::Ptr cloud){
  pcl::PointCloud<PointT>::Ptr inlierPoints(new pcl::PointCloud<PointT>);
 
  // Object for storing the plane model coefficients.
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  // Create the segmentation object.
  pcl::SACSegmentation<PointT> segmentation;
  segmentation.setInputCloud(cloud);
  // Configure the object to look for a plane.
  segmentation.setModelType(pcl::SACMODEL_CYLINDER);
  // Use RANSAC method.
  segmentation.setMethodType(pcl::SAC_RANSAC);
  // Set the maximum allowed distance to the model.
  segmentation.setDistanceThreshold(0.01);
  // Enable model coefficient refinement (optional).
  segmentation.setOptimizeCoefficients(true);
  // Set minimum and maximum radii of the cylinder.
  segmentation.setRadiusLimits(0, 0.1); 
  pcl::PointIndices inlierIndices;
  segmentation.segment(inlierIndices, *coefficients); 
  if (inlierIndices.indices.size() == 0)
    std::cout << "Could not find any points that fitted the cylinder model." << std::endl;
  // Copy all inliers of the model to another cloud.
  else pcl::copyPointCloud<PointT>(*cloud, inlierIndices, *inlierPoints);
    return cloud;
}

pcl::PointCloud<PointT>::Ptr voxelGridFiltering(pcl::PointCloud<PointT>::Ptr cloud){
  pcl::VoxelGrid<PointT> sor;
  pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);
  return cloud_filtered;
}
bool customCondition(const PointT& seedPoint, const PointT& candidatePoint, float squaredDistance){
  // Do whatever you want here.
  if (candidatePoint.y < seedPoint.y)
    return false;
 
  return true;
}

void generateClusters(std::vector<pcl::PointIndices> clusters,pcl::PointCloud<PointT>::Ptr cloud,std::string const &clusterType){
  // For every cluster...
  int currentClusterNum = 1;
  for (std::vector<pcl::PointIndices>::const_iterator i = clusters.begin(); i != clusters.end(); ++i)
  {
    // ...add all its points to a new cloud...
    pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>);
    for (std::vector<int>::const_iterator point = i->indices.begin(); point != i->indices.end(); point++)
      cluster->points.push_back(cloud->points[*point]);
    cluster->width = cluster->points.size();
    cluster->height = 1;
    cluster->is_dense = true;
 
    // ...and save it to disk.
    if (cluster->points.size() <= 0)
      break;
    std::cout << "Cluster " << currentClusterNum << " has " << cluster->points.size() << " points." << std::endl;
    std::string fileName = clusterType+"_cluster" + boost::to_string(currentClusterNum) + ".pcd";
    pcl::io::savePCDFileASCII(fileName, *cluster);
 
    currentClusterNum++;
  }
}

void conditionalEucludieanClustering(pcl::PointCloud<PointT>::Ptr cloud){
// Conditional Euclidean clustering object.
  pcl::ConditionalEuclideanClustering<PointT> clustering;
  clustering.setClusterTolerance(0.02);
  clustering.setMinClusterSize(100);
  clustering.setMaxClusterSize(25000);
  clustering.setInputCloud(cloud);
  // Set the function that will be called for every pair of points to check.
  clustering.setConditionFunction(&customCondition);
  std::vector<pcl::PointIndices> clusters;
  clustering.segment(clusters);
  generateClusters(clusters,cloud,"conditionalEucludieanClustering");
} 

void eucludieanClustering (pcl::PointCloud<PointT>::Ptr cloud){
  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
  
  kdtree->setInputCloud(cloud);
 
  // Euclidean clustering object.
  pcl::EuclideanClusterExtraction<PointT> clustering;
  // Set cluster tolerance to 2cm (small values may cause objects to be divided
  // in several clusters, whereas big values may join objects in a same cluster).
  clustering.setClusterTolerance(0.02);
  // Set the minimum and maximum number of points that a cluster can have.
  clustering.setMinClusterSize(100);
  clustering.setMaxClusterSize(25000);
  clustering.setSearchMethod(kdtree);
  clustering.setInputCloud(cloud);
  std::vector<pcl::PointIndices> clusters;
  clustering.extract(clusters);
  generateClusters(clusters,cloud,"eucludieanClustering");  
}
//In this function, will learn  segment arbitrary plane models from a given point cloud dataset.
pcl::PointCloud<PointT>::Ptr planarSegmentation(pcl::PointCloud<PointT>::Ptr cloud){
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::ExtractIndices<PointT> extract;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    pcl::PointCloud <PointT>::Ptr cloud1(new pcl::PointCloud<PointT>);

    //cloud1->points.resize (inliers->indices.size ());
    cloud1=extractInliers(cloud,inliers,false);
    return cloud1;
}
void regionBasedClustering(pcl::PointCloud<PointT>::Ptr cloud){
  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
 
  kdtree->setInputCloud(cloud);
 
  // Estimate the normals.
  pcl::NormalEstimation<PointT, pcl::Normal> normalEstimation;
  normalEstimation.setInputCloud(cloud);
  normalEstimation.setRadiusSearch(0.03);
  normalEstimation.setSearchMethod(kdtree);
  normalEstimation.compute(*normals);
 
  // Region growing clustering object.
  pcl::RegionGrowing<PointT, pcl::Normal> clustering;
  clustering.setMinClusterSize(100);
  clustering.setMaxClusterSize(10000);
  clustering.setSearchMethod(kdtree);
  clustering.setNumberOfNeighbours(30);
  clustering.setInputCloud(cloud);
  clustering.setInputNormals(normals);
  // Set the angle in radians that will be the smoothness threshold
  // (the maximum allowable deviation of the normals).
  clustering.setSmoothnessThreshold(7.0 / 180.0 * M_PI); // 7 degrees.
  // Set the curvature threshold. The disparity between curvatures will be
  // tested after the normal deviation check has passed.
  clustering.setCurvatureThreshold(1.0);
 
  std::vector <pcl::PointIndices> clusters;
  clustering.extract(clusters);
  generateClusters(clusters,cloud,"regionBasedClustering");
 
}
void colorBasedClustering(pcl::PointCloud<PointT>::Ptr cloud){
// kd-tree object for searches.
  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>);
  kdtree->setInputCloud(cloud);
 
  // Color-based region growing clustering object.
  pcl::RegionGrowingRGB<PointT> clustering;
  clustering.setInputCloud(cloud);
  clustering.setSearchMethod(kdtree);
  // Here, the minimum cluster size affects also the postprocessing step:
  // clusters smaller than this will be merged with their neighbors.
  clustering.setMinClusterSize(100);
  // Set the distance threshold, to know which points will be considered neighbors.
  clustering.setDistanceThreshold(10);
  // Color threshold for comparing the RGB color of two points.
  clustering.setPointColorThreshold(6);
  // Region color threshold for the postprocessing step: clusters with colors
  // within the threshold will be merged in one.
  clustering.setRegionColorThreshold(5);
 
  std::vector <pcl::PointIndices> clusters;
  clustering.extract(clusters);
  generateClusters(clusters,cloud,"colorBasedClustering");
 } 
void minCutBasedClustering(pcl::PointCloud<PointT>::Ptr cloud){
  // Min-cut clustering object.
  pcl::MinCutSegmentation<PointT> clustering;
  clustering.setInputCloud(cloud);
  // Create a cloud that lists all the points that we know belong to the object
  // (foreground points). We should set here the object's center.
  pcl::PointCloud<PointT>::Ptr foregroundPoints(new pcl::PointCloud<PointT>());
  PointT point;
  point.x = 100.0;
  point.y = 100.0;
  point.z = 100.0;
  foregroundPoints->points.push_back(point);
  clustering.setForegroundPoints(foregroundPoints);
  // Set sigma, which affects the smooth cost calculation. It should be
  // set depending on the spacing between points in the cloud (resolution).
  clustering.setSigma(0.05);
  // Set the radius of the object we are looking for.
  clustering.setRadius(0.20);
  // Set the number of neighbors to look for. Increasing this also increases
  // the number of edges the graph will have.
  clustering.setNumberOfNeighbours(20);
  // Set the foreground penalty. It is the weight of the edges
  // that connect clouds points with the source vertex.
  clustering.setSourceWeight(0.6);
 
  std::vector <pcl::PointIndices> clusters;
  clustering.extract(clusters);
 
  std::cout << "Maximum flow is " << clustering.getMaxFlow() << "." << std::endl;
  generateClusters(clusters,cloud,"minCutBasedClustering");
}
int main (int argc, char** argv)
{
  bool filterd=true;

  //load point cloud data
  pcl::PointCloud <PointT>::Ptr cloud (new pcl::PointCloud <PointT>);
  if ( pcl::io::loadPCDFile <PointT> (argv[1], *cloud) == -1 )
  {
    std::cout << "Cloud reading failed." << std::endl;
    return -1;
  }
  pcl::PointCloud<PointT>::Ptr cloud1(new pcl::PointCloud <PointT>); 
  if (pcl::console::find_argument (argc, argv, "-filter") >= 0){
    cloud1=filterNANS(cloud);
  }
  else{
    filterd=false;
    cloud1=cloud;
    //viewCloud(cloud1,"viewer 2"); 
  }
  pcl::PointCloud<PointT>::Ptr cloud2;
  if (pcl::console::find_argument (argc, argv, "-pla") >= 0){
    cloud2=planarSegmentation(cloud1);
    viewCloud(cloud2,"viewer 2"); 
   }
  else if (pcl::console::find_argument (argc, argv, "-cyl") >= 0){
    cloud2=cylinderSegmentation(cloud1);
    viewCloud(cloud2,"viewer 2"); 
   }
  else if (pcl::console::find_argument (argc, argv, "-euc") >= 0){
      eucludieanClustering(cloud1);        
      }
  else if (pcl::console::find_argument (argc, argv, "-eucCon") >= 0){
      conditionalEucludieanClustering(cloud1);        
      }
  else if (pcl::console::find_argument (argc, argv, "-reg") >= 0){
      regionBasedClustering(cloud1);
      }
  else if (pcl::console::find_argument (argc, argv, "-regRGB") >= 0){
    colorBasedClustering(cloud1);
      }
  else if (pcl::console::find_argument (argc, argv, "-mincut") >= 0){
      minCutBasedClustering(cloud1);
      }
   else
  {
   // showHelp (argv[0]);
    std::cout<<BOLDRED<<"Follow read me file for instructions"<<RESET<<std::endl;
    exit (0);
  }
}
