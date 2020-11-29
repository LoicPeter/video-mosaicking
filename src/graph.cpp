#include "graph.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>




template <>
void Graph<double>::save(const std::string& folder, const std::string& graph_filename) const
{
    std::ofstream infile(folder + "/" + graph_filename);
    int nb_vertices = (int)m_graph_relationships.size();
    infile << nb_vertices << std::endl;
    for (int i=0; i<nb_vertices; ++i)
    {
        for(const auto& n : m_graph_relationships[i])
            infile << i << " " << n.first << " " << n.second << std::endl;
    }
    infile.close();
}

void writeMatToFile(const cv::Mat& m, const std::string& filename)
{
    CV_Assert(m.depth() == CV_64F);
    std::ofstream fout(filename);

    if (!fout)
    {
        std::cout << "File Not Opened"<< std::endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout << m.at<double>(i,j) << "\t";
        }
        fout << std::endl;
    }

    fout.close();
}

void writeFloatMatToFile(const cv::Mat& m, const std::string& filename)
{
    CV_Assert(m.depth() == CV_32F);
    std::ofstream fout(filename);

    if (!fout)
    {
        std::cout << "File Not Opened"<< std::endl;  return;
    }

    for(int i=0; i<m.rows; i++)
    {
        for(int j=0; j<m.cols; j++)
        {
            fout << m.at<float>(i,j) << "\t";
        }
        fout << std::endl;
    }

    fout.close();
}


void loadMatFile(cv::Mat& m, const std::string& filename, int nb_rows, int nb_cols)
{
    m = cv::Mat(nb_rows,nb_cols,CV_64F);
    std::ifstream fin(filename);

    if (!fin)
    {
        std::cout << "File Not Opened"<< std::endl;  return;
    }

    for(int i=0; i<nb_rows; i++)
    {
        for(int j=0; j<nb_cols; j++)
        {
            fin >> m.at<double>(i,j);
        }
    }

    fin.close();
}

void loadFloatMatFile(cv::Mat& m, const std::string& filename, int nb_rows, int nb_cols)
{
    m = cv::Mat(nb_rows,nb_cols,CV_32F);
    std::ifstream fin(filename);

    if (!fin)
    {
        std::cout << "File Not Opened"<< std::endl;  return;
    }

    for(int i=0; i<nb_rows; i++)
    {
        for(int j=0; j<nb_cols; j++)
        {
            fin >> m.at<float>(i,j);
        }
    }

    fin.close();
}

void computeSumWeights(std::vector<double>& sum_weights, std::vector<int>& nb_edges, const Graph<double>& graph) // there can be several edges connecting two given vertices!
{
    int nb_vertices = (int)graph.m_graph_relationships.size();
    sum_weights = std::vector<double>(nb_vertices,0);
    nb_edges = std::vector<int>(nb_vertices,0);

    for (int i=0; i<nb_vertices; ++i)
    {
        for(const auto& n : graph.m_graph_relationships[i])
        {
            int j = n.first;
            double weight = n.second;
            sum_weights[i] += weight;
            sum_weights[j] += weight;
            ++nb_edges[i];
            ++nb_edges[j];
        }
    }

}

void computeAverageWeights(double& average_weights, const Graph<double>& graph) // there can be several edges connecting two given vertices!
{
    int nb_vertices = (int)graph.m_graph_relationships.size();
    average_weights = 0;
    int nb_edges = 0;
    for (int i=0; i<nb_vertices; ++i)
    {
        for(const auto& n : graph.m_graph_relationships[i])
        {
            double weight = n.second;
            average_weights += weight;
            ++nb_edges;
        }
    }

}

void convertToImage(cv::Mat& image, Graph<double>& graph)
{
    int nb_vertices = (int)graph.m_graph_relationships.size();
    image = cv::Mat(nb_vertices,nb_vertices,CV_32FC3,cv::Scalar(1,0,0));

    // Find max
    double max_weight(-1);
    for (int i=0; i<nb_vertices; ++i)
    {
        for (const auto& n : graph.m_graph_relationships[i])
        {
            double weight = n.second;
            if (weight>max_weight)
                max_weight = weight;
        }
    }

    // Fill image
    for (int i=0; i<nb_vertices; ++i)
    {
        for (const auto& n : graph.m_graph_relationships[i])
        {
            int j = n.first;
            double weight = n.second;
            image.at<cv::Vec3f>(i,j)[0] = 0;
            image.at<cv::Vec3f>(i,j)[2] = (float)(weight/max_weight);
        }
    }


}




void getShortestPathToAllTargets(std::vector<double>& length_path, const Graph<double>& graph, int source)
{
    int nb_vertices = (int)graph.m_graph_relationships.size();
    double infinity = 1000000000;
    length_path = std::vector<double>(nb_vertices,infinity);
    std::vector<bool> visited_vertex(nb_vertices,false);

    length_path[source] = 0;

    for (int it=0; it<nb_vertices; ++it)
    {
        //std::cout << it << " ";
        // Find non-visited vertex that has minimal distance to the source
        double min_distance = infinity+1;
        int ind_next_vertex(-1);
        for (int k=0; k<nb_vertices; ++k)
        {
            if ((visited_vertex[k]==false) && (length_path[k]<min_distance))
            {
                min_distance = length_path[k];
                ind_next_vertex = k;
            }
        }

        // Update neighbours
        for (const auto& n : graph.m_graph_relationships[ind_next_vertex])
        {
            int neighbor = n.first;
            double weight = n.second;
            if ((min_distance + weight) < length_path[neighbor])
                length_path[neighbor] = min_distance + weight;
        }

        // Mark current vertex as visited
        visited_vertex[ind_next_vertex] = true;
    }
}


void getDistanceBetweenAllVertices(cv::Mat& distance_matrix, const Graph<double>& graph)
{
 //   std::cout << "Start Floyds" << std::endl;

    int nb_vertices = (int)graph.m_graph_relationships.size();
    double infinity = 1000000000;
    distance_matrix = cv::Mat(nb_vertices,nb_vertices,CV_64F,cv::Scalar::all(infinity));
    for (int k=0; k<nb_vertices; ++k)
        distance_matrix.at<double>(k,k) = 0;
    for (int i=0; i<nb_vertices; ++i)
    {
        for (const auto& n : graph.m_graph_relationships[i])
        {
            int neighbor = n.first;
            double weight = n.second;
            distance_matrix.at<double>(i,neighbor) = weight;
            distance_matrix.at<double>(neighbor,i) = weight;
        }
    }

    for (int k=0; k<nb_vertices; ++k)
    {
 //       std::cout << k << " ";
        for (int i=0; i<nb_vertices; ++i)
        {
            for (int j=(i+1); j<nb_vertices; ++j)
            {
                double sum = distance_matrix.at<double>(i,k) + distance_matrix.at<double>(k,j);
                if (distance_matrix.at<double>(i,j) > sum)
                {
                    distance_matrix.at<double>(i,j) = sum;
                    distance_matrix.at<double>(j,i) = sum;
                }
            }
        }
    }

 //   std::cout << std::endl << "End Floyds" << std::endl;

}




void getDistanceBetweenAllVertices(cv::Mat& distance_matrix, cv::Mat& path_reconstruction, const Graph<double>& graph)
{
   // std::cout << "Start Floyds" << std::endl;

    int nb_vertices = (int)graph.m_graph_relationships.size();
    double infinity = 1000000000;
    distance_matrix = cv::Mat(nb_vertices,nb_vertices,CV_64F,cv::Scalar::all(infinity));
    path_reconstruction = cv::Mat(nb_vertices,nb_vertices,CV_64F,cv::Scalar::all(-1));
    for (int k=0; k<nb_vertices; ++k)
        distance_matrix.at<double>(k,k) = 0;
    for (int i=0; i<nb_vertices; ++i)
    {
        for (const auto& n : graph.m_graph_relationships[i])
        {
            int neighbor = n.first;
            double weight = n.second;
            distance_matrix.at<double>(i,neighbor) = weight;
            distance_matrix.at<double>(neighbor,i) = weight;
            path_reconstruction.at<double>(i,neighbor) = neighbor;
            path_reconstruction.at<double>(neighbor,i) = i;
        }
    }

    for (int k=0; k<nb_vertices; ++k)
    {
       // std::cout << k << " ";
        for (int i=0; i<nb_vertices; ++i)
        {
            for (int j=(i+1); j<nb_vertices; ++j)
            {
                double sum = distance_matrix.at<double>(i,k) + distance_matrix.at<double>(k,j);
                if (distance_matrix.at<double>(i,j) > sum)
                {
                    distance_matrix.at<double>(i,j) = sum;
                    distance_matrix.at<double>(j,i) = sum;
                    path_reconstruction.at<double>(i,j) = path_reconstruction.at<double>(i,k);
                    path_reconstruction.at<double>(j,i) = path_reconstruction.at<double>(j,k);
                }
            }
        }
    }

   // std::cout << std::endl << "End Floyds" << std::endl;

}






