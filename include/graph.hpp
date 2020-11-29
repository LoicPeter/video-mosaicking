#ifndef GRAPH_INCLUDED
#define GRAPH_INCLUDED

#include <iostream>
#include <vector>
#include "unionfind.hpp"
#include <unordered_map>
#include <fstream>
#include <opencv2/core.hpp>


template <class Edge>
class Graph
{
    public:
    Graph();
    Graph(int nb_vertices);
    void remove_edge(int i, int j);
    void add_edge(int i, int j, const Edge& e);
    void edit_edge(int i, int j, const Edge& e);
    Edge get_edge(int i, int j);
    Edge* get_edge_ptr(int i, int j);
    bool is_there_edge(int i, int j) const;
    void save(const std::string& folder, const std::string& graph_filename) const;
    void load(const std::string& folder, const std::string& graph_filename);
    void load(const std::string& folder, const std::string& graph_filename, int nb_vertices);
    void labelConnectedComponents(std::vector<int>& connected_components, int& nb_cc);
    void keepBiggestConnectedComponent();
    int getNbVertices() const;

    // Public member for simplicity (iterations...)
    std::vector<std::unordered_map<int,Edge>> m_graph_relationships;

};

void convertToImage(cv::Mat& image, Graph<double>& graph);

void writeMatToFile(const cv::Mat& m, const std::string& filename);

void writeFloatMatToFile(const cv::Mat& m, const std::string& filename);

void loadMatFile(cv::Mat& m, const std::string& filename, int nb_rows, int nb_cols);

void loadFloatMatFile(cv::Mat& m, const std::string& filename, int nb_rows, int nb_cols);

void computeSumWeights(std::vector<double>& sum_weights, std::vector<int>& nb_neighbors, const Graph<double>& graph);

void computeAverageWeights(double& average_weights, const Graph<double>& graph);

template <class Edge>
Graph<Edge>::Graph(int nb_vertices)
{
    m_graph_relationships = std::vector<std::unordered_map<int,Edge>>(nb_vertices);
}

template <class Edge>
Graph<Edge>::Graph()
{
    m_graph_relationships = std::vector<std::unordered_map<int,Edge>>(1);
}

template <class Edge>
int Graph<Edge>::getNbVertices() const
{
    return m_graph_relationships.size();
}

template <class Edge>
void Graph<Edge>::add_edge(int i, int j, const Edge& e)
{
    std::pair<int,Edge> to_insert(j,e);
    m_graph_relationships[i].insert(to_insert);
}

template <class Edge>
void Graph<Edge>::edit_edge(int i, int j, const Edge& e)
{
   m_graph_relationships[i][j] = e;
}

template <class Edge>
Edge Graph<Edge>::get_edge(int i, int j)
{
   return m_graph_relationships[i][j];
}

template <class Edge>
Edge* Graph<Edge>::get_edge_ptr(int i, int j)
{
   return &m_graph_relationships[i][j];
}


template <class Edge>
void Graph<Edge>::remove_edge(int i, int j)
{
    m_graph_relationships[i].erase(j);
}

template <class Edge>
bool Graph<Edge>::is_there_edge(int i, int j) const
{
    return (m_graph_relationships[i].count(j)>0);
}

template <class Edge>
void Graph<Edge>::save(const std::string& folder, const std::string& graph_filename) const
{
    std::cout << "General save graph function not implemented" << std::endl;
}

template <>
inline void Graph<cv::Mat>::save(const std::string& folder, const std::string& graph_filename) const
{
    std::ofstream infile(folder + "/" + graph_filename);
    int nb_vertices = (int)m_graph_relationships.size();
    infile << nb_vertices << std::endl;
    for (int i=0; i<nb_vertices; ++i)
    {
        std::ostringstream oss_i;
        oss_i << i;
        for(const auto& n : m_graph_relationships[i])
        {
            std::string filename_pair;
            std::ostringstream oss_j;
            int j = n.first;
            oss_j << j;
            filename_pair = "pairwise_" + oss_i.str() + "_" + oss_j.str() + ".txt";
            infile << i << " " << j << " " << filename_pair << std::endl;
            writeMatToFile(n.second,folder + "/" + filename_pair);
        }

    }
    infile.close();
}




template <class Edge>
void Graph<Edge>::load(const std::string& folder, const std::string& graph_filename)
{
    std::cout << "General load graph function not implemented" << std::endl;
}

template <class Edge>
void Graph<Edge>::load(const std::string& folder, const std::string& graph_filename, int nb_vertices)
{
    std::cout << "General load graph function not implemented" << std::endl;
}



template <>
inline void Graph<cv::Mat>::load(const std::string& folder, const std::string& graph_filename, int nb_vertices)
{
    std::ifstream infile(folder + "/" + graph_filename);
    int nb_vertices_in_file;
    infile >> nb_vertices_in_file;
    if (((int)m_graph_relationships.size())!=nb_vertices)
        m_graph_relationships =  std::vector<std::unordered_map<int,cv::Mat>>(nb_vertices);
    int i, j;
    std::string filename;
    while (infile >> i >> j >> filename)
    {
        cv::Mat H;
        loadMatFile(H,folder + "/" + filename,3,3);
        if ((i<nb_vertices) && (j<nb_vertices))
            this->add_edge(i,j,H);
    }
    infile.close();
}


template <>
inline void Graph<cv::Mat>::load(const std::string& folder, const std::string& graph_filename)
{
    std::ifstream infile(folder + "/" + graph_filename);
    int nb_vertices_in_file;
    infile >> nb_vertices_in_file;
    infile.close();
    this->load(folder,graph_filename,nb_vertices_in_file);
}




template <class Edge>
void Graph<Edge>::labelConnectedComponents(std::vector<int>& connected_components, int& nb_cc)
{
    int nb_vertices = (int)m_graph_relationships.size();
    connected_components = std::vector<int>(nb_vertices,-1);

    // Create union find structure
    std::vector<UnionFindObject> union_find_structure(nb_vertices);
    for (int k=0; k<nb_vertices; ++k)
        union_find_structure[k].makeSet();
    for (int k=0; k<nb_vertices; ++k)
    {
        for (const auto& n : m_graph_relationships[k])
            union_find_structure[k].mergeWith(&union_find_structure[n.first]);
    }

    // Label connected components
    nb_cc = 0;
    for (int k=0; k<nb_vertices; ++k)
    {
        UnionFindObject *representative = union_find_structure[k].findRepresentative();
        int ind_representative = representative - &union_find_structure[0];
        if (connected_components[ind_representative]==(-1))
        {
            connected_components[ind_representative] = nb_cc;
            connected_components[k] = nb_cc;
            ++nb_cc;
        }
        else
            connected_components[k] = connected_components[ind_representative];
    }
}

template <class Edge>
void Graph<Edge>::keepBiggestConnectedComponent()
{
    // Identify connected components
    std::vector<int> connected_components;
    int nb_cc;
    this->labelConnectedComponents(connected_components,nb_cc);

    // Find biggest connected component
    std::vector<int> sizes_cc(nb_cc,0);
    int nb_vertices = (int)connected_components.size();
    for (int k=0; k<nb_vertices; ++k)
        ++sizes_cc[connected_components[k]];
    std::vector<int>::iterator it_max = std::max_element(sizes_cc.begin(),sizes_cc.end());
    int largest_cc = it_max - sizes_cc.begin();

    // Removes other connected components
    for (int k=0; k<nb_vertices; ++k)
    {
        if (connected_components[k]!=largest_cc)
            m_graph_relationships[k].clear();
    }
}


void getShortestPathToAllTargets(std::vector<double>& length_path, const Graph<double>& graph, int source);

void getDistanceBetweenAllVertices(cv::Mat& distance_matrix, const Graph<double>& graph);

void getDistanceBetweenAllVertices(cv::Mat& distance_matrix, cv::Mat& path_reconstruction, const Graph<double>& graph);


#endif // GRAPH_INCLUDED
