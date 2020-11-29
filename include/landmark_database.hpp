#ifndef LANDMARK_DATABASE_H_INCLUDED
#define LANDMARK_DATABASE_H_INCLUDED

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "graph.hpp"
#include "utilities.hpp"


class LandmarkDatabase
{
    public:
    LandmarkDatabase();
    LandmarkDatabase(int nb_frames);
    int getNbFrames() const {return m_graph.getNbVertices();};
    void load(const std::string& filename);
    void save(const std::string& filename);
    void addToDatabase(int input_frame, int output_frame, const std::vector<cv::Point2f>& landmarks_input, const std::vector<cv::Point2f>& landmarks_output);
    void eraseFromDatabase(int input_frame, int output_frame);
    bool isPairInDatabase(int input_frame, int output_frame) const;
    void loadLandmarks(int input_frame, int output_frame, std::vector<cv::Point2f>& landmarks_input, std::vector<cv::Point2f>& landmarks_output);

    private:
   // std::vector<double> m_list_landmarks;   // size 6*nb_landmarks :  series of 6-uples (frame i, frame j, landmark input x, landmark input y, landmark output x, landmark output y)
    Graph<std::vector<double>> m_graph;
};


#endif
